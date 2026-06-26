import json
import requests

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_THRESHOLDS

ALLOW = "ALLOW"
REVIEW = "HUMAN REVIEW"
# NOTE: there is intentionally no automated REMOVE decision in v3. The system
# never removes content on its own; the highest-confidence matches are routed to
# urgent human review. A human can still choose REMOVE from the dashboard.


def _build_prompt(video_text, matched_seed_text, composite_score):
    return (
        f"You are a Trust and Safety content reviewer.\n\n"
        f"An automated system flagged a video as a near-duplicate of known harmful content "
        f"with a similarity score of {composite_score:.2f}.\n\n"
        f"Incoming video text:\n\"{video_text}\"\n\n"
        f"Known harmful seed text:\n\"{matched_seed_text}\"\n\n"
        f"Does the incoming video carry the same harmful intent as the seed? "
        f"Reply with a JSON object only, no explanation:\n"
        f"{{\"harmful_probability\": <float 0.0-1.0>, \"reasoning\": \"<one sentence>\"}}"
    )


def _call_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        raw = response.json().get("response", "{}")
        parsed = json.loads(raw)
        prob = float(parsed.get("harmful_probability", 0.5))
        reasoning = parsed.get("reasoning", "")
        return max(0.0, min(1.0, prob)), reasoning
    except Exception as e:
        return None, f"llm_error: {str(e)}"


def decide(match_confidence, video_text=None, matched_seed_text=None):
    """v3 decision logic — NO AUTO-REMOVAL. Anchors on match_confidence.

    < 0.30   -> ALLOW
    0.30-0.59 -> HUMAN REVIEW (normal priority)
    >= 0.60  -> invoke Ollama for a second opinion, but the LLM can only
                escalate priority, never remove:
        LLM >= 0.70 -> HUMAN REVIEW (urgent)
        LLM 0.50-0.69 -> HUMAN REVIEW (normal)
        LLM < 0.50  -> ALLOW

    Returns dict: decision, priority, match_confidence, llm_score, reasoning
    """
    t = LLM_THRESHOLDS

    if match_confidence < t["allow_below"]:
        return {
            "decision": ALLOW,
            "priority": None,
            "match_confidence": match_confidence,
            "llm_score": None,
            "reasoning": f"match_confidence {match_confidence:.4f} below allow threshold {t['allow_below']}",
        }

    if match_confidence < t["review_below"]:
        return {
            "decision": REVIEW,
            "priority": "normal",
            "match_confidence": match_confidence,
            "llm_score": None,
            "reasoning": f"match_confidence {match_confidence:.4f} in review band",
        }

    llm_score, reasoning = _call_ollama(
        _build_prompt(video_text or "", matched_seed_text or "", match_confidence)
    )

    if llm_score is None:
        # fail safe to review — never auto-act when the verifier is unavailable
        return {
            "decision": REVIEW,
            "priority": "normal",
            "match_confidence": match_confidence,
            "llm_score": None,
            "reasoning": f"llm unavailable ({reasoning}); defaulting to review",
        }

    if llm_score >= t["llm_urgent_above"]:
        decision, priority = REVIEW, "urgent"
    elif llm_score >= t["llm_allow_below"]:
        decision, priority = REVIEW, "normal"
    else:
        decision, priority = ALLOW, None

    return {
        "decision": decision,
        "priority": priority,
        "match_confidence": match_confidence,
        "llm_score": round(llm_score, 4),
        "reasoning": reasoning,
    }
