from src.config import COMPOSITE_WEIGHTS


def compute_composite_score(text_score, ocr_score, asr_score, metadata_score):
    """Weighted composite score with missing modality re-normalization.

    When OCR or ASR is None (missing modality), the weight is redistributed
    proportionally across the available signals so the final score stays in [0, 1].

    Returns dict with all sub-scores plus final_score and reason.
    """
    weights = dict(COMPOSITE_WEIGHTS)
    scores = {
        "text": text_score,
        "ocr": ocr_score,
        "asr": asr_score,
        "metadata": metadata_score,
    }

    available = {k: v for k, v in scores.items() if v is not None}
    if not available:
        return {
            "text_score": text_score,
            "ocr_score": ocr_score,
            "asr_score": asr_score,
            "metadata_score": metadata_score,
            "final_score": 0.0,
            "reason": "no signals available",
        }

    total_weight = sum(weights[k] for k in available)
    normalized_weights = {k: weights[k] / total_weight for k in available}

    final_score = sum(normalized_weights[k] * available[k] for k in available)
    final_score = round(max(0.0, min(1.0, final_score)), 4)

    missing = [k for k in scores if scores[k] is None]
    reasons = []
    if text_score is not None and text_score >= 0.90:
        reasons.append("high text similarity")
    if metadata_score is not None and metadata_score >= 0.70:
        reasons.append("metadata match")
    if missing:
        reasons.append(f"missing: {', '.join(missing)}")

    return {
        "text_score": text_score,
        "ocr_score": ocr_score,
        "asr_score": asr_score,
        "metadata_score": metadata_score,
        "final_score": final_score,
        "weights_used": normalized_weights,
        "reason": "; ".join(reasons) if reasons else "composite",
    }


def compute_composite_batch(text_scores, ocr_scores, asr_scores, metadata_scores):
    """Compute composite scores for a batch of matched pairs."""
    return [
        compute_composite_score(t, o, a, m)
        for t, o, a, m in zip(text_scores, ocr_scores, asr_scores, metadata_scores)
    ]


# ---------------------------------------------------------------------------
# v3 decoupled scoring
# ---------------------------------------------------------------------------
from src.config import TEXT_CUTOFF, METADATA_LIFT, FEATURE_HIGH


def classify_duplicate_type(is_cross_medium, feature_score):
    """What KIND of duplicate (drives routing/priority, never the keep decision)."""
    if is_cross_medium:
        return "cross_medium"
    if feature_score is not None and feature_score >= FEATURE_HIGH:
        return "same_asset"
    return "re_authored"


def lane_for(duplicate_type):
    """fast lane = quick-confirm reuploads; investigation lane = needs analysis."""
    return "fast" if duplicate_type == "same_asset" else "investigation"


def score_match(text_score, metadata_score, is_cross_medium, feature_score):
    """v3 decoupled scoring.

    Produces TWO outputs from a matched candidate:
      match_confidence : text_score + capped metadata lift  ("is it a duplicate?")
      duplicate_type   : from medium config + feature similarity  ("what kind?")

    Metadata corroborates, never rescues: below the text gate it is not applied,
    and its lift is capped so it can never carry a weak text match over the bar.
    Feature similarity affects only the TYPE, never the confidence.
    """
    if text_score is None or text_score < TEXT_CUTOFF:
        return {
            "text_score": text_score,
            "metadata_score": None,          # never computed below the gate
            "feature_score": None,
            "match_confidence": round(float(text_score or 0.0), 4),
            "duplicate_type": None,
            "lane": None,
            "reason": f"text {text_score} below gate {TEXT_CUTOFF}; no duplicate",
        }

    lift = METADATA_LIFT * (metadata_score or 0.0)
    match_confidence = round(min(1.0, text_score + lift), 4)
    dtype = classify_duplicate_type(is_cross_medium, feature_score)
    return {
        "text_score": round(float(text_score), 4),
        "metadata_score": None if metadata_score is None else round(float(metadata_score), 4),
        "feature_score": None if feature_score is None else round(float(feature_score), 4),
        "match_confidence": match_confidence,
        "duplicate_type": dtype,
        "lane": lane_for(dtype),
        "reason": f"text {round(text_score,3)} + metadata lift {round(lift,3)} -> {dtype}",
    }
