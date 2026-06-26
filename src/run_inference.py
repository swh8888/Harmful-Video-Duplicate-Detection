import os
import json
import numpy as np
import pandas as pd

from src.config import SPLIT_DIR, MODELS_DIR, REPORTS_DIR, TEXT_CUTOFF, load_manifest
from src.embedding import embed_texts
from src.train_model import load_artifacts
from src.matching.text_matching import GridTextMatcher
from src.scoring.ocr_scoring import score_ocr
from src.scoring.asr_scoring import score_asr
from src.scoring.metadata_scoring import score_metadata
from src.scoring.composite import score_match, classify_duplicate_type
from src.llm_verifier import decide

# Candidates surfaced per video = ALL matches above the text gate, ranked.
# Analysts review the full cluster and decide lane assignment. Set an int to cap
# the list if a review page ever gets unwieldy; None = show every above-gate match.
MAX_CANDIDATES = None

MEDIUM_TEXT_COLS = ["video_id", "title_text", "ocr_text", "asr_text"]


def load_seed_corpus():
    df = pd.read_parquet(os.path.join(SPLIT_DIR, "train.parquet"))
    seed_df = df[df["label"] == 1].copy()
    print(f"seed corpus: {len(seed_df)} harmful videos")
    return seed_df


def build_matcher(seed_df, model=None, rebuild=False):
    matcher = GridTextMatcher(model=model)
    index_path = os.path.join(MODELS_DIR, "faiss_grid.index")
    if os.path.exists(index_path) and not rebuild:
        matcher.load_index()
        if model is not None:
            matcher.model = model
    else:
        matcher.build_index(seed_df[MEDIUM_TEXT_COLS])
        matcher.save_index()
    return matcher


def row_to_ocr_dict(row):
    cols = ["font_size", "contrast_level", "ocr_confidence", "text_density", "is_animated", "text_location"]
    return {c: row.get(c) for c in cols if c in row}


def row_to_asr_dict(row):
    cols = ["speech_pace", "speech_volume", "avg_confidence", "has_silence", "has_overlap", "is_distorted"]
    return {c: row.get(c) for c in cols if c in row}


def row_to_meta_dict(row):
    cols = ["user_id", "device_id", "audio_id", "region", "uploaded_at"]
    return {c: row.get(c) for c in cols if c in row}


def _query_texts(row):
    return {m: row.get(m) for m in ("title_text", "ocr_text", "asr_text")}


def _same_medium_feature(match, row_dict, seed_row):
    """Same-medium feature similarity for a match; None for title/cross-medium."""
    if match["is_cross_medium"]:
        return None
    m = match["q_medium"]
    if m == "ocr":
        return score_ocr(row_to_ocr_dict(row_dict), row_to_ocr_dict(seed_row))
    if m == "asr":
        return score_asr(row_to_asr_dict(row_dict), row_to_asr_dict(seed_row))
    return None


def _candidate(match, row_dict, seed_lookup):
    """Compact candidate-duplicate record for the reviewer (not the decision)."""
    seed_row = seed_lookup.get(match["seed_video_id"], {})
    feat = _same_medium_feature(match, row_dict, seed_row)
    dtype = (classify_duplicate_type(match["is_cross_medium"], feat)
             if match["text_score"] >= TEXT_CUTOFF else None)
    return {
        "seed_id": match["seed_video_id"],
        "text_score": match["text_score"],
        "pair": f"q.{match['q_medium']}<->c.{match['c_medium']}",
        "is_cross_medium": match["is_cross_medium"],
        "duplicate_type": dtype,
    }


def _blank(vid, layer1_label, harmful_prob, decision, reasoning, lane=None, priority=None):
    return {
        "video_id": vid, "layer1_label": layer1_label, "harmful_prob": harmful_prob,
        "matched_seed_id": None, "best_pair": None, "is_cross_medium": None,
        "medium_transition": None, "text_score": None, "feature_score": None,
        "metadata_score": None, "match_confidence": None, "duplicate_type": None,
        "lane": lane, "decision": decision, "priority": priority,
        "llm_score": None, "reasoning": reasoning,
        "n_candidates": 0, "candidate_matches": "[]",
    }


def run_inference(input_df, seed_df, matcher, model, scaler, manifest):
    """Layer 1 (harmful/clean) then Layer 2 (cross-medium grid match + decoupled
    scoring + no-auto-removal routing) for harmful videos."""
    seed_lookup = seed_df.set_index("video_id").to_dict("index")

    # Layer 1 still uses the pooled combined_text embedding + numeric features.
    input_embeddings = embed_texts(input_df["combined_text"].fillna(""))
    X = np.hstack([input_embeddings, input_df[manifest["numeric_features"]].values.astype(np.float32)])
    layer1_pred = model.predict(scaler.transform(X))
    layer1_prob = model.predict_proba(scaler.transform(X))[:, 1]

    results = []
    for i, (_, row) in enumerate(input_df.iterrows()):
        vid = row["video_id"]
        harmful_prob = round(float(layer1_prob[i]), 4)

        if int(layer1_pred[i]) != 1:
            results.append(_blank(vid, "clean", harmful_prob, "ALLOW", "layer1 classified as clean"))
            continue

        matches = matcher.match(_query_texts(row))  # all seeds, ranked best first
        if not matches:
            results.append(_blank(vid, "harmful", harmful_prob, "HUMAN REVIEW",
                                  "harmful but no seed match", lane="investigation", priority="normal"))
            continue

        row_dict = row.to_dict()
        best = matches[0]
        seed_row = seed_lookup.get(best["seed_video_id"], {})

        # same-medium feature score only; None for title or cross-medium matches
        feature_score = _same_medium_feature(best, row_dict, seed_row)

        # metadata only consulted once a text match clears the gate
        metadata_score = None
        if best["text_score"] >= TEXT_CUTOFF:
            metadata_score = score_metadata(row_to_meta_dict(row_dict), row_to_meta_dict(seed_row))

        sm = score_match(best["text_score"], metadata_score, best["is_cross_medium"], feature_score)
        verdict = decide(sm["match_confidence"], video_text=row.get("combined_text", ""),
                         matched_seed_text=seed_row.get("combined_text", ""))
        lane = sm["lane"] or "investigation"
        priority = verdict.get("priority") or ("urgent" if sm["duplicate_type"] in ("cross_medium", "re_authored") else "normal")

        # full candidate cluster for the reviewer: every match above the gate
        above_gate = [m for m in matches if m["text_score"] >= TEXT_CUTOFF]
        if MAX_CANDIDATES:
            above_gate = above_gate[:MAX_CANDIDATES]
        candidates = [_candidate(m, row_dict, seed_lookup) for m in above_gate]
        n_candidates = len(candidates)

        results.append({
            "video_id": vid, "layer1_label": "harmful", "harmful_prob": harmful_prob,
            "matched_seed_id": best["seed_video_id"],
            "best_pair": f"q.{best['q_medium']} <-> c.{best['c_medium']}",
            "is_cross_medium": best["is_cross_medium"],
            "medium_transition": best["medium_transition"],
            "text_score": sm["text_score"], "feature_score": sm["feature_score"],
            "metadata_score": sm["metadata_score"], "match_confidence": sm["match_confidence"],
            "duplicate_type": sm["duplicate_type"], "lane": lane,
            "decision": verdict["decision"], "priority": verdict.get("priority") or priority,
            "llm_score": verdict["llm_score"], "reasoning": sm["reason"],
            "n_candidates": n_candidates, "candidate_matches": json.dumps(candidates),
        })

    return pd.DataFrame(results)


def run_batch_inference(split_name="test", rebuild_index=False):
    manifest = load_manifest()
    model, scaler = load_artifacts()
    seed_df = load_seed_corpus()
    matcher = build_matcher(seed_df, rebuild=rebuild_index)
    input_df = pd.read_parquet(os.path.join(SPLIT_DIR, f"{split_name}.parquet"))
    print(f"\nrunning inference on {split_name}: {len(input_df)} videos")
    results_df = run_inference(input_df, seed_df, matcher, model, scaler, manifest)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, f"inference_results_{split_name}.parquet")
    results_df.to_parquet(out_path, index=False)
    print(f"saved: {out_path}")
    print("decisions:", results_df["decision"].value_counts().to_dict())
    print("duplicate_type:", results_df["duplicate_type"].value_counts(dropna=False).to_dict())
    print("lanes:", results_df["lane"].value_counts(dropna=False).to_dict())
    return results_df


if __name__ == "__main__":
    run_batch_inference(split_name="test")
