import math
from datetime import datetime

EXACT_MATCH_FIELDS = ["user_id", "device_id", "audio_id", "region"]
TEMPORAL_FIELD = "uploaded_at"

TEMPORAL_DECAY_SCALE_HOURS = 24.0


def temporal_proximity(t1_str, t2_str):
    """Exponential decay score based on time difference.

    Same hour ~ 0.96, 24h ~ 0.37, 72h ~ 0.05
    """
    try:
        t1 = datetime.strptime(t1_str, "%Y-%m-%d %H:%M:%S")
        t2 = datetime.strptime(t2_str, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return 0.0

    hours_diff = abs((t1 - t2).total_seconds()) / 3600.0
    return round(math.exp(-hours_diff / TEMPORAL_DECAY_SCALE_HOURS), 4)


def score_metadata(input_meta, seed_meta):
    """Compare metadata between input video and seed video.

    Always runs independently of text/OCR/ASR matching.
    Returns metadata_score in [0, 1].
    """
    if input_meta is None or seed_meta is None:
        return 0.0

    exact_scores = []
    for field in EXACT_MATCH_FIELDS:
        iv = input_meta.get(field)
        sv = seed_meta.get(field)
        if iv is not None and sv is not None:
            exact_scores.append(1.0 if str(iv) == str(sv) else 0.0)

    exact_avg = sum(exact_scores) / len(exact_scores) if exact_scores else 0.0

    t1 = input_meta.get(TEMPORAL_FIELD)
    t2 = seed_meta.get(TEMPORAL_FIELD)
    temp_score = temporal_proximity(t1, t2) if (t1 and t2) else 0.0

    metadata_score = 0.70 * exact_avg + 0.30 * temp_score

    return round(float(metadata_score), 4)


def score_metadata_batch(input_rows, seed_rows):
    """Score metadata similarity for matched pairs."""
    return [score_metadata(inp, seed) for inp, seed in zip(input_rows, seed_rows)]
