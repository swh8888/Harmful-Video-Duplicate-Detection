import numpy as np

ASR_NUMERIC_FEATURES = ["speech_pace", "speech_volume", "avg_confidence"]
ASR_BOOL_FEATURES = ["has_silence", "has_overlap", "is_distorted"]


def score_asr(input_features, seed_features):
    """Compare ASR features between input video and seed video.

    Returns asr_score in [0, 1] where 1.0 means identical ASR features.
    Returns None if the video has no audio (both sides missing).
    """
    if input_features is None and seed_features is None:
        return None
    if input_features is None or seed_features is None:
        return 0.0

    scores = []

    for col in ASR_NUMERIC_FEATURES:
        iv = input_features.get(col)
        sv = seed_features.get(col)
        if iv is None or sv is None or np.isnan(iv) or np.isnan(sv):
            continue
        if col == "speech_pace":
            diff = abs(iv - sv) / max(abs(sv), 1.0)
            scores.append(max(0.0, 1.0 - diff))
        else:
            scores.append(max(0.0, 1.0 - abs(iv - sv)))

    for col in ASR_BOOL_FEATURES:
        iv = input_features.get(col)
        sv = seed_features.get(col)
        if iv is not None and sv is not None:
            scores.append(1.0 if iv == sv else 0.0)

    if not scores:
        return 0.0

    return round(float(np.mean(scores)), 4)


def score_asr_batch(input_rows, seed_rows):
    """Score ASR similarity for matched pairs."""
    return [score_asr(inp, seed) for inp, seed in zip(input_rows, seed_rows)]
