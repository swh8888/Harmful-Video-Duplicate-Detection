import numpy as np

OCR_NUMERIC_FEATURES = ["font_size", "contrast_level", "ocr_confidence", "text_density"]
OCR_BOOL_FEATURES = ["is_animated"]
OCR_CATEGORICAL = "text_location"


def score_ocr(input_features, seed_features):
    """Compare OCR features between input video and seed video.

    Both arguments are dicts with keys matching the OCR feature columns.
    Returns ocr_score in [0, 1] where 1.0 means identical OCR features.
    Returns None if both inputs lack OCR data.
    """
    if input_features is None and seed_features is None:
        return None
    if input_features is None or seed_features is None:
        return 0.0

    scores = []

    for col in OCR_NUMERIC_FEATURES:
        iv = input_features.get(col)
        sv = seed_features.get(col)
        if iv is None or sv is None or np.isnan(iv) or np.isnan(sv):
            continue
        if col == "font_size":
            diff = abs(iv - sv) / max(abs(sv), 1.0)
            scores.append(max(0.0, 1.0 - diff))
        else:
            scores.append(max(0.0, 1.0 - abs(iv - sv)))

    for col in OCR_BOOL_FEATURES:
        iv = input_features.get(col)
        sv = seed_features.get(col)
        if iv is not None and sv is not None:
            scores.append(1.0 if iv == sv else 0.0)

    iv_loc = input_features.get(OCR_CATEGORICAL)
    sv_loc = seed_features.get(OCR_CATEGORICAL)
    if iv_loc is not None and sv_loc is not None:
        scores.append(1.0 if iv_loc == sv_loc else 0.0)

    if not scores:
        return 0.0

    return round(float(np.mean(scores)), 4)


def score_ocr_batch(input_rows, seed_rows):
    """Score OCR similarity for matched pairs.

    input_rows and seed_rows are lists of dicts (or pandas row dicts).
    Returns list of ocr_scores.
    """
    return [score_ocr(inp, seed) for inp, seed in zip(input_rows, seed_rows)]
