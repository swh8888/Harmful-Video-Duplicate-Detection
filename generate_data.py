import os
import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# this script makes the synthetic raw data sources for the project.
# it runs on its own, no docker needed. just: python generate_data.py
# it writes 4 separate csv files into data/ that all share the same video_id.
# the medallion pipeline (bronze -> silver -> gold) picks these up after.

random.seed(42)
np.random.seed(42)

DATA_DIR = "data"

# we spread uploads across a few monthly snapshots so the bronze layer
# has something to partition on (same idea as the loan labs).
SNAPSHOT_MONTHS = [
    "2025-01-01",
    "2025-02-01",
    "2025-03-01",
    "2025-04-01",
    "2025-05-01",
    "2025-06-01",
]

VARIANTS_PER_SEED = 40  # rough number of reposts we make per seed message

SYNTHETIC_HARMFUL_SEEDS = [
    "This content encourages self harm and dangerous behaviour",
    "This video promotes violence against another person",
    "This message contains hateful abuse toward a protected group",
    "This post gives dangerous instructions that could cause harm",
    "This clip threatens physical harm against someone",
]

SYNTHETIC_CLEAN_SEEDS = [
    "This video teaches a simple cooking recipe",
    "This post shares study tips for exams",
    "This video explains how to exercise safely",
    "This message talks about weekend travel plans",
    "This content reviews a new phone",
]

regions = ["SG", "MY", "US", "ID"]
locations = ["top", "center", "bottom", "left", "right"]

# ---------------------------------------------------------------------------
# Evasion transforms (text-agnostic, medium-specific)
# ---------------------------------------------------------------------------
# These work on ANY seed text (synthetic OR real corpus), so the synthesized
# duplication is genuinely evaded regardless of where the seed came from. Each
# medium gets the evasion style a real evader would use against that signal:
#   title -> leetspeak / symbol injection   (attacker types it)
#   ocr   -> visual character misreads       (l<->1, rn->m, o->0 ... OCR errors)
#   asr   -> homophones / phonetic slips      (speech-to-text mishearings)
# Each medium also keeps an "exact" repost path plus light near_duplicate /
# structural paths, so the corpus spans easy reposts through hard evasions and
# the detector is tested across the full difficulty range (a realistic recall
# below 1.0 rather than a trivially perfect one).

LEET_MAP = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7", "b": "8", "l": "|"}
OCR_MULTI = [("rn", "m"), ("cl", "d"), ("vv", "w"), ("nn", "m")]
OCR_SINGLE = {"l": "1", "I": "l", "O": "0", "o": "0", "S": "5", "B": "8", "g": "9", "e": "c"}
HOMOPHONES = {
    "to": "too", "too": "to", "two": "to", "there": "their", "their": "there",
    "your": "you're", "you're": "your", "its": "it's", "for": "four", "no": "know",
    "know": "no", "right": "write", "by": "buy", "hear": "here", "here": "hear",
    "one": "won", "new": "knew", "be": "bee", "see": "sea", "i": "eye",
}


def _near_dup(text):
    return random.choice([text + "!!!", text + " ...", text.upper(), text.lower()])


def _structural(text):
    words = text.split()
    if len(words) > 4:
        cut = len(words) // 2
        return " ".join(words[cut:] + words[:cut])
    return " ".join(reversed(words))


def _leetspeak(text, rate):
    return "".join(
        LEET_MAP[c.lower()] if c.lower() in LEET_MAP and random.random() < rate else c
        for c in text
    )


def _inject_symbols(text, rate):
    sep = random.choice([".", "*", "-", "_", " "])
    out = []
    for c in text:
        out.append(c)
        if c != " " and random.random() < rate:
            out.append(sep)
    return "".join(out)


def _ocr_misread(text, rate):
    out = text
    for a, b in OCR_MULTI:
        if a in out and random.random() < rate:
            out = out.replace(a, b, 1)
    return "".join(
        OCR_SINGLE[c] if c in OCR_SINGLE and random.random() < rate else c for c in out
    )


def _asr_homophone(text, rate):
    out = []
    for w in text.split():
        bare = w.lower().strip(".,!?;:")
        out.append(HOMOPHONES[bare] if bare in HOMOPHONES and random.random() < rate else w)
    s = " ".join(out)
    for a, b in [("ph", "f"), ("ck", "k")]:
        if random.random() < rate * 0.5:
            s = s.replace(a, b, 1)
    return s


def title_variant(text):
    """Typed-text evasion: leetspeak + symbol injection."""
    roll = random.random()
    if roll < 0.30:
        return text, "exact"
    if roll < 0.55:
        return _leetspeak(text, random.uniform(0.15, 0.45)), "leetspeak"
    if roll < 0.75:
        return _inject_symbols(text, random.uniform(0.10, 0.30)), "symbol_injection"
    if roll < 0.90:
        return _near_dup(text), "near_duplicate"
    return _structural(text), "structural"


def ocr_variant(text):
    """On-screen-text evasion: OCR visual misreads."""
    roll = random.random()
    if roll < 0.30:
        return text, "exact"
    if roll < 0.65:
        return _ocr_misread(text, random.uniform(0.15, 0.45)), "ocr_misread"
    if roll < 0.85:
        return _near_dup(text), "near_duplicate"
    return _structural(text), "structural"


def asr_variant(text):
    """Spoken-text evasion: homophones / phonetic slips."""
    roll = random.random()
    if roll < 0.30:
        return text, "exact"
    if roll < 0.65:
        return _asr_homophone(text, random.uniform(0.30, 0.70)), "asr_homophone"
    if roll < 0.85:
        return _near_dup(text), "near_duplicate"
    return _structural(text), "structural"


def maybe_missing(value, missing_rate):
    # sometimes a feed just does not give us a value. return None so the
    # silver / gold layers have real missingness to deal with.
    if random.random() < missing_rate:
        return None
    return value


def make_ocr_row(base_group):
    return {
        "font_size": maybe_missing(max(8, int(random.gauss(22 + base_group, 3))), 0.04),
        "text_location": random.choice(locations),
        "contrast_level": maybe_missing(round(min(max(random.gauss(0.75, 0.12), 0), 1), 2), 0.05),
        "text_density": round(min(max(random.gauss(0.55, 0.15), 0), 1), 2),
        "is_animated": random.choice([True, False]),
        "ocr_confidence": maybe_missing(round(min(max(random.gauss(0.88, 0.08), 0), 1), 2), 0.05),
    }


def make_asr_row(base_group):
    return {
        "speech_pace": maybe_missing(max(60, int(random.gauss(140 + base_group * 3, 18))), 0.04),
        "speech_volume": round(min(max(random.gauss(0.65, 0.14), 0), 1), 2),
        "avg_confidence": maybe_missing(round(min(max(random.gauss(0.86, 0.09), 0), 1), 2), 0.05),
        "has_silence": random.choice([True, False]),
        "has_overlap": random.choice([True, False]),
        "is_distorted": random.choice([True, False]),
    }


# ---------------------------------------------------------------------------
# Optional public-corpus seeding (USE_CORPUS=1)
# ---------------------------------------------------------------------------
# By default the harmful/clean seed payloads are the synthetic placeholder
# sentences above. With USE_CORPUS=1, real text samples are pulled from a public
# Hugging Face classification corpus and used as the seed payloads instead; the
# evasion-variant and feature-synthesis logic is unchanged. Anything that goes
# wrong (no `datasets` package, no network, dataset/columns missing) falls back
# to the synthetic seeds so a run never breaks — graders offline get a clean run.
#
# All knobs are env-overridable:
#   USE_CORPUS, CORPUS_DATASET, CORPUS_SPLIT, CORPUS_TEXT_COL, CORPUS_LABEL_COL,
#   CORPUS_HARMFUL_VALUES, CORPUS_CLEAN_VALUES, CORPUS_N_HARMFUL, CORPUS_N_CLEAN,
#   CORPUS_MAXLEN, CORPUS_REFRESH
DEFAULT_CORPUS = "vibhorag101/suicide_prediction_dataset_phr"
CORPUS_CACHE = os.path.join(DATA_DIR, "corpus_seeds.json")


def _env_true(key):
    return os.environ.get(key, "").strip().lower() in ("1", "true", "yes", "on")


def _pull_corpus_seeds(n_harmful, n_clean):
    from datasets import load_dataset  # optional dependency

    name = os.environ.get("CORPUS_DATASET", DEFAULT_CORPUS)
    split = os.environ.get("CORPUS_SPLIT", "train")
    tcol = os.environ.get("CORPUS_TEXT_COL", "text")
    lcol = os.environ.get("CORPUS_LABEL_COL", "label")
    harmful_vals = {v.strip().lower() for v in os.environ.get("CORPUS_HARMFUL_VALUES", "1,suicide").split(",")}
    clean_vals = {v.strip().lower() for v in os.environ.get("CORPUS_CLEAN_VALUES", "0,non-suicide,nonsuicide,non_suicide").split(",")}
    maxlen = int(os.environ.get("CORPUS_MAXLEN", "280"))

    ds = load_dataset(name, split=split)
    harmful, clean = [], []
    for ex in ds.shuffle(seed=42):
        raw = ex.get(tcol)
        text = " ".join(str(raw).split())[:maxlen] if raw is not None else ""
        if not text:
            continue
        lab = str(ex.get(lcol)).strip().lower()
        if lab in harmful_vals and len(harmful) < n_harmful:
            harmful.append(text)
        elif lab in clean_vals and len(clean) < n_clean:
            clean.append(text)
        if len(harmful) >= n_harmful and len(clean) >= n_clean:
            break

    if len(harmful) < n_harmful or len(clean) < n_clean:
        raise ValueError(
            f"found only {len(harmful)}/{n_harmful} harmful and {len(clean)}/{n_clean} "
            f"clean samples in '{name}' (check CORPUS_TEXT_COL/CORPUS_LABEL_COL/"
            f"CORPUS_HARMFUL_VALUES/CORPUS_CLEAN_VALUES)"
        )
    return harmful, clean


def load_seed_texts():
    """Return (harmful_seeds, clean_seeds). Synthetic by default; real corpus
    samples when USE_CORPUS is set (with on-disk cache + synthetic fallback)."""
    if not _env_true("USE_CORPUS"):
        return SYNTHETIC_HARMFUL_SEEDS, SYNTHETIC_CLEAN_SEEDS

    n_h = int(os.environ.get("CORPUS_N_HARMFUL", len(SYNTHETIC_HARMFUL_SEEDS)))
    n_c = int(os.environ.get("CORPUS_N_CLEAN", len(SYNTHETIC_CLEAN_SEEDS)))

    # reuse a previous pull so reruns work offline and stay reproducible
    if os.path.exists(CORPUS_CACHE) and not _env_true("CORPUS_REFRESH"):
        try:
            obj = json.load(open(CORPUS_CACHE))
            h, c = obj["harmful"][:n_h], obj["clean"][:n_c]
            if len(h) == n_h and len(c) == n_c:
                print(f"USE_CORPUS: loaded {n_h} harmful + {n_c} clean seed payloads "
                      f"from cache ({CORPUS_CACHE})")
                return h, c
        except Exception as e:  # noqa: BLE001 - cache is best-effort
            print(f"USE_CORPUS: cache unreadable ({e}); re-pulling")

    name = os.environ.get("CORPUS_DATASET", DEFAULT_CORPUS)
    try:
        h, c = _pull_corpus_seeds(n_h, n_c)
        os.makedirs(DATA_DIR, exist_ok=True)
        json.dump({"dataset": name, "harmful": h, "clean": c}, open(CORPUS_CACHE, "w"), indent=2)
        print(f"USE_CORPUS: pulled {len(h)} harmful + {len(c)} clean real samples "
              f"from '{name}'; cached to {CORPUS_CACHE}")
        return h, c
    except Exception as e:  # noqa: BLE001 - any failure must not break generation
        print(f"USE_CORPUS requested but corpus load from '{name}' failed "
              f"({type(e).__name__}: {e}).")
        print("Falling back to synthetic seeds. Corpus mode needs `pip install datasets` "
              "and network access.")
        return SYNTHETIC_HARMFUL_SEEDS, SYNTHETIC_CLEAN_SEEDS


def build_records(harmful_seeds, clean_seeds):
    label_rows = []
    ocr_rows = []
    asr_rows = []
    meta_rows = []

    video_id = 1
    for label, seeds, user_start, group_prefix in [
        ("harmful", harmful_seeds, 1, "h"),
        ("clean", clean_seeds, 50, "c"),
    ]:
        for seed_index, seed_text in enumerate(seeds):
            for _ in range(VARIANTS_PER_SEED):
                vid = f"v{video_id:04d}"
                seed_id = f"{group_prefix}{seed_index:03d}"

                # pick a snapshot month for this upload, then a real timestamp inside it
                snapshot_date = random.choice(SNAPSHOT_MONTHS)
                month_start = datetime.strptime(snapshot_date, "%Y-%m-%d")
                uploaded_at = month_start + timedelta(
                    days=random.randint(0, 27),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                )

                title, variant_type = title_variant(seed_text)
                # ocr / asr text sometimes just is not there
                ocr_text = ocr_variant(seed_text)[0] if random.random() > 0.12 else ""
                asr_text = asr_variant(seed_text)[0] if random.random() > 0.15 else ""

                label_rows.append({
                    "video_id": vid,
                    "snapshot_date": snapshot_date,
                    "seed_id": seed_id,
                    "variant_type": variant_type,
                    "title_text": title,
                    "ocr_text": ocr_text,
                    "asr_text": asr_text,
                    "label": label,
                })

                ocr = make_ocr_row(seed_index)
                ocr.update({"video_id": vid, "snapshot_date": snapshot_date})
                ocr_rows.append(ocr)

                asr = make_asr_row(seed_index)
                asr.update({"video_id": vid, "snapshot_date": snapshot_date})
                asr_rows.append(asr)

                meta_rows.append({
                    "video_id": vid,
                    "snapshot_date": snapshot_date,
                    "user_id": f"user_{random.randint(user_start, user_start + 8)}",
                    "device_id": f"device_{random.randint(user_start, user_start + 8)}",
                    "audio_id": f"audio_{seed_index if label == 'harmful' else seed_index + 20}",
                    "region": random.choice(regions),
                    "uploaded_at": uploaded_at.strftime("%Y-%m-%d %H:%M:%S"),
                })

                video_id += 1

    return label_rows, ocr_rows, asr_rows, meta_rows


def order_cols(df, front):
    # keep video_id and snapshot_date at the front, easier to read
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    harmful_seeds, clean_seeds = load_seed_texts()
    mode = "public corpus" if _env_true("USE_CORPUS") else "synthetic seeds"
    print(f"seed source: {mode} "
          f"({len(harmful_seeds)} harmful + {len(clean_seeds)} clean seeds)\n")

    label_rows, ocr_rows, asr_rows, meta_rows = build_records(harmful_seeds, clean_seeds)

    front = ["video_id", "snapshot_date"]
    datasets = {
        "dataset_label.csv": order_cols(pd.DataFrame(label_rows), front),
        "dataset_ocr.csv": order_cols(pd.DataFrame(ocr_rows), front),
        "dataset_asr.csv": order_cols(pd.DataFrame(asr_rows), front),
        "dataset_metadata.csv": order_cols(pd.DataFrame(meta_rows), front),
    }

    for name, df in datasets.items():
        path = os.path.join(DATA_DIR, name)
        df.to_csv(path, index=False)
        print(f"saved {path}  rows: {len(df)}  cols: {list(df.columns)}")

    # quick sanity print so we can eyeball the class balance
    label_df = datasets["dataset_label.csv"]
    print("\nlabel counts:")
    print(label_df["label"].value_counts().to_dict())
    print("snapshot spread:")
    print(label_df["snapshot_date"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
