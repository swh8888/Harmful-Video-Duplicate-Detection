import os
import json
from datetime import datetime

import numpy as np
import pyspark
from sklearn.model_selection import train_test_split

import utils.data_processing_gold_table as gold

# data splitting step.
# reads the gold feature store and label store, joins them on the full grain
# (video_id + snapshot_date), then does a stratified 80 / 10 / 10 train / val /
# test split. stratified because the harmful vs clean classes are not perfectly
# balanced and we want the same class ratio in every split (slides, lecture 3).
# video_id is unique per video, so splitting the rows keeps any single video in
# exactly one split and there is no train / test leakage. we assert that grain
# before splitting so the guarantee can never quietly break.

GOLD_DIR = "datamart/gold/"
SPLIT_DIR = "datamart/gold/splits/"
REPORT_PATH = "reports/data_quality_report.json"
MANIFEST_PATH = "reports/feature_manifest.json"

RANDOM_STATE = 42

# columns that are identifiers or text, never numeric model inputs. note that
# some ids (audio_id, seed_id) encode the label, so feeding them to a model
# would be target leakage. the model layer should read feature_manifest.json and
# use only the numeric feature list (plus combined_text for a text model).
ID_COLUMNS = ["video_id", "snapshot_date", "seed_id", "user_id", "device_id", "audio_id", "region", "label_def"]
TEXT_COLUMNS = ["combined_text", "title_text", "ocr_text", "asr_text"]
TARGET = "label"


def build_modeling_frame():
    spark = pyspark.sql.SparkSession.builder \
        .appName("harmful_video_split") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # the gold store is written one parquet folder per snapshot, so we glob all
    # of them and let spark stack the snapshots into a single frame.
    features = spark.read.parquet(os.path.join(GOLD_DIR, "feature_store", "*.parquet"))
    labels = spark.read.parquet(os.path.join(GOLD_DIR, "label_store", "*.parquet"))

    # join features to label on the full grain (video_id + snapshot_date).
    # joining on both keys means a video that somehow lands in two snapshots can
    # never fan out into a cross product, the rows only line up when the snapshot
    # matches too. with the current data video_id is already unique, this just
    # makes the guarantee hold even if that ever stops being true.
    joined = features.join(labels, on=["video_id", "snapshot_date"], how="inner")

    pdf = joined.toPandas()
    spark.stop()
    return pdf


def stratified_split(pdf):
    # 80 / 10 / 10. first peel off 20 percent, then halve it into val and test.
    train_df, temp_df = train_test_split(
        pdf,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=pdf["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"],
    )
    return train_df, val_df, test_df


def label_dist(df):
    return {int(k): int(v) for k, v in df["label"].value_counts().to_dict().items()}


def imputation_summary(pdf):
    # how often did each numeric feature need imputing in the end
    out = {}
    for c in pdf.columns:
        if c.endswith("_was_missing"):
            base = c.replace("_was_missing", "")
            out[base] = {
                "imputed_count": int(pdf[c].sum()),
                "imputed_rate": round(float(pdf[c].mean()), 4),
            }
    return out


def assert_ml_ready(pdf):
    # last line of defence before the data leaves for the model layer. if any of
    # these trips the run dies loudly here instead of shipping a quietly broken
    # table into training.
    numeric = gold.numeric_feature_columns()

    # 1. one row per video. a bad join is the classic way to get duplicates and
    # silent row count inflation, so we refuse to continue if that happened.
    n_dupes = int(pdf["video_id"].duplicated().sum())
    if n_dupes != 0:
        raise ValueError("video_id is not unique, found " + str(n_dupes) + " duplicate rows")

    # 2. target is a clean binary int, nothing unexpected slipped in.
    bad_labels = sorted(set(int(v) for v in pdf["label"].unique()) - {0, 1})
    if bad_labels:
        raise ValueError("label has values outside {0, 1}: " + str(bad_labels))

    # 3. every numeric model input exists and is finite (no null, NaN or inf).
    for c in numeric:
        if c not in pdf.columns:
            raise ValueError("numeric feature missing from modeling frame: " + c)
        values = pdf[c].to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError("numeric feature has non finite values: " + c)

    # 4. the text column the text model reads is never null.
    for c in TEXT_COLUMNS:
        if c not in pdf.columns:
            raise ValueError("text feature missing from modeling frame: " + c)
        if pdf[c].isnull().any():
            raise ValueError("text feature has nulls: " + c)

    print("ml readiness checks passed:", len(numeric), "numeric features, all finite")
    return numeric


def write_manifest(pdf):
    # the contract the model layer reads so it never has to guess which columns
    # are safe to feed a model. the id columns are listed out on purpose, some of
    # them (audio_id, seed_id) encode the label and would leak the answer if a
    # model ever trained on them.
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    numeric = gold.numeric_feature_columns()
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target": TARGET,
        "numeric_features": numeric,
        "text_features": TEXT_COLUMNS,
        "id_columns": ID_COLUMNS,
        "n_numeric_features": len(numeric),
        "usage": "feed numeric_features (and optionally text_features) to the model. never feed id_columns, some of them encode the label.",
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print("saved feature manifest:", MANIFEST_PATH)
    return manifest


def write_report(pdf, train_df, val_df, test_df):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_videos": int(len(pdf)),
        "n_features": int(len([c for c in pdf.columns if c not in ("label", "label_def")])),
        "n_numeric_features": int(len(gold.numeric_feature_columns())),
        "ml_ready": True,
        "label_distribution_overall": label_dist(pdf),
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "label_distribution_per_split": {
            "train": label_dist(train_df),
            "val": label_dist(val_df),
            "test": label_dist(test_df),
        },
        "has_ocr_text_rate": round(float(pdf["has_ocr_text"].mean()), 4),
        "has_asr_text_rate": round(float(pdf["has_asr_text"].mean()), 4),
        "imputation_summary": imputation_summary(pdf),
        "snapshot_distribution": {
            str(k): int(v) for k, v in pdf["snapshot_date"].value_counts().sort_index().to_dict().items()
        },
        "columns": list(pdf.columns),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print("saved data quality report:", REPORT_PATH)
    return report


def save_split(df, name):
    os.makedirs(SPLIT_DIR, exist_ok=True)
    # parquet for the training job, csv as a human readable backup
    df.to_parquet(os.path.join(SPLIT_DIR, name + ".parquet"), index=False)
    df.to_csv(os.path.join(SPLIT_DIR, name + ".csv"), index=False)
    print("saved split:", name, "rows:", len(df))


def run_split():
    pdf = build_modeling_frame()

    # guard against an empty gold layer so the error is obvious
    if len(pdf) == 0:
        raise ValueError("gold layer is empty, run the medallion pipeline before splitting")

    # fail loudly right here if the joined table is not actually model ready,
    # then write the manifest the model layer reads.
    assert_ml_ready(pdf)
    write_manifest(pdf)

    train_df, val_df, test_df = stratified_split(pdf)

    save_split(train_df, "train")
    save_split(val_df, "val")
    save_split(test_df, "test")

    report = write_report(pdf, train_df, val_df, test_df)
    print("split done:", report["split_sizes"])
    return report


if __name__ == "__main__":
    run_split()
