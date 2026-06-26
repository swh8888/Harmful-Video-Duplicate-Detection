import os

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, DoubleType

# gold layer.
# this is where the feature engineering happens: impute the missing numbers,
# drop duplicate records, encode the categoricals, add a few calculated
# features, and join the four sources into one wide row per video. the output
# is the feature store and the label store, and those two are what the model
# training step reads from.
#
# ml readiness contract for the feature store:
#   - every numeric feature column is finite. no null, no NaN, no inf, ever.
#   - categoricals are one hot encoded against a fixed vocabulary, so the column
#     set is identical from one snapshot to the next.
#   - combined_text is never null (empty string at worst) for the text models.
#   - id columns (video_id, the user / device / audio ids, seed_id, region) are
#     kept for reference but are listed out as ids, not model inputs. some of
#     them encode the label, so the model layer must read the feature manifest
#     and only feed the numeric features.

OCR_NUMERIC = ["font_size", "contrast_level", "ocr_confidence", "text_density"]
ASR_NUMERIC = ["speech_pace", "speech_volume", "avg_confidence"]
META_NUMERIC = ["upload_hour", "upload_dayofweek"]
OCR_BOOL = ["is_animated"]
ASR_BOOL = ["has_silence", "has_overlap", "is_distorted"]
LOCATIONS = ["top", "center", "bottom", "left", "right"]
REGIONS = ["SG", "MY", "US", "ID"]  # silver upper cases region, so match that here


def read_silver(table_name, snapshot_date_str, silver_directory, spark):
    name = "silver_" + table_name + "_" + snapshot_date_str.replace("-", "_") + ".parquet"
    path = os.path.join(silver_directory, table_name, name)
    df = spark.read.parquet(path)
    # remove duplicate records. one row per video is the contract for the
    # feature store, so if a video got ingested twice we keep a single copy.
    return df.dropDuplicates(["video_id"])


def not_finite(x):
    # true when a double is null, NaN or plus / minus infinity. we treat all of
    # those as missing so nothing un modelable ever reaches the feature store.
    inf = F.lit(float("inf"))
    return x.isNull() | F.isnan(x) | (x == inf) | (x == -inf)


def impute_median(df, columns):
    # fill missing numbers with the column median and keep a flag so we never
    # lose the fact that the value was originally missing. anything that is not
    # finite (null, NaN, inf) counts as missing here.
    for c in columns:
        x = col(c).cast(DoubleType())
        bad = not_finite(x)
        # compute the median over the finite values only, so a stray NaN can
        # not poison the statistic.
        clean = df.withColumn("_clean", F.when(bad, None).otherwise(x))
        quantiles = clean.approxQuantile("_clean", [0.5], 0.0)
        median = quantiles[0] if quantiles else 0.0
        if median is None:
            median = 0.0
        df = df.withColumn(c + "_was_missing", bad.cast(IntegerType()))
        df = df.withColumn(c, F.when(bad, F.lit(float(median))).otherwise(x))
    return df


def bool_to_int(df, columns):
    # booleans to 0 / 1, null becomes 0.
    for c in columns:
        df = df.withColumn(c, F.coalesce(col(c).cast(IntegerType()), F.lit(0)))
    return df


def one_hot(df, source_col, prefix, vocabulary):
    # fixed vocabulary one hot encoding. an unknown or null category lands as all
    # zeros, and because the vocabulary is fixed the output columns are always
    # the same set no matter what shows up in a given snapshot.
    df = df.withColumn(source_col, F.coalesce(col(source_col), F.lit("unknown")))
    for v in vocabulary:
        df = df.withColumn(prefix + v.lower(), (col(source_col) == v).cast(IntegerType()))
    return df


def make_finite(df, columns):
    # final safety net after the joins. a left join can introduce nulls when one
    # side is missing a video, so we force every numeric model input to a finite
    # double here. this is what actually guarantees the ml readiness contract.
    for c in columns:
        x = col(c).cast(DoubleType())
        df = df.withColumn(c, F.when(not_finite(x), F.lit(0.0)).otherwise(x))
    return df


def numeric_feature_columns():
    # the full list of numeric model inputs, in a stable order. used both for the
    # finite safety net and for the feature manifest the split step writes out.
    return (
        ["title_len", "combined_len", "has_ocr_text", "has_asr_text"]
        + OCR_NUMERIC
        + [c + "_was_missing" for c in OCR_NUMERIC]
        + OCR_BOOL
        + ["loc_" + loc for loc in LOCATIONS]
        + ASR_NUMERIC
        + [c + "_was_missing" for c in ASR_NUMERIC]
        + ASR_BOOL
        + META_NUMERIC
        + [c + "_was_missing" for c in META_NUMERIC]
        + ["reg_" + r.lower() for r in REGIONS]
    )


def process_gold_feature_store(snapshot_date_str, silver_directory, gold_directory, spark):
    label = read_silver("label", snapshot_date_str, silver_directory, spark)
    ocr = read_silver("ocr", snapshot_date_str, silver_directory, spark)
    asr = read_silver("asr", snapshot_date_str, silver_directory, spark)
    meta = read_silver("metadata", snapshot_date_str, silver_directory, spark)

    # --- text side features (from the label table, minus the label itself) ---
    label = label.withColumn("title_len", F.size(F.split(col("title_text"), " ")))
    label = label.withColumn("combined_len", F.size(F.split(col("combined_text"), " ")))
    label = label.withColumn("has_ocr_text", (col("ocr_missing") == 0).cast(IntegerType()))
    label = label.withColumn("has_asr_text", (col("asr_missing") == 0).cast(IntegerType()))
    text_feats = label.select(
        "video_id", "snapshot_date", "seed_id", "combined_text",
        # v3: carry the per-medium texts through so Layer-2 matching can embed
        # each medium separately (3x3 grid) instead of pooling into combined_text.
        "title_text", "ocr_text", "asr_text",
        "title_len", "combined_len", "has_ocr_text", "has_asr_text",
    )

    # --- ocr features: impute, one hot the location, booleans to int ---
    ocr = impute_median(ocr, OCR_NUMERIC)
    ocr = bool_to_int(ocr, OCR_BOOL)
    ocr = one_hot(ocr, "text_location", "loc_", LOCATIONS)
    ocr_feats = ocr.select(
        "video_id",
        *OCR_NUMERIC,
        *[c + "_was_missing" for c in OCR_NUMERIC],
        *OCR_BOOL,
        *["loc_" + loc for loc in LOCATIONS],
    )

    # --- asr features: impute, booleans to int ---
    asr = impute_median(asr, ASR_NUMERIC)
    asr = bool_to_int(asr, ASR_BOOL)
    asr_feats = asr.select(
        "video_id",
        *ASR_NUMERIC,
        *[c + "_was_missing" for c in ASR_NUMERIC],
        *ASR_BOOL,
    )

    # --- metadata: impute the time parts, one hot the region, keep the ids ---
    meta = impute_median(meta, META_NUMERIC)
    meta = one_hot(meta, "region", "reg_", REGIONS)
    meta_feats = meta.select(
        "video_id", "user_id", "device_id", "audio_id", "region",
        *META_NUMERIC,
        *[c + "_was_missing" for c in META_NUMERIC],
        *["reg_" + r.lower() for r in REGIONS],
    )

    # join everything into one wide row per video
    features = (
        text_feats
        .join(ocr_feats, on="video_id", how="left")
        .join(asr_feats, on="video_id", how="left")
        .join(meta_feats, on="video_id", how="left")
    )

    # safety net: guarantee every numeric model input is a finite number even if
    # a join left a gap. after this line the feature store is ml ready.
    features = make_finite(features, numeric_feature_columns())

    out_dir = os.path.join(gold_directory, "feature_store")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = "gold_feature_store_" + snapshot_date_str.replace("-", "_") + ".parquet"
    out_path = os.path.join(out_dir, out_name)
    features.write.mode("overwrite").parquet(out_path)
    print("saved feature store:", out_path, "row count:", features.count())

    return features


def process_gold_label_store(snapshot_date_str, silver_directory, gold_directory, spark):
    # label engineering, medallion style. map the raw label to a clean int target
    # and stamp it with a label definition so it is reproducible. only the exact
    # string "harmful" maps to 1, everything else is 0, so a missing or unexpected
    # label can never silently become the positive class.
    label = read_silver("label", snapshot_date_str, silver_directory, spark)

    label = label.withColumn(
        "label",
        F.when(col("label") == "harmful", F.lit(1)).otherwise(F.lit(0)).cast(IntegerType()),
    )
    label = label.withColumn("label_def", F.lit("harmful_vs_clean").cast(StringType()))
    label = label.select("video_id", "snapshot_date", "label", "label_def")

    out_dir = os.path.join(gold_directory, "label_store")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = "gold_label_store_" + snapshot_date_str.replace("-", "_") + ".parquet"
    out_path = os.path.join(out_dir, out_name)
    label.write.mode("overwrite").parquet(out_path)
    print("saved label store:", out_path, "row count:", label.count())

    return label
