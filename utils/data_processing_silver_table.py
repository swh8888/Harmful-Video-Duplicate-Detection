import os

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, BooleanType, TimestampType

# silver layer.
# this is the "data processing" step. we enforce the schema (cast every column
# to the type we actually want), clean up the text, and tidy the categoricals
# and timestamps. we do NOT impute or drop duplicates here on purpose - that is
# feature engineering work and it happens in gold. silver should be clean and
# typed, but still a faithful, per-record view of the data.


def clean_text_col(c):
    # lowercase, drop anything that is not a letter / number / space, then
    # squeeze repeated spaces. keeps the text comparable downstream.
    c = F.lower(c)
    c = F.regexp_replace(c, r"[^a-z0-9\s]", " ")
    c = F.regexp_replace(c, r"\s+", " ")
    return F.trim(c)


def silver_label(df):
    df = df.withColumn("video_id", col("video_id").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(StringType()))
    df = df.withColumn("seed_id", col("seed_id").cast(StringType()))
    df = df.withColumn("variant_type", col("variant_type").cast(StringType()))
    df = df.withColumn("label", F.lower(F.trim(col("label"))).cast(StringType()))

    # flag where text was missing before we clean it, gold uses these flags
    df = df.withColumn("ocr_missing", col("ocr_text").isNull().cast(IntegerType()))
    df = df.withColumn("asr_missing", col("asr_text").isNull().cast(IntegerType()))

    # clean the three text fields, null becomes empty string after cleaning
    for c in ["title_text", "ocr_text", "asr_text"]:
        df = df.withColumn(c, F.coalesce(col(c), F.lit("")))
        df = df.withColumn(c, clean_text_col(col(c)))

    # one combined text field that the model layer can use directly
    df = df.withColumn(
        "combined_text",
        F.trim(F.concat_ws(" ", col("title_text"), col("ocr_text"), col("asr_text"))),
    )
    return df


def silver_ocr(df):
    df = df.withColumn("video_id", col("video_id").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(StringType()))
    df = df.withColumn("font_size", col("font_size").cast(IntegerType()))
    df = df.withColumn("text_location", F.lower(F.trim(col("text_location"))).cast(StringType()))
    df = df.withColumn("contrast_level", col("contrast_level").cast(FloatType()))
    df = df.withColumn("text_density", col("text_density").cast(FloatType()))
    df = df.withColumn("is_animated", col("is_animated").cast(BooleanType()))
    df = df.withColumn("ocr_confidence", col("ocr_confidence").cast(FloatType()))
    return df


def silver_asr(df):
    df = df.withColumn("video_id", col("video_id").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(StringType()))
    df = df.withColumn("speech_pace", col("speech_pace").cast(IntegerType()))
    df = df.withColumn("speech_volume", col("speech_volume").cast(FloatType()))
    df = df.withColumn("avg_confidence", col("avg_confidence").cast(FloatType()))
    df = df.withColumn("has_silence", col("has_silence").cast(BooleanType()))
    df = df.withColumn("has_overlap", col("has_overlap").cast(BooleanType()))
    df = df.withColumn("is_distorted", col("is_distorted").cast(BooleanType()))
    return df


def silver_metadata(df):
    df = df.withColumn("video_id", col("video_id").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(StringType()))
    df = df.withColumn("user_id", col("user_id").cast(StringType()))
    df = df.withColumn("device_id", col("device_id").cast(StringType()))
    df = df.withColumn("audio_id", col("audio_id").cast(StringType()))
    df = df.withColumn("region", F.upper(F.trim(col("region"))).cast(StringType()))
    df = df.withColumn("uploaded_at", col("uploaded_at").cast(TimestampType()))

    # pull a couple of simple time parts out of the timestamp, handy later
    df = df.withColumn("upload_hour", F.hour(col("uploaded_at")).cast(IntegerType()))
    df = df.withColumn("upload_dayofweek", F.dayofweek(col("uploaded_at")).cast(IntegerType()))
    return df


# small lookup so the dispatcher stays short
SILVER_BUILDERS = {
    "label": silver_label,
    "ocr": silver_ocr,
    "asr": silver_asr,
    "metadata": silver_metadata,
}


def process_silver_table(snapshot_date_str, table_name, bronze_directory, silver_directory, spark):
    # read the matching bronze partition
    partition_name = "bronze_" + table_name + "_" + snapshot_date_str.replace("-", "_") + ".csv"
    in_path = os.path.join(bronze_directory, table_name, partition_name)
    # multiLine + escape so quoted text with commas / newlines round-trips safely
    df = spark.read.csv(in_path, header=True, inferSchema=True, multiLine=True, escape='"')
    print("loaded from:", in_path, "row count:", df.count())

    # run the cleaning / typing for this specific table
    df = SILVER_BUILDERS[table_name](df)

    # save typed parquet, one partition per table per snapshot
    out_dir = os.path.join(silver_directory, table_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_name = "silver_" + table_name + "_" + snapshot_date_str.replace("-", "_") + ".parquet"
    out_path = os.path.join(out_dir, out_name)
    df.write.mode("overwrite").parquet(out_path)
    print("saved to:", out_path)

    return df
