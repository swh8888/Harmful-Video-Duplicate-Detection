import os

import pyspark

import utils.data_processing_bronze_table as bronze
import utils.data_processing_silver_table as silver
import utils.data_processing_gold_table as gold
import split_data

# this is the orchestration entry point. it runs the full backfill across all
# snapshots: raw -> bronze -> silver -> gold, and then kicks off the train /
# val / test split at the end. this is the thing that runs inside docker.
#
# note: run generate_data.py first (outside docker) so the data/ csvs exist.

TABLES = ["label", "ocr", "asr", "metadata"]

BRONZE_DIR = "datamart/bronze/"
SILVER_DIR = "datamart/silver/"
GOLD_DIR = "datamart/gold/"


def get_snapshot_dates(spark):
    # figure out which snapshots we have straight from the raw label feed,
    # rather than hardcoding a date range. keeps it honest if the data changes.
    df = spark.read.csv("data/dataset_label.csv", header=True, inferSchema=True, multiLine=True, escape='"')
    dates = [r["snapshot_date"] for r in df.select("snapshot_date").distinct().collect()]
    dates = sorted(str(d) for d in dates)
    return dates


def main():
    spark = pyspark.sql.SparkSession.builder \
        .appName("harmful_video_medallion") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    for d in [BRONZE_DIR, SILVER_DIR, GOLD_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    dates_lst = get_snapshot_dates(spark)
    print("snapshots to process:", dates_lst)

    # bronze backfill
    print("\n=== bronze ===")
    for table in TABLES:
        for date_str in dates_lst:
            bronze.process_bronze_table(date_str, table, BRONZE_DIR, spark)

    # silver backfill
    print("\n=== silver ===")
    for table in TABLES:
        for date_str in dates_lst:
            silver.process_silver_table(date_str, table, BRONZE_DIR, SILVER_DIR, spark)

    # gold backfill (feature store + label store)
    print("\n=== gold ===")
    for date_str in dates_lst:
        gold.process_gold_feature_store(date_str, SILVER_DIR, GOLD_DIR, spark)
        gold.process_gold_label_store(date_str, SILVER_DIR, GOLD_DIR, spark)

    # quick read back so we can see the gold tables landed.
    # one parquet folder per snapshot, so glob them all to read the whole store.
    fs = spark.read.parquet(os.path.join(GOLD_DIR, "feature_store", "*.parquet"))
    ls = spark.read.parquet(os.path.join(GOLD_DIR, "label_store", "*.parquet"))
    print("\ngold feature_store rows:", fs.count())
    print("gold label_store rows:", ls.count())

    spark.stop()

    # data splitting happens after gold is built
    print("\n=== split ===")
    split_data.run_split()

    print("\n--- pipeline done ---")


if __name__ == "__main__":
    main()
