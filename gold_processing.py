import argparse

import pyspark

import utils.data_processing_gold_table as gold

# run one gold snapshot from the command line. builds both the feature store
# and the label store for that date.
# example: python gold_processing.py --snapshotdate 2025-01-01

SILVER_DIR = "datamart/silver/"
GOLD_DIR = "datamart/gold/"


def main(snapshotdate):
    print("\n--- starting gold job:", snapshotdate, "---\n")
    spark = pyspark.sql.SparkSession.builder \
        .appName("gold_job") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    gold.process_gold_feature_store(snapshotdate, SILVER_DIR, GOLD_DIR, spark)
    gold.process_gold_label_store(snapshotdate, SILVER_DIR, GOLD_DIR, spark)

    spark.stop()
    print("\n--- gold job done ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run gold for one snapshot")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
