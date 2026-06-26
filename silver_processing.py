import argparse

import pyspark

import utils.data_processing_silver_table as silver

# run one silver snapshot from the command line.
# example: python silver_processing.py --snapshotdate 2025-01-01

TABLES = ["label", "ocr", "asr", "metadata"]
BRONZE_DIR = "datamart/bronze/"
SILVER_DIR = "datamart/silver/"


def main(snapshotdate, table):
    print("\n--- starting silver job:", snapshotdate, "---\n")
    spark = pyspark.sql.SparkSession.builder \
        .appName("silver_job") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    tables = TABLES if table == "all" else [table]
    for t in tables:
        silver.process_silver_table(snapshotdate, t, BRONZE_DIR, SILVER_DIR, spark)

    spark.stop()
    print("\n--- silver job done ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run silver for one snapshot")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--table", type=str, default="all", help="label / ocr / asr / metadata / all")
    args = parser.parse_args()
    main(args.snapshotdate, args.table)
