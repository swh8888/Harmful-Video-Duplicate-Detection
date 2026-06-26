import argparse

import pyspark

import utils.data_processing_bronze_table as bronze

# run one bronze snapshot from the command line. handy for an orchestrator
# (airflow etc.) that fires one task per snapshot date.
# example: python bronze_processing.py --snapshotdate 2025-01-01

TABLES = ["label", "ocr", "asr", "metadata"]
BRONZE_DIR = "datamart/bronze/"


def main(snapshotdate, table):
    print("\n--- starting bronze job:", snapshotdate, "---\n")
    spark = pyspark.sql.SparkSession.builder \
        .appName("bronze_job") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    tables = TABLES if table == "all" else [table]
    for t in tables:
        bronze.process_bronze_table(snapshotdate, t, BRONZE_DIR, spark)

    spark.stop()
    print("\n--- bronze job done ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run bronze for one snapshot")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--table", type=str, default="all", help="label / ocr / asr / metadata / all")
    args = parser.parse_args()
    main(args.snapshotdate, args.table)
