import os

from pyspark.sql.functions import col

# bronze layer.
# job here is dead simple on purpose: take the raw source feed, grab only the
# rows for the snapshot we are processing, and land them as-is. no cleaning,
# no type fixing. bronze is meant to be a faithful copy of what we received so
# we can always replay from it later.

# the four raw feeds the generator produces, and where they live
SOURCE_FILES = {
    "label": "data/dataset_label.csv",
    "ocr": "data/dataset_ocr.csv",
    "asr": "data/dataset_asr.csv",
    "metadata": "data/dataset_metadata.csv",
}


def process_bronze_table(snapshot_date_str, table_name, bronze_directory, spark):
    # which raw feed are we ingesting
    source_path = SOURCE_FILES[table_name]

    # connect to source - IRL this would be a db pull or an object store read.
    # multiLine + escape so commas, quotes and newlines inside the text fields
    # (titles, ocr and asr text can all contain them) never split one row in two.
    df = spark.read.csv(source_path, header=True, inferSchema=True, multiLine=True, escape='"')

    # keep only this snapshot. snapshot_date in the raw file is the upload month.
    df = df.filter(col("snapshot_date") == snapshot_date_str)
    print(snapshot_date_str, table_name, "row count:", df.count())

    # land it. one csv per table per snapshot so partitions stay tidy.
    out_dir = os.path.join(bronze_directory, table_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    partition_name = "bronze_" + table_name + "_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = os.path.join(out_dir, partition_name)
    df.toPandas().to_csv(filepath, index=False)
    print("saved to:", filepath)

    return df
