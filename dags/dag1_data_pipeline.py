"""DAG 1: Data pipeline — generate -> bronze -> silver -> gold -> split.

Runs the full medallion backfill and train/val/test split.
Triggered manually or on a monthly schedule matching the snapshot cadence.
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/opt/airflow/project"

with DAG(
    dag_id="dag1_data_pipeline",
    description="Medallion data pipeline: synthetic data -> bronze -> silver -> gold -> split",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@monthly",
    catchup=False,
    tags=["data", "medallion"],
) as dag:

    generate = BashOperator(
        task_id="generate_synthetic_data",
        bash_command=f"cd {PROJECT} && USE_CORPUS=1 python generate_data.py",
    )

    medallion = BashOperator(
        task_id="run_medallion_pipeline",
        bash_command=f"cd {PROJECT} && python main.py",
    )

    generate >> medallion
