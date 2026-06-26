"""DAG 3: Inference pipeline — batch inference -> log predictions -> monitor drift.

Runs on a daily schedule against the test split (or new incoming data).
Sends results to PostgreSQL and generates a monitoring report.
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/opt/airflow/project"

with DAG(
    dag_id="dag3_inference",
    description="Daily batch inference: classify -> duplicate match -> log -> monitor",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["inference", "monitoring"],
) as dag:

    infer = BashOperator(
        task_id="run_batch_inference",
        bash_command=f"cd {PROJECT} && python -m src.run_inference",
    )

    log_preds = BashOperator(
        task_id="log_predictions",
        bash_command=f"cd {PROJECT} && python -c \""
            "import pandas as pd, os;"
            "from src.config import REPORTS_DIR;"
            "from src.persistence.prediction_logger import init_db, log_prediction_batch;"
            "init_db();"
            "path = os.path.join(REPORTS_DIR, 'inference_results_test.parquet');"
            "df = pd.read_parquet(path);"
            "log_prediction_batch(df.to_dict('records'))"
            "\"",
    )

    monitor = BashOperator(
        task_id="run_drift_monitor",
        bash_command=f"cd {PROJECT} && python -m src.monitoring.drift_monitor",
    )

    infer >> log_preds >> monitor
