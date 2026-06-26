"""DAG 2: Model training — embed -> train -> evaluate -> register.

Triggered after DAG 1 completes or manually when new labelled data arrives.
The MLflow promotion gate (recall >= 0.90, precision >= 0.85) runs inside
mlflow_registry.py — a failed gate logs the run but does NOT promote to Production.
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/opt/airflow/project"

with DAG(
    dag_id="dag2_model_training",
    description="Train harmful detector: embed -> train -> evaluate -> MLflow register",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["training", "mlflow"],
) as dag:

    embed = BashOperator(
        task_id="generate_embeddings",
        bash_command=f"cd {PROJECT} && python -m src.embedding",
    )

    train = BashOperator(
        task_id="train_model",
        bash_command=f"cd {PROJECT} && python -m src.train_model",
    )

    register = BashOperator(
        task_id="evaluate_and_register",
        bash_command=f"cd {PROJECT} && python -m src.mlflow_registry",
    )

    embed >> train >> register
