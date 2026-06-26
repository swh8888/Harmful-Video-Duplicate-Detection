import os

import mlflow
import mlflow.sklearn

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_MODEL_NAME,
    PROMOTION_RECALL_THRESHOLD,
    PROMOTION_PRECISION_THRESHOLD,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
    RANDOM_STATE,
    REPORTS_DIR,
    load_manifest,
)
from src.train_model import load_artifacts
from src.evaluate_model import run_evaluation


def register_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("harmful_video_detection_v2")

    model, scaler = load_artifacts()
    all_metrics = run_evaluation()

    val_metrics = all_metrics["val"]

    with mlflow.start_run(run_name="harmful_detector_logreg") as run:
        manifest = load_manifest()
        mlflow.log_param("embedding_model", EMBEDDING_MODEL_NAME)
        mlflow.log_param("embedding_dim", EMBEDDING_DIM)
        mlflow.log_param("n_numeric_features", len(manifest["numeric_features"]))
        mlflow.log_param("total_features", EMBEDDING_DIM + len(manifest["numeric_features"]))
        mlflow.log_param("classifier", "LogisticRegression")
        mlflow.log_param("C", 1.0)
        mlflow.log_param("solver", "lbfgs")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("random_state", RANDOM_STATE)

        for split_name, metrics in all_metrics.items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{split_name}_{metric_name}", value)

        plots_dir = os.path.join(REPORTS_DIR, "plots")
        if os.path.isdir(plots_dir):
            mlflow.log_artifacts(plots_dir, artifact_path="plots")

        report_path = os.path.join(REPORTS_DIR, "model_evaluation_report.json")
        if os.path.isfile(report_path):
            mlflow.log_artifact(report_path)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME,
        )

        run_id = run.info.run_id
        print(f"\nMLflow run: {run_id}")

    passes_recall = val_metrics["recall"] >= PROMOTION_RECALL_THRESHOLD
    passes_precision = val_metrics["precision"] >= PROMOTION_PRECISION_THRESHOLD

    print("\n--- promotion gate ---")
    print(f"  val recall:    {val_metrics['recall']:.4f}  (threshold: {PROMOTION_RECALL_THRESHOLD})  {'PASS' if passes_recall else 'FAIL'}")
    print(f"  val precision: {val_metrics['precision']:.4f}  (threshold: {PROMOTION_PRECISION_THRESHOLD})  {'PASS' if passes_precision else 'FAIL'}")

    if passes_recall and passes_precision:
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["None"])
        if latest_versions:
            version = latest_versions[0].version
            client.transition_model_version_stage(
                name=MLFLOW_MODEL_NAME,
                version=version,
                stage="Production",
            )
            print(f"\n  Model v{version} promoted to Production")
        else:
            print("\n  WARNING: no model version found to promote")
    else:
        print("\n  Model NOT promoted -- does not meet gate thresholds")

    print("\n--- mlflow registration done ---")
    return all_metrics


if __name__ == "__main__":
    register_model()
