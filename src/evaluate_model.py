import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from src.config import REPORTS_DIR, load_manifest
from src.train_model import build_feature_matrix, load_artifacts


def evaluate_split(model, scaler, split_name, manifest):
    X, y, df = build_feature_matrix(split_name, manifest)
    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "f1": round(float(f1_score(y, y_pred)), 4),
        "recall": round(float(recall_score(y, y_pred)), 4),
        "precision": round(float(precision_score(y, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y, y_prob)), 4),
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
    }

    cm = confusion_matrix(y, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics, y, y_pred, y_prob, cm


def save_confusion_matrix(cm, split_name):
    os.makedirs(os.path.join(REPORTS_DIR, "plots"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["clean", "harmful"],
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix ({split_name})")
    path = os.path.join(REPORTS_DIR, "plots", f"confusion_matrix_{split_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved confusion matrix: {path}")
    return path


def print_metrics(split_name, metrics):
    print(f"\n{'='*40}")
    print(f"  {split_name.upper()} SET EVALUATION")
    print(f"{'='*40}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Samples:   {metrics['n_samples']} (pos={metrics['n_positive']}, neg={metrics['n_negative']})")


def check_overfitting(train_metrics, val_metrics):
    print(f"\n{'='*40}")
    print("  OVERFITTING CHECK (train vs val)")
    print(f"{'='*40}")
    for metric in ["accuracy", "f1", "recall", "precision", "roc_auc"]:
        diff = train_metrics[metric] - val_metrics[metric]
        flag = " *** OVERFIT" if diff > 0.10 else ""
        print(f"  {metric:12s}  train={train_metrics[metric]:.4f}  val={val_metrics[metric]:.4f}  gap={diff:+.4f}{flag}")


def run_evaluation():
    manifest = load_manifest()
    model, scaler = load_artifacts()

    all_metrics = {}
    for split_name in ["train", "val", "test"]:
        metrics, y, y_pred, y_prob, cm = evaluate_split(model, scaler, split_name, manifest)
        print_metrics(split_name, metrics)
        save_confusion_matrix(cm, split_name)
        all_metrics[split_name] = metrics

        print(f"\n{classification_report(y, y_pred, target_names=['clean', 'harmful'])}")

    check_overfitting(all_metrics["train"], all_metrics["val"])

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "model_evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nsaved evaluation report: {report_path}")

    print("\n--- evaluation done ---")
    return all_metrics


if __name__ == "__main__":
    run_evaluation()
