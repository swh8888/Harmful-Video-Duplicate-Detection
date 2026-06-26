import os
import json
import numpy as np
from datetime import datetime

from src.config import REPORTS_DIR
from src.persistence.prediction_logger import fetch_recent_predictions

PSI_BINS = 10
PSI_ALERT_THRESHOLD = 0.20
HARMFUL_RATE_ALERT = 0.30
OVERRIDE_RATE_ALERT = 0.20
MISSING_MODALITY_ALERT = 0.50


def psi(reference, current, bins=PSI_BINS):
    """Population Stability Index between two score distributions.

    PSI < 0.10: no drift
    PSI 0.10-0.20: moderate drift
    PSI > 0.20: significant drift
    """
    ref = np.array(reference, dtype=float)
    cur = np.array(current, dtype=float)

    if len(ref) == 0 or len(cur) == 0:
        return None

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = (ref_counts + 1e-6) / (len(ref) + 1e-6 * bins)
    cur_pct = (cur_counts + 1e-6) / (len(cur) + 1e-6 * bins)

    psi_val = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi_val, 4)


def compute_metrics(rows):
    if not rows:
        return {}

    decisions = [r["decision"] for r in rows if r.get("decision")]
    confidences = [r["match_confidence"] for r in rows if r.get("match_confidence") is not None]
    harmful_labels = [r["layer1_label"] for r in rows if r.get("layer1_label")]
    cross_medium = sum(1 for r in rows if r.get("is_cross_medium"))
    feature_missing = sum(1 for r in rows if r.get("feature_score") is None)
    overrides = [r for r in rows if r.get("reviewer_verdict") is not None]

    n = len(rows)
    return {
        "n": n,
        "harmful_rate": round(sum(1 for l in harmful_labels if l == "harmful") / max(n, 1), 4),
        "review_rate": round(decisions.count("HUMAN REVIEW") / max(n, 1), 4),
        "allow_rate": round(decisions.count("ALLOW") / max(n, 1), 4),
        "urgent_rate": round(sum(1 for r in rows if r.get("priority") == "urgent") / max(n, 1), 4),
        "cross_medium_rate": round(cross_medium / max(n, 1), 4),
        "mean_match_confidence": round(float(np.mean(confidences)), 4) if confidences else None,
        "std_match_confidence": round(float(np.std(confidences)), 4) if confidences else None,
        "feature_missing_rate": round(feature_missing / max(n, 1), 4),
        "override_rate": round(len(overrides) / max(n, 1), 4),
        "match_confidences": confidences,
    }


def check_alerts(current_metrics, reference_metrics):
    alerts = []

    harmful_diff = abs(
        current_metrics.get("harmful_rate", 0) - reference_metrics.get("harmful_rate", 0)
    )
    if harmful_diff > HARMFUL_RATE_ALERT:
        alerts.append({
            "type": "harmful_rate_drift",
            "message": f"Harmful rate changed by {harmful_diff:.2%}",
            "severity": "high",
        })

    if current_metrics.get("override_rate", 0) > OVERRIDE_RATE_ALERT:
        alerts.append({
            "type": "high_override_rate",
            "message": f"Analysts overriding {current_metrics['override_rate']:.2%} of decisions",
            "severity": "medium",
        })

    if current_metrics.get("feature_missing_rate", 0) > MISSING_MODALITY_ALERT:
        alerts.append({
            "type": "feature_score_missing_high",
            "message": f"Feature score unavailable (cross-medium/title) in "
                       f"{current_metrics['feature_missing_rate']:.2%} of matches",
            "severity": "low",
        })

    ref_scores = reference_metrics.get("match_confidences", [])
    cur_scores = current_metrics.get("match_confidences", [])
    if ref_scores and cur_scores:
        psi_val = psi(ref_scores, cur_scores)
        if psi_val is not None and psi_val > PSI_ALERT_THRESHOLD:
            alerts.append({
                "type": "score_distribution_drift",
                "message": f"PSI on match_confidence = {psi_val:.3f} (threshold {PSI_ALERT_THRESHOLD})",
                "severity": "high",
                "psi": psi_val,
            })

    return alerts


def run_monitoring_report():
    rows = fetch_recent_predictions(limit=1000)
    if not rows:
        print("no prediction data available")
        return {}

    n = len(rows)
    half = n // 2
    reference_rows = list(rows)[half:]
    current_rows = list(rows)[:half]

    reference_metrics = compute_metrics(reference_rows)
    current_metrics = compute_metrics(current_rows)

    alerts = check_alerts(current_metrics, reference_metrics)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "current_window_n": current_metrics.get("n"),
        "reference_window_n": reference_metrics.get("n"),
        "current": {k: v for k, v in current_metrics.items() if k != "match_confidences"},
        "reference": {k: v for k, v in reference_metrics.items() if k != "match_confidences"},
        "alerts": alerts,
        "alert_count": len(alerts),
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "monitoring_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved monitoring report: {out_path}")

    if alerts:
        print(f"\n*** {len(alerts)} ALERT(S) ***")
        for a in alerts:
            print(f"  [{a['severity'].upper()}] {a['message']}")
    else:
        print("no alerts")

    return report


if __name__ == "__main__":
    run_monitoring_report()
