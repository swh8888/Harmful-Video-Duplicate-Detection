# Harmful Video Duplicate Detection - MLOps Pipeline

Group 2 · CS611 Machine Learning Engineering

A two-layer ML pipeline that detects harmful video content and identifies duplicate uploads.
Layer 1 classifies incoming videos as harmful or clean. Layer 2 runs a four-stage similarity
engine against a seed corpus to detect coordinated reposting campaigns.

---

## Quick Start

```bash
# Step 1: generate synthetic raw data (no Docker needed)
python generate_data.py

# Step 2: run the full stack
docker compose up --build
```

When it finishes you have:
- Medallion data pipeline (Bronze / Silver / Gold) with train/val/test splits
- MLflow model registry at http://localhost:5000
- Airflow orchestration at http://localhost:8080 (admin / admin)
- Streamlit analyst dashboard at http://localhost:8501
- PostgreSQL prediction log at localhost:5432

---

## Architecture

```
                      ┌─────────────────────────────────────────────────────┐
                      │  LAYER 1: Harmful Detection                         │
  Incoming video ───► │  Sentence Transformer (all-MiniLM-L6-v2, 384-dim)  │
                      │  + Logistic Regression (419 features)               │
                      │  MLflow registry · promotion gate: recall≥0.90      │
                      └──────────────────┬──────────────────────────────────┘
                                         │ harmful only
                      ┌──────────────────▼──────────────────────────────────┐
                      │  LAYER 2: Duplicate Detection Engine                │
                      │  Step 5: Text match (FAISS) — gate ≥ 0.55          │
                      │  Step 6: OCR scoring   (feature sim → type only)   │
                      │  Step 7: ASR scoring   (feature sim → type only)   │
                      │  Step 8: Metadata      (temporal, capped lift 0.15)│
                      │  Step 9: → match_confidence + duplicate_type      │
                      └──────────────────┬──────────────────────────────────┘
                                         │
                      ┌──────────────────▼──────────────────────────────────┐
                      │  Step 10: Ollama LLM Verifier (llama3.2)           │
                      │  score < 0.30  → ALLOW                             │
                      │  0.30 – 0.59   → HUMAN REVIEW                     │
                      │  ≥ 0.60        → invoke LLM                       │
                      │    LLM ≥ 0.70  → URGENT HUMAN REVIEW            │
                      │    LLM 0.50-0.69 → REVIEW                         │
                      │    LLM < 0.50  → ALLOW                            │
                      └──────────────────┬──────────────────────────────────┘
                                         │
                      ┌──────────────────▼──────────────────────────────────┐
                      │  Step 12: Streamlit Dashboard                       │
                      │  T&S analyst reviews REVIEW queue                  │
                      │  Per-feature SHAP breakdown · REMOVE / ALLOW buttons│
                      │  Verdicts written to PostgreSQL → fed back to train │
                      └─────────────────────────────────────────────────────┘
```

### Data Flow (Medallion Architecture)

```
generate_data.py  (host, no docker)
       │
       ▼
  data/*.csv  (4 raw sources, keyed by video_id + snapshot_date)
       │
       ▼
   BRONZE   raw ingestion, append-only, one snapshot at a time
       │
       ▼
   SILVER   type-cast, clean, normalise text (no imputation)
       │
       ▼
   GOLD     impute, one-hot encode, join 4 sources into one row
            → feature_store  (35 numeric features + combined_text)
            → label_store    (binary label)
       │
       ▼
  split_data.py  stratified 80/10/10 · ML readiness checks · feature manifest
       │
       ▼
  src/embedding.py         384-dim sentence embeddings per split
  src/train_model.py       LogisticRegression on 419 features
  src/evaluate_model.py    metrics + confusion matrix
  src/mlflow_registry.py   MLflow log + promotion gate
       │
       ▼
  src/run_inference.py     batch inference: Layer 1 + Layer 2 + LLM
  src/dashboard/app.py     Streamlit analyst UI
  src/logging/             PostgreSQL prediction log
  src/monitoring/          drift detection (PSI on score distribution)
```

---

## Running Each Stage Manually

```bash
# Data pipeline
make generate
make process        # bronze -> silver -> gold -> split (full backfill)

# Model training
make embed          # generate sentence embeddings for train/val/test
make train          # fit LogisticRegression
make evaluate       # compute metrics + confusion matrices
make register       # log to MLflow + promote if gate passes

# Inference
make infer          # batch inference on test split
make dashboard      # open Streamlit dashboard at http://localhost:8501
make monitor        # compute drift report from PostgreSQL

# One snapshot at a time (for Airflow task-level runs)
python bronze_processing.py --snapshotdate 2025-01-01
python silver_processing.py --snapshotdate 2025-01-01
python gold_processing.py   --snapshotdate 2025-01-01
```

---

## Project Layout

```
harmful-video-pipeline/
├── generate_data.py            synthetic data generator
├── main.py                     full medallion backfill orchestrator
├── bronze_processing.py        CLI: one bronze snapshot
├── silver_processing.py        CLI: one silver snapshot
├── gold_processing.py          CLI: one gold snapshot
├── split_data.py               stratified split + ML readiness checks
├── utils/
│   ├── data_processing_bronze_table.py
│   ├── data_processing_silver_table.py
│   └── data_processing_gold_table.py
├── src/
│   ├── config.py               shared constants + env vars
│   ├── embedding.py            sentence transformer embeddings (Step 4a)
│   ├── train_model.py          LogisticRegression training (Step 4b)
│   ├── evaluate_model.py       metrics + confusion matrix (Step 4c)
│   ├── mlflow_registry.py      MLflow log + promotion gate (Step 4d)
│   ├── llm_verifier.py         Ollama LLM second opinion (Step 10)
│   ├── run_inference.py        batch inference pipeline (Step 11)
│   ├── matching/
│   │   └── text_matching.py    FAISS text similarity (Step 5)
│   ├── scoring/
│   │   ├── ocr_scoring.py      OCR feature similarity (Step 6)
│   │   ├── asr_scoring.py      ASR feature similarity (Step 7)
│   │   ├── metadata_scoring.py metadata corroboration (Step 8)
│   │   └── composite.py        weighted composite score (Step 9)
│   ├── dashboard/
│   │   └── app.py              Streamlit analyst UI (Step 12)
│   ├── persistence/
│   │   └── prediction_logger.py PostgreSQL prediction log (Step 13)
│   └── monitoring/
│       └── drift_monitor.py    PSI drift detection (Step 14)
├── dags/
│   ├── dag1_data_pipeline.py   Airflow DAG 1: data
│   ├── dag2_model_training.py  Airflow DAG 2: train
│   └── dag3_inference.py       Airflow DAG 3: inference + monitor
├── data/                       raw CSVs (generated, gitignored)
├── datamart/                   bronze / silver / gold output
├── models/                     trained model artifacts
├── reports/                    evaluation + monitoring reports
├── Dockerfile
├── docker-compose.yaml         6 services: pipeline, mlflow, airflow,
│                               streamlit, ollama, postgres
├── Makefile
└── requirements.txt
```

---

## Services

| Service | URL | Purpose |
|---|---|---|
| MLflow | http://localhost:5000 | Model registry + experiment tracking |
| Airflow | http://localhost:8080 | DAG orchestration (admin/admin) |
| Streamlit | http://localhost:8501 | Analyst review dashboard |
| PostgreSQL | localhost:5432 | Prediction log + Airflow metadata |
| Ollama | http://localhost:11434 | Local LLM (llama3.2) |

Pull the Ollama model after first startup:
```bash
make pull-ollama
```

---

## Design Choices

- **Non-LLM similarity first.** FAISS + feature scoring runs in milliseconds and is fully auditable.
  LLM verification only triggers when composite score >= 0.60, keeping costs low.
- **Composite score re-normalisation.** When OCR or ASR is missing, the remaining signal weights
  are proportionally rescaled so the final score always sits in [0, 1].
- **MLflow promotion gate.** recall >= 0.90 AND precision >= 0.85 (v2). A model that passes the
  gate is promoted to the Production stage automatically; one that fails is logged but not deployed.
- **Feedback loop.** Human reviewer verdicts (REMOVE / ALLOW overrides) are written to PostgreSQL.
  The intended design is for DAG 2 to pick these up on the next retraining run, closing the
  annotation loop (see Known Limitations for the current execution caveat).
- **CPU only.** LogisticRegression + sentence-transformers inference are fast on CPU; no GPU needed.
  Docker base image stays as python:3.12-slim.

## Assumptions

- Data is fully synthetic. Harmful text uses generic placeholder wording.
- OCR and ASR feature values are simulated, not extracted from real video.
- The Ollama llama3.2 model is used for LLM verification; swap OLLAMA_MODEL env var to change it.
- Median imputation is computed per snapshot in the Gold layer.

## Known Limitations / Notes

- **Airflow DAGs are design- and parse-validated, not executable as shipped.** All three DAGs load
  in the Airflow UI with no import errors, but the stock `apache/airflow:2.9.1` worker does not have
  the project's ML dependencies (sentence-transformers, scikit-learn, mlflow, faiss, shap) or
  PySpark + Java installed. Triggering a DAG therefore fails with `ModuleNotFoundError` (DAG 2 / 3)
  or `java: command not found` (DAG 1). Every stage is proven individually in the Dockerfile-built
  `pipeline` / `streamlit` image. To make the DAGs run end to end, build a custom Airflow image that
  extends the base image with `requirements.txt` plus a JDK.
- **Pull the Ollama model once before live LLM verification.** Run `make pull-ollama` to download
  `llama3.2`. Until then the LLM verifier (Step 10) fails safe to HUMAN REVIEW by design, so the
  pipeline still runs without it.
- **MLflow is pinned to `2.21.3`.** MLflow 3.x removes the model-registry *stages* API used by
  `src/mlflow_registry.py` and adds a host-header security middleware that blocks cross-container
  calls. The `mlflow` service installs `mlflow==2.21.3` and uses its own **`mlflow_db`** database
  (separate from the `harmful_video` database) to avoid an Alembic migration clash with Airflow's
  metadata tables.
