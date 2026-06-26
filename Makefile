# quick commands. DATE is only needed for the single snapshot targets.
DATE ?= 2025-01-01

# --- data pipeline (Steps 1-3) ---
generate:
	python generate_data.py

bronze:
	python bronze_processing.py --snapshotdate $(DATE)

silver:
	python silver_processing.py --snapshotdate $(DATE)

gold:
	python gold_processing.py --snapshotdate $(DATE)

process:
	python main.py

split:
	python split_data.py

# --- model training (Step 4) ---
embed:
	python -m src.embedding

train:
	python -m src.train_model

evaluate:
	python -m src.evaluate_model

register:
	python -m src.mlflow_registry

# --- inference + dashboard (Steps 10-12) ---
infer:
	python -m src.run_inference

dashboard:
	streamlit run src/dashboard/app.py

# --- monitoring (Step 14) ---
monitor:
	python -m src.monitoring.drift_monitor

# --- infrastructure ---
docker:
	docker compose up --build

docker-down:
	docker compose down -v

pull-ollama:
	docker exec harmful_video_ollama ollama pull llama3.2

clean:
	rm -rf datamart reports models
	find . -type d -name __pycache__ -exec rm -rf {} +
