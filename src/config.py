import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPLIT_DIR = os.path.join(BASE_DIR, "datamart", "gold", "splits")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MANIFEST_PATH = os.path.join(REPORTS_DIR, "feature_manifest.json")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

RANDOM_STATE = 42

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_MODEL_NAME = "harmful_video_detector"

PROMOTION_RECALL_THRESHOLD = 0.90
PROMOTION_PRECISION_THRESHOLD = 0.85

# --- v2 flat composite weights (deprecated in v3; kept for reference only) ---
COMPOSITE_WEIGHTS = {
    "text": 0.45,
    "ocr": 0.20,
    "asr": 0.20,
    "metadata": 0.15,
}

# --- v3 decoupled scoring ---
# Stage-1 text gate: below this a candidate is not a duplicate and metadata is
# never applied (metadata corroborates a real text match, never rescues one).
TEXT_CUTOFF = 0.55
# Stage-2 corroboration budget: max lift metadata can add on top of a text match.
METADATA_LIFT = 0.15
# Same-medium feature similarity at/above this = literal reupload (same_asset);
# below it = re-authored copy.
FEATURE_HIGH = 0.80

# --- v3 decision thresholds (NO AUTO-REMOVAL) ---
# Operates on match_confidence. The top tier routes to URGENT human review, not
# an automated REMOVE; every removal is a human decision.
LLM_THRESHOLDS = {
    "allow_below": 0.30,       # below -> ALLOW (not a duplicate worth a human)
    "review_below": 0.60,      # [0.30, 0.60) -> HUMAN REVIEW (normal)
    "llm_urgent_above": 0.70,  # LLM second opinion >= this -> HUMAN REVIEW (urgent)
    "llm_allow_below": 0.50,   # LLM second opinion < this -> ALLOW
}

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN",
    "postgresql://pipeline:pipeline@localhost:5432/harmful_video",
)


def load_manifest():
    with open(MANIFEST_PATH, "r") as f:
        return json.load(f)
