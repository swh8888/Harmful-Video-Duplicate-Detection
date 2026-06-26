from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor

from src.config import POSTGRES_DSN

CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id                SERIAL PRIMARY KEY,
    video_id          TEXT NOT NULL,
    layer1_label      TEXT,
    harmful_prob      FLOAT,
    matched_seed_id   TEXT,
    best_pair         TEXT,
    is_cross_medium   BOOLEAN,
    medium_transition TEXT,
    text_score        FLOAT,
    feature_score     FLOAT,
    metadata_score    FLOAT,
    match_confidence  FLOAT,
    duplicate_type    TEXT,
    lane              TEXT,
    decision          TEXT,
    priority          TEXT,
    llm_score         FLOAT,
    reasoning         TEXT,
    reviewer_verdict  TEXT,
    reviewed_by       TEXT,
    reviewed_at       TIMESTAMP,
    created_at        TIMESTAMP DEFAULT NOW()
);
"""

CREATE_REVIEWER_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS reviewer_log (
    id          SERIAL PRIMARY KEY,
    video_id    TEXT NOT NULL,
    verdict     TEXT NOT NULL,
    reviewed_by TEXT,
    reviewed_at TIMESTAMP DEFAULT NOW()
);
"""


def get_conn():
    return psycopg2.connect(POSTGRES_DSN)


def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_PREDICTIONS_TABLE)
            cur.execute(CREATE_REVIEWER_LOG_TABLE)
        conn.commit()
    print("prediction log tables initialized")


def log_prediction(row: dict):
    """Insert one inference result row into the predictions table."""
    fields = [
        "video_id", "layer1_label", "harmful_prob", "matched_seed_id", "best_pair",
        "is_cross_medium", "medium_transition", "text_score", "feature_score",
        "metadata_score", "match_confidence", "duplicate_type", "lane",
        "decision", "priority", "llm_score", "reasoning",
    ]
    cols = ", ".join(fields)
    placeholders = ", ".join(["%s"] * len(fields))
    values = [row.get(f) for f in fields]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO predictions ({cols}) VALUES ({placeholders})",
                values,
            )
        conn.commit()


def log_prediction_batch(rows):
    """Bulk-insert a list of inference result dicts."""
    for row in rows:
        log_prediction(row)
    print(f"logged {len(rows)} predictions")


def log_reviewer_verdict(video_id, verdict, reviewed_by):
    """Record a human reviewer override and update the predictions row."""
    now = datetime.now()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO reviewer_log (video_id, verdict, reviewed_by, reviewed_at) "
                "VALUES (%s, %s, %s, %s)",
                (video_id, verdict, reviewed_by, now),
            )
            cur.execute(
                "UPDATE predictions SET reviewer_verdict=%s, reviewed_by=%s, reviewed_at=%s "
                "WHERE video_id=%s",
                (verdict, reviewed_by, now, video_id),
            )
        conn.commit()
    print(f"logged reviewer verdict: {video_id} -> {verdict}")


def fetch_recent_predictions(limit=500):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            return cur.fetchall()
