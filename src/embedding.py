import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import (
    SPLIT_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
)


def load_split(name):
    path = os.path.join(SPLIT_DIR, name + ".parquet")
    return pd.read_parquet(path)


def embed_texts(texts, model=None):
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(
        texts.tolist(),
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def save_embeddings(embeddings, name):
    out_path = os.path.join(SPLIT_DIR, name + "_embeddings.npy")
    np.save(out_path, embeddings)
    print(f"saved {out_path}  shape: {embeddings.shape}")
    return out_path


def load_embeddings(name):
    path = os.path.join(SPLIT_DIR, name + "_embeddings.npy")
    return np.load(path)


def run_embedding():
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"loaded model: {EMBEDDING_MODEL_NAME}  dim: {EMBEDDING_DIM}")

    for split_name in ["train", "val", "test"]:
        df = load_split(split_name)
        print(f"\n--- {split_name} ---")
        print(f"rows: {len(df)}")

        texts = df["combined_text"].fillna("")
        embeddings = embed_texts(texts, model=model)

        assert embeddings.shape == (len(df), EMBEDDING_DIM), (
            f"expected ({len(df)}, {EMBEDDING_DIM}), got {embeddings.shape}"
        )

        save_embeddings(embeddings, split_name)

    print("\n--- embedding done ---")


if __name__ == "__main__":
    run_embedding()
