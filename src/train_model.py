import os

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, RANDOM_STATE, load_manifest
from src.embedding import load_split, load_embeddings


def build_feature_matrix(split_name, manifest):
    df = load_split(split_name)
    embeddings = load_embeddings(split_name)

    numeric_cols = manifest["numeric_features"]
    X_numeric = df[numeric_cols].values.astype(np.float32)

    X = np.hstack([embeddings, X_numeric])
    y = df["label"].values.astype(int)

    return X, y, df


def train(X_train, y_train, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
    else:
        X_scaled = scaler.transform(X_train)

    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(X_scaled, y_train)
    return model, scaler


def save_artifacts(model, scaler):
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "harmful_detector.joblib")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"saved model: {model_path}")
    print(f"saved scaler: {scaler_path}")
    return model_path, scaler_path


def load_artifacts():
    model_path = os.path.join(MODELS_DIR, "harmful_detector.joblib")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def run_training():
    manifest = load_manifest()
    n_numeric = len(manifest["numeric_features"])
    print(f"numeric features: {n_numeric}")
    print(f"total features: 384 (embedding) + {n_numeric} (numeric) = {384 + n_numeric}")

    X_train, y_train, train_df = build_feature_matrix("train", manifest)
    print(f"\ntrain set: {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f"train label dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    model, scaler = train(X_train, y_train)

    train_acc = model.score(scaler.transform(X_train), y_train)
    print(f"\ntrain accuracy: {train_acc:.4f}")

    save_artifacts(model, scaler)
    print("\n--- training done ---")
    return model, scaler


if __name__ == "__main__":
    run_training()
