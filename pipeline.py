"""
Spaceship Titanic, Pipeline Trainer

Trains the pipeline locally and saves a single pkl to artifacts/

Usage:
    python train_pipeline.py
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from utils_folder.helper import make_pipeline

RANDOM_STATE = 42
ACCURACY_THRESHOLD = 0.75
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def main():
    print("=" * 60)
    print("  Spaceship Titanic, Training Pipeline")
    print("=" * 60)

    # 1 Load raw data
    print("\n[Step 1] Loading raw data")
    df = pd.read_csv("data/raw/train.csv")
    print(f"  Shape: {df.shape}")

    # 2 Separate target
    y = df["Transported"].astype(int)
    X = df.drop(columns=["Transported"])

    # 3 Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"  Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}")

    # 4 Train
    print("\n[Step 2] Training")
    pipe = make_pipeline()
    pipe.fit(X_train, y_train)
    print("  Pipeline fitted")

    # 5 Evaluate
    print("\n[Step 3] Evaluating")
    y_pred = pipe.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_val, y_pred))

    # 6 Save single pkl
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    pkl_path = os.path.join(ARTIFACTS_DIR, "logistic_regression_pipeline.pkl")
    joblib.dump(pipe, pkl_path)
    print(f"  Saved to {pkl_path}")

    # 7 Verdict
    print("\n" + "=" * 60)
    if acc >= ACCURACY_THRESHOLD:
        print(f"  APPROVED  (accuracy={acc:.4f} >= {ACCURACY_THRESHOLD})")
    else:
        print(f"  REJECTED  (accuracy={acc:.4f}  < {ACCURACY_THRESHOLD})")
    print("=" * 60)


if __name__ == "__main__":
    main()
