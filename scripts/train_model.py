from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATASET_PATH = Path("Lead Scoring.csv")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_PATH = ARTIFACT_DIR / "lead_model.pkl"

# Minimal features for API demo. You can expand later.
FEATURE_COLUMNS = [
    "Total Time Spent on Website",
    "Page Views Per Visit",
    "TotalVisits",
]
TARGET_COLUMN = "Converted"


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    clean_df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    clean_df[FEATURE_COLUMNS] = clean_df[FEATURE_COLUMNS].fillna(clean_df[FEATURE_COLUMNS].median())
    clean_df = clean_df.dropna(subset=[TARGET_COLUMN])

    X = clean_df[FEATURE_COLUMNS]
    y = clean_df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy: {acc:.4f}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_PATH)
    print(f"Saved model to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
