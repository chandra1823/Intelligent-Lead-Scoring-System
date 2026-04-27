from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

DATASET_PATH = Path("Lead Scoring.csv")
ARTIFACT_DIR = Path("artifacts")

FEATURE_COLUMNS = [
    "Total Time Spent on Website",
    "Page Views Per Visit",
    "TotalVisits",
]
TARGET_COLUMN = "Converted"


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return {"random_forest": 0.34, "xgboost": 0.33, "lightgbm": 0.33}
    return {k: v / total for k, v in weights.items()}


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

    models = {
        "random_forest": RandomForestClassifier(n_estimators=250, random_state=42),
    }

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )

    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            random_state=42,
        )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    scores: Dict[str, float] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = float(accuracy_score(y_test, pred))
        scores[name] = score

        out_name = f"{name}.pkl"
        joblib.dump(model, ARTIFACT_DIR / out_name)
        print(f"Saved {name} model to {ARTIFACT_DIR / out_name} with accuracy={score:.4f}")

    # If any model unavailable in environment, assign a tiny baseline score so
    # weights still produce valid hybrid metadata.
    scores.setdefault("xgboost", 0.0001)
    scores.setdefault("lightgbm", 0.0001)
    scores.setdefault("random_forest", 0.0001)

    weights = normalize_weights(scores)

    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "scores": scores,
        "weights": weights,
    }
    (ARTIFACT_DIR / "hybrid_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved hybrid metadata to {ARTIFACT_DIR / 'hybrid_meta.json'}")


if __name__ == "__main__":
    main()
