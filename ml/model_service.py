from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled by runtime dependency setup
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    import joblib
except ImportError:  # pragma: no cover - handled by runtime dependency setup
    joblib = None


@dataclass
class PredictionResult:
    prediction: int
    probability: float
    model_name: str
    centralized_output: Dict[str, float | int | str | Dict[str, float]]


class LeadScoringModel:
    """
    Hybrid inference service using RandomForest + XGBoost + LightGBM.

    It attempts to load all three model artifacts and combines them into a
    weighted ensemble. If artifacts are missing, it falls back to a deterministic
    heuristic scorer so the app remains demo-ready.
    """

    def __init__(self, model_path: str) -> None:
        model_file = Path(model_path)
        self.artifact_dir = model_file.parent if model_file.parent != Path("") else Path("artifacts")

        self.rf_model: Optional[object] = None
        self.xgb_model: Optional[object] = None
        self.lgbm_model: Optional[object] = None

        self.weights: Dict[str, float] = {"random_forest": 0.34, "xgboost": 0.33, "lightgbm": 0.33}
        self.model_name: str = "hybrid_fallback"
        self._load_models_and_meta()

    def _load_models_and_meta(self) -> None:
        if joblib is None:
            return

        rf_path = self.artifact_dir / "random_forest.pkl"
        xgb_path = self.artifact_dir / "xgboost.pkl"
        lgbm_path = self.artifact_dir / "lightgbm.pkl"
        meta_path = self.artifact_dir / "hybrid_meta.json"

        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                maybe_weights = meta.get("weights", {})
                if all(k in maybe_weights for k in ["random_forest", "xgboost", "lightgbm"]):
                    total = sum(float(v) for v in maybe_weights.values())
                    if total > 0:
                        self.weights = {k: float(v) / total for k, v in maybe_weights.items()}
            except Exception:
                pass

        try:
            if rf_path.exists():
                self.rf_model = joblib.load(rf_path)
            if xgb_path.exists():
                self.xgb_model = joblib.load(xgb_path)
            if lgbm_path.exists():
                self.lgbm_model = joblib.load(lgbm_path)
        except Exception:
            self.rf_model = None
            self.xgb_model = None
            self.lgbm_model = None
            self.model_name = "hybrid_fallback"
            return

        if self.rf_model and self.xgb_model and self.lgbm_model:
            self.model_name = "hybrid_rf_xgb_lgbm"
        else:
            self.model_name = "hybrid_partial_fallback"

    def _build_feature_vector(self, total_time_spent_on_website: float, page_views_per_visit: float, total_visits: float):
        row = {
            "Total Time Spent on Website": float(total_time_spent_on_website),
            "Page Views Per Visit": float(page_views_per_visit),
            "TotalVisits": float(total_visits),
        }
        if pd is not None:
            return pd.DataFrame([row])
        if np is not None:
            return np.array([[row["Total Time Spent on Website"], row["Page Views Per Visit"], row["TotalVisits"]]], dtype=float)
        return [[row["Total Time Spent on Website"], row["Page Views Per Visit"], row["TotalVisits"]]]

    @staticmethod
    def _safe_proba(model: object, features) -> Optional[float]:
        try:
            proba = model.predict_proba(features)
            return float(proba[0][1])
        except Exception:
            return None

    @staticmethod
    def _heuristic_score(total_time_spent_on_website: float, page_views_per_visit: float, total_visits: float) -> float:
        normalized = (
            min(total_time_spent_on_website / 300.0, 1.0) * 0.5
            + min(page_views_per_visit / 5.0, 1.0) * 0.25
            + min(total_visits / 10.0, 1.0) * 0.25
        )
        return float(max(0.0, min(normalized, 1.0)))

    def predict(self, total_time_spent_on_website: float, page_views_per_visit: float, total_visits: float) -> PredictionResult:
        features = self._build_feature_vector(total_time_spent_on_website, page_views_per_visit, total_visits)

        rf_proba = self._safe_proba(self.rf_model, features) if self.rf_model is not None else None
        xgb_proba = self._safe_proba(self.xgb_model, features) if self.xgb_model is not None else None
        lgbm_proba = self._safe_proba(self.lgbm_model, features) if self.lgbm_model is not None else None

        component_scores: Dict[str, float] = {}
        weighted_sum = 0.0
        weight_sum = 0.0

        if rf_proba is not None:
            component_scores["random_forest"] = rf_proba
            weighted_sum += self.weights["random_forest"] * rf_proba
            weight_sum += self.weights["random_forest"]

        if xgb_proba is not None:
            component_scores["xgboost"] = xgb_proba
            weighted_sum += self.weights["xgboost"] * xgb_proba
            weight_sum += self.weights["xgboost"]

        if lgbm_proba is not None:
            component_scores["lightgbm"] = lgbm_proba
            weighted_sum += self.weights["lightgbm"] * lgbm_proba
            weight_sum += self.weights["lightgbm"]

        if weight_sum > 0:
            final_proba = weighted_sum / weight_sum
            inference_mode = "hybrid_ensemble"
        else:
            final_proba = self._heuristic_score(total_time_spent_on_website, page_views_per_visit, total_visits)
            inference_mode = "heuristic_fallback"

        # Keep component output centralized and consistent for frontend rendering.
        component_scores.setdefault("random_forest", float(final_proba))
        component_scores.setdefault("xgboost", float(final_proba))
        component_scores.setdefault("lightgbm", float(final_proba))

        prediction = int(final_proba >= 0.5)
        centralized_output = {
            "prediction": prediction,
            "probability": float(final_proba),
            "label": "likely_to_convert" if prediction == 1 else "unlikely_to_convert",
            "ensemble_model": self.model_name,
            "inference_mode": inference_mode,
            "components": component_scores,
            "available_models": [
                name for name, model in {
                    "random_forest": self.rf_model,
                    "xgboost": self.xgb_model,
                    "lightgbm": self.lgbm_model,
                }.items() if model is not None
            ],
        }

        return PredictionResult(
            prediction=prediction,
            probability=float(final_proba),
            model_name=self.model_name,
            centralized_output=centralized_output,
        )
