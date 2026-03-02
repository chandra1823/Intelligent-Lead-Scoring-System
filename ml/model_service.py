from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled by runtime dependency setup
    np = None

try:
    import joblib
except ImportError:  # pragma: no cover - handled by runtime dependency setup
    joblib = None


@dataclass
class PredictionResult:
    prediction: int
    probability: float
    model_name: str


class LeadScoringModel:
    """
    Service layer for loading and predicting with a trained model.

    If no model artifact exists, fallback rule-based scoring is used so
    the API remains demo-ready for a college project.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self.model: Optional[object] = None
        self.model_name: str = "rule_based_fallback"
        self._load_model()

    def _load_model(self) -> None:
        if self.model_path.exists() and joblib is not None:
            self.model = joblib.load(self.model_path)
            self.model_name = self.model.__class__.__name__

    def predict(self, total_time_spent_on_website: float, page_views_per_visit: float, total_visits: float) -> PredictionResult:
        if np is not None:
            features = np.array([[total_time_spent_on_website, page_views_per_visit, total_visits]], dtype=float)
        else:
            features = [[float(total_time_spent_on_website), float(page_views_per_visit), float(total_visits)]]

        if self.model is not None:
            probability = float(self.model.predict_proba(features)[0, 1])
            prediction = int(probability >= 0.5)
            return PredictionResult(prediction=prediction, probability=probability, model_name=self.model_name)

        # Fallback heuristic for demo usage only.
        normalized = (
            min(total_time_spent_on_website / 300.0, 1.0) * 0.5
            + min(page_views_per_visit / 5.0, 1.0) * 0.25
            + min(total_visits / 10.0, 1.0) * 0.25
        )
        probability = float(max(0.0, min(normalized, 1.0)))
        prediction = int(probability >= 0.5)
        return PredictionResult(prediction=prediction, probability=probability, model_name=self.model_name)
