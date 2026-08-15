"""
Inference service for the lead scoring hybrid ensemble.

Loads whichever model pipelines exist in an artifact directory, blends their
probabilities using the weights recorded at training time, and degrades to a
heuristic scorer when no artifacts are available.

Operates entirely on canonical field names (see ml/canonical.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover - handled by runtime dependency setup
    joblib = None

from ml.canonical import (
    BY_NAME,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    UNKNOWN_TOKEN,
)

MODEL_NAMES = ("random_forest", "xgboost", "lightgbm")

BAND_THRESHOLDS = ((0.75, "hot"), (0.50, "warm"), (0.25, "cool"))


def band_for(probability: float) -> str:
    for floor, label in BAND_THRESHOLDS:
        if probability >= floor:
            return label
    return "cold"


def _label(field_name: str) -> str:
    spec = BY_NAME.get(field_name)
    if spec is None:
        return field_name
    return spec.name.replace("_", " ").capitalize()


@dataclass
class PredictionResult:
    prediction: int
    probability: float
    model_name: str
    centralized_output: dict[str, Any]
    contributions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def band(self) -> str:
        return band_for(self.probability)


class LeadScoringModel:
    """
    Hybrid inference over RandomForest + XGBoost + LightGBM pipelines.

    Each artifact is a self-contained pipeline that preprocesses a raw
    canonical record, so this class never needs the encoding details.
    """

    def __init__(self, artifact_dir: str | Path) -> None:
        path = Path(artifact_dir)
        # Tolerate being handed a file path (the pre-2.0 config style).
        self.artifact_dir = path.parent if path.suffix == ".pkl" else path

        self.models: dict[str, object] = {}
        self.weights: dict[str, float] = {}
        self.metrics: dict[str, dict[str, float]] = {}
        self.baseline_row: dict[str, Any] = {}
        self.category_options: dict[str, list[str]] = {}
        self.dataset_info: dict[str, Any] = {}
        self.feature_statistics: dict[str, Any] = {}
        self.leakage_report: dict[str, Any] = {}
        self.calibrator: object | None = None
        self.version: str = "unversioned"
        self.tier: str = "generic"
        self.model_name: str = "heuristic_fallback"

        self._load()

    # ------------------------------------------------------------------ load

    def _load(self) -> None:
        self.calibrator = None
        self._load_meta()

        if joblib is None:
            return

        self._load_pipelines(self.artifact_dir)

        # Tier 1: this directory holds only a calibrator fitted on top of
        # another model's output. Borrow that model's pipelines and metadata,
        # then apply the calibrator to whatever it produces.
        if not self.models:
            self._load_recalibration()

        if not self.models:
            self.model_name = "heuristic_fallback"
            return

        # Drop weights for models that failed to load, then renormalise so the
        # remainder still sums to 1 instead of silently scaling scores down.
        usable = {name: self.weights.get(name, 0.0) for name in self.models}
        total = sum(usable.values())
        if total > 0:
            self.weights = {name: value / total for name, value in usable.items()}
        else:
            equal = 1.0 / len(self.models)
            self.weights = {name: equal for name in self.models}

        if len(self.models) == len(MODEL_NAMES):
            self.model_name = "hybrid_rf_xgb_lgbm"
        else:
            self.model_name = "hybrid_partial_" + "_".join(sorted(self.models))

        if self.calibrator is not None:
            self.model_name += "_recalibrated"

    def _load_pipelines(self, directory: Path) -> None:
        for name in MODEL_NAMES:
            path = directory / f"{name}.pkl"
            if not path.exists():
                continue
            try:
                self.models[name] = joblib.load(path)
            except Exception:  # pragma: no cover - corrupt or incompatible artifact
                continue

    def _load_recalibration(self) -> None:
        """Load a Tier 1 artifact: someone else's model plus our calibrator."""
        meta_path = self.artifact_dir / "recalibration_meta.json"
        calibrator_path = self.artifact_dir / "calibrator.pkl"
        if not (meta_path.exists() and calibrator_path.exists()):
            return

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            base_dir = Path(meta["recalibrates"])
            self.calibrator = joblib.load(calibrator_path)
        except (ValueError, OSError, KeyError):
            return

        self._load_pipelines(base_dir)
        if not self.models:
            self.calibrator = None
            return

        # Ranking comes from the borrowed model, so take its weights and
        # baselines; the tier and metrics stay ours.
        base_meta_path = base_dir / "hybrid_meta.json"
        if base_meta_path.exists():
            try:
                base_meta = json.loads(base_meta_path.read_text(encoding="utf-8"))
                self.weights = {k: float(v) for k, v in base_meta.get("weights", {}).items()}
                self.baseline_row = base_meta.get("baseline_row", {})
                self.category_options = base_meta.get("category_options", {})
                self.feature_statistics = base_meta.get("feature_statistics", {})
            except (ValueError, OSError):
                pass

        self.version = meta.get("version", self.version)
        self.tier = meta.get("tier", "recalibrated")
        self.dataset_info = meta.get("dataset", self.dataset_info)

    def _calibrate(self, probability: float) -> float:
        if self.calibrator is None:
            return probability
        try:
            return float(self.calibrator.predict([probability])[0])
        except Exception:  # pragma: no cover
            return probability

    def _load_meta(self) -> None:
        meta_path = self.artifact_dir / "hybrid_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self.weights = {k: float(v) for k, v in meta.get("weights", {}).items()}
                self.metrics = meta.get("metrics", {})
                self.baseline_row = meta.get("baseline_row", {})
                self.category_options = meta.get("category_options", {})
                self.dataset_info = meta.get("dataset", {})
                self.feature_statistics = meta.get("feature_statistics", {})
                self.leakage_report = meta.get("leakage_report", {})
                self.version = meta.get("version", "unversioned")
                self.tier = meta.get("tier", "generic")
            except (ValueError, OSError):
                pass

    # --------------------------------------------------------------- helpers

    def _build_row(self, features: dict[str, Any]) -> pd.DataFrame:
        """Fill any field the caller omitted with the training-set baseline."""
        row: dict[str, Any] = {}

        for column in FEATURE_COLUMNS:
            value = features.get(column)
            if value is None or (isinstance(value, str) and not value.strip()):
                value = self.baseline_row.get(column)
            if value is None:
                value = 0.0 if column in NUMERIC_FEATURES else UNKNOWN_TOKEN
            row[column] = value

        frame = pd.DataFrame([row])
        for column in NUMERIC_FEATURES:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

        return frame[FEATURE_COLUMNS]

    def _component_probabilities(self, frame: pd.DataFrame) -> dict[str, float]:
        scores: dict[str, float] = {}
        for name, pipeline in self.models.items():
            try:
                scores[name] = float(pipeline.predict_proba(frame)[0][1])
            except Exception:  # pragma: no cover - runtime mismatch guard
                continue
        return scores

    def _blend(self, component_scores: dict[str, float]) -> float | None:
        weighted_sum = 0.0
        weight_total = 0.0

        for name, score in component_scores.items():
            weight = self.weights.get(name, 0.0)
            weighted_sum += weight * score
            weight_total += weight

        if weight_total <= 0:
            return None
        return weighted_sum / weight_total

    @staticmethod
    def _heuristic_score(features: dict[str, Any]) -> float:
        """Deterministic fallback so the system still runs without artifacts."""
        time_spent = float(features.get("time_on_site_seconds") or 0.0)
        page_views = float(features.get("page_views_per_visit") or 0.0)
        visits = float(features.get("total_visits") or 0.0)

        score = (
            min(time_spent / 300.0, 1.0) * 0.5
            + min(page_views / 5.0, 1.0) * 0.25
            + min(visits / 10.0, 1.0) * 0.25
        )
        return float(max(0.0, min(score, 1.0)))

    # --------------------------------------------------------------- predict

    def predict(self, features: dict[str, Any]) -> PredictionResult:
        frame = self._build_row(features)
        component_scores = self._component_probabilities(frame)
        blended = self._blend(component_scores)

        if blended is None:
            probability = self._heuristic_score(features)
            inference_mode = "heuristic_fallback"
        else:
            probability = self._calibrate(blended)
            inference_mode = "recalibrated_ensemble" if self.calibrator else "hybrid_ensemble"

        prediction = int(probability >= 0.5)
        centralized_output = {
            "prediction": prediction,
            "probability": round(probability, 4),
            "label": "likely_to_convert" if prediction else "unlikely_to_convert",
            "band": band_for(probability),
            "ensemble_model": self.model_name,
            "model_version": self.version,
            "model_tier": self.tier,
            "inference_mode": inference_mode,
            # Only real per-model scores appear here. Absent models are absent,
            # rather than being backfilled with the blended value.
            "components": {name: round(score, 4) for name, score in component_scores.items()},
            "weights": {name: round(weight, 4) for name, weight in self.weights.items()},
            "available_models": sorted(self.models),
        }

        return PredictionResult(
            prediction=prediction,
            probability=probability,
            model_name=self.model_name,
            centralized_output=centralized_output,
        )

    def predict_frame(self, frame: pd.DataFrame) -> list[float]:
        """Vectorised scoring for batch work — one pipeline pass, not one per row."""
        if not self.models:
            return [
                self._heuristic_score(row)
                for row in frame.to_dict(orient="records")
            ]

        prepared = frame.copy()
        for column in FEATURE_COLUMNS:
            if column not in prepared.columns:
                prepared[column] = None

        for column in NUMERIC_FEATURES:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
            fallback = self.baseline_row.get(column, 0.0)
            prepared[column] = prepared[column].fillna(fallback)

        for column in CATEGORICAL_FEATURES:
            fallback = self.baseline_row.get(column, UNKNOWN_TOKEN)
            prepared[column] = prepared[column].astype("object").fillna(fallback)

        prepared = prepared[FEATURE_COLUMNS]

        weighted_total = None
        weight_sum = 0.0
        for name, pipeline in self.models.items():
            weight = self.weights.get(name, 0.0)
            if weight <= 0:
                continue
            try:
                probabilities = pipeline.predict_proba(prepared)[:, 1]
            except Exception:  # pragma: no cover
                continue
            contribution = probabilities * weight
            weighted_total = contribution if weighted_total is None else weighted_total + contribution
            weight_sum += weight

        if weighted_total is None or weight_sum <= 0:
            return [self._heuristic_score(row) for row in frame.to_dict(orient="records")]

        blended = weighted_total / weight_sum
        if self.calibrator is not None:
            try:
                blended = self.calibrator.predict(blended)
            except Exception:  # pragma: no cover
                pass
        return [float(value) for value in blended]

    # --------------------------------------------------------------- explain

    def explain(self, features: dict[str, Any], top_n: int = 6) -> PredictionResult:
        """
        Attribute the score to individual fields by ablation.

        For each field, the lead is re-scored with that one value reset to the
        training baseline. The shift in probability is that field's measured
        contribution for this specific lead — a real model-derived number, not
        a global importance ranking.
        """
        result = self.predict(features)

        if not self.models or not self.baseline_row:
            result.contributions = []
            return result

        resolved = self._build_row(features).iloc[0].to_dict()
        contributions: list[dict[str, Any]] = []

        for column in FEATURE_COLUMNS:
            baseline_value = self.baseline_row.get(column)
            if baseline_value is None:
                continue

            actual_value = resolved.get(column)
            if column in NUMERIC_FEATURES:
                if abs(float(actual_value) - float(baseline_value)) < 1e-9:
                    continue
            elif str(actual_value) == str(baseline_value):
                continue

            counterfactual = dict(resolved)
            counterfactual[column] = baseline_value

            frame = self._build_row(counterfactual)
            blended = self._blend(self._component_probabilities(frame))
            if blended is None:
                continue

            delta = result.probability - blended
            if abs(delta) < 0.001:
                continue

            contributions.append(
                {
                    "feature": _label(column),
                    "field": column,
                    "value": actual_value,
                    "baseline": baseline_value,
                    "impact": round(delta, 4),
                    "direction": "increases" if delta > 0 else "decreases",
                }
            )

        contributions.sort(key=lambda item: abs(item["impact"]), reverse=True)
        result.contributions = contributions[:top_n]
        return result

    def summarize(self, result: PredictionResult) -> str:
        """One-paragraph narrative built from the measured contributions."""
        percent = result.probability * 100

        if result.centralized_output["inference_mode"] == "heuristic_fallback":
            return (
                f"No trained artifacts are loaded, so this {percent:.1f}% score comes from the "
                "engagement heuristic. Run scripts/train_model.py for model-backed scoring."
            )

        if result.prediction == 1:
            headline = f"This lead scores {percent:.1f}% and is prioritised for follow-up."
        else:
            headline = f"This lead scores {percent:.1f}%, below the 50% follow-up threshold."

        if not result.contributions:
            return f"{headline} Its attributes sit close to the typical lead, so no single field stands out."

        drivers = []
        for item in result.contributions[:3]:
            points = abs(item["impact"]) * 100
            verb = "raising" if item["impact"] > 0 else "lowering"
            drivers.append(f"{item['feature']} ({item['value']}) {verb} it by {points:.1f} points")

        return f"{headline} Biggest factors versus a typical lead: " + "; ".join(drivers) + "."

    # ------------------------------------------------------------ diagnostics

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.model_name,
            "model_version": self.version,
            "model_tier": self.tier,
            "available_models": sorted(self.models),
            "hybrid_weights": {name: round(weight, 4) for name, weight in self.weights.items()},
            "artifacts_loaded": bool(self.models),
            "artifact_dir": str(self.artifact_dir),
        }
