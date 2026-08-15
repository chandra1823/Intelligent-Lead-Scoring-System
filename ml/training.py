"""
Training service.

One entry point trains the shared base model and every tenant model. What
changes between them is the data source and the cold-start tier; the pipeline,
the leakage gate, and the evaluation are identical.

Promotion is champion/challenger: a freshly trained model only becomes active
if it beats the incumbent on the same holdout by a configured margin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.core.config import settings
from ml.canonical import FEATURE_COLUMNS, TARGET_COLUMN
from ml.features import (
    build_preprocessor,
    category_options,
    clean_frame,
    compute_baseline_row,
    feature_statistics,
)
from ml.leakage import LeakageReport, safe_features
from ml.registry import (
    TIER_CONTINUOUS,
    TIER_GENERIC,
    TIER_RECALIBRATED,
    TIER_TENANT,
    tier_for_label_count,
)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None

RANDOM_STATE = 42
# Below this a holdout split is meaningless, so we recalibrate instead of train.
MIN_ROWS_FOR_TRAINING = 200
MIN_POSITIVES = 25


class TrainingError(RuntimeError):
    """Training could not proceed. The message is shown to the user."""


@dataclass
class TrainingResult:
    version: str
    tier: str
    artifact_dir: Path
    metrics: dict[str, dict[str, float]]
    weights: dict[str, float]
    leakage: dict[str, Any]
    training_rows: int
    promoted: bool = False
    promotion_reason: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tier": self.tier,
            "artifact_dir": str(self.artifact_dir),
            "metrics": self.metrics,
            "weights": self.weights,
            "leakage": self.leakage,
            "training_rows": self.training_rows,
            "promoted": self.promoted,
            "promotion_reason": self.promotion_reason,
            "notes": self.notes,
        }


def build_estimators() -> dict[str, object]:
    """Base estimators, constrained to keep artifacts small and reduce overfit."""
    estimators: dict[str, object] = {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    }

    if XGBClassifier is not None:
        estimators["xgboost"] = XGBClassifier(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    if LGBMClassifier is not None:
        estimators["lightgbm"] = LGBMClassifier(
            n_estimators=350,
            num_leaves=31,
            learning_rate=0.05,
            min_child_samples=25,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )

    return estimators


def evaluate(y_true, probabilities) -> dict[str, float]:
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "brier": float(brier_score_loss(y_true, probabilities)),
    }


def auc_based_weights(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    """
    Weight each model by how far its ROC-AUC beats random guessing.

    A model at AUC 0.50 carries no information and receives zero weight, unlike
    accuracy-normalised weights which would still hand it roughly a third.
    """
    lift = {
        name: max(scores["roc_auc"] - 0.5, 0.0)
        for name, scores in metrics.items()
        if name != "hybrid_ensemble"
    }
    total = sum(lift.values())

    if total <= 0:
        equal = 1.0 / len(lift) if lift else 1.0
        return {name: equal for name in lift}

    return {name: value / total for name, value in lift.items()}


def _new_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _validate(frame: pd.DataFrame, target: pd.Series) -> None:
    if len(frame) < MIN_ROWS_FOR_TRAINING:
        raise TrainingError(
            f"Need at least {MIN_ROWS_FOR_TRAINING} labelled leads to train; got {len(frame)}. "
            "The base model keeps scoring until then."
        )
    positives = int(target.sum())
    if positives < MIN_POSITIVES or (len(target) - positives) < MIN_POSITIVES:
        raise TrainingError(
            f"Need at least {MIN_POSITIVES} examples of each outcome; "
            f"got {positives} converted and {len(target) - positives} not converted."
        )


def train_model(
    raw: pd.DataFrame,
    artifact_dir: Path,
    tier: str = TIER_GENERIC,
    leakage_overrides: list[str] | None = None,
    run_leakage_check: bool = True,
) -> TrainingResult:
    """
    Train a hybrid ensemble on a canonical-named frame.

    `raw` must contain the target column. Leakage detection runs against every
    candidate feature before anything is fitted.
    """
    if TARGET_COLUMN not in raw.columns:
        raise TrainingError(f"Training data must include a '{TARGET_COLUMN}' column.")

    frame = raw.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")

    # Accept yes/no and true/false labels, not just 1/0.
    if target.isna().any():
        mapped = (
            frame[TARGET_COLUMN].astype(str).str.strip().str.lower()
            .map({"yes": 1, "true": 1, "won": 1, "converted": 1, "1": 1,
                  "no": 0, "false": 0, "lost": 0, "0": 0})
        )
        target = target.fillna(mapped)

    keep = target.notna()
    frame, target = frame[keep].reset_index(drop=True), target[keep].astype(int).reset_index(drop=True)

    _validate(frame, target)

    notes: list[str] = []
    features = list(FEATURE_COLUMNS)
    leakage: LeakageReport | None = None

    if run_leakage_check:
        candidate_frame = clean_frame(frame)
        features, leakage = safe_features(candidate_frame, target, features, leakage_overrides)
        if leakage.quarantined:
            notes.append(
                "Quarantined as likely target leakage: " + ", ".join(leakage.quarantined)
            )
        if not features:
            raise TrainingError(
                "Every candidate feature was flagged as target leakage. "
                "Review the report and approve any feature you know is legitimate."
            )

    X = clean_frame(frame)
    y = target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    estimators = build_estimators()
    missing = [n for n in ("xgboost", "lightgbm") if n not in estimators]
    if missing:
        notes.append(f"Not installed, skipped: {', '.join(missing)}")

    metrics: dict[str, dict[str, float]] = {}
    holdout_probabilities: dict[str, np.ndarray] = {}

    for name, estimator in estimators.items():
        # Isotonic calibration on internal CV folds, so predicted probabilities
        # can be read as real conversion likelihoods rather than raw scores.
        folds = 5 if len(X_train) >= 2500 else 3
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("classify", CalibratedClassifierCV(estimator, method="isotonic", cv=folds)),
            ]
        )

        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        metrics[name] = evaluate(y_test, probabilities)
        holdout_probabilities[name] = probabilities

        joblib.dump(pipeline, artifact_dir / f"{name}.pkl", compress=3)

    weights = auc_based_weights(metrics)

    ensemble = np.zeros(len(y_test), dtype=float)
    for name, probabilities in holdout_probabilities.items():
        ensemble += probabilities * weights.get(name, 0.0)
    metrics["hybrid_ensemble"] = evaluate(y_test, ensemble)

    version = _new_version()
    baseline_rate = float(y.mean())

    meta = {
        "version": version,
        "tier": tier,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_models": list(weights),
        "weights": weights,
        "metrics": metrics,
        "features_used": features,
        "baseline_row": compute_baseline_row(frame),
        "category_options": category_options(frame),
        "feature_statistics": feature_statistics(frame),
        "leakage_report": leakage.to_dict() if leakage else {"is_clean": True, "findings": []},
        "dataset": {
            "rows": int(len(frame)),
            "conversion_rate": baseline_rate,
            "majority_class_accuracy": float(max(baseline_rate, 1 - baseline_rate)),
        },
        "notes": notes,
    }

    (artifact_dir / "hybrid_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return TrainingResult(
        version=version,
        tier=tier,
        artifact_dir=artifact_dir,
        metrics=metrics,
        weights=weights,
        leakage=meta["leakage_report"],
        training_rows=int(len(frame)),
        notes=notes,
    )


def should_promote(
    challenger: dict[str, dict[str, float]],
    champion: dict[str, dict[str, float]] | None,
    margin: float | None = None,
) -> tuple[bool, str]:
    """
    Champion/challenger decision on holdout ROC-AUC.

    A challenger that merely ties does not win — churning the active model has
    a cost, and ties are usually noise.
    """
    margin = settings.promotion_margin if margin is None else margin
    challenger_auc = challenger.get("hybrid_ensemble", {}).get("roc_auc")

    if challenger_auc is None:
        return False, "challenger produced no ensemble metric"

    if not champion:
        return True, f"first model for this tenant (AUC {challenger_auc:.4f})"

    champion_auc = champion.get("hybrid_ensemble", {}).get("roc_auc")
    if champion_auc is None:
        return True, f"incumbent has no recorded AUC (challenger {challenger_auc:.4f})"

    if challenger_auc >= champion_auc + margin:
        return True, (
            f"challenger AUC {challenger_auc:.4f} beats champion {champion_auc:.4f} "
            f"by at least {margin:.3f}"
        )

    return False, (
        f"challenger AUC {challenger_auc:.4f} did not beat champion {champion_auc:.4f} "
        f"by the required {margin:.3f}; keeping the incumbent"
    )


def recalibrate_base(
    labelled: pd.DataFrame,
    artifact_dir: Path,
    base_dir: Path | None = None,
) -> TrainingResult:
    """
    Tier 1: keep the base model's ranking, fit its probabilities to this
    tenant's conversion rate.

    Used when there are labels but not enough to train from scratch. Isotonic
    regression on the base model's own outputs is cheap and needs far less data
    than a full fit.
    """
    from sklearn.isotonic import IsotonicRegression

    from ml.registry import registry

    base = registry.load(base_dir or settings.base_artifact_dir)
    if not base.models:
        raise TrainingError("No base model available to recalibrate. Train the base model first.")

    frame = labelled.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)

    if len(frame) < settings.tier1_min_labels:
        raise TrainingError(f"Need at least {settings.tier1_min_labels} labelled lead(s) to recalibrate.")
    if target.nunique() < 2:
        raise TrainingError("Recalibration needs both converted and non-converted examples.")

    raw_scores = np.asarray(base.predict_frame(clean_frame(frame)))

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_scores, target.to_numpy())

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, artifact_dir / "calibrator.pkl", compress=3)

    calibrated = calibrator.predict(raw_scores)
    metrics = {
        "base_model": evaluate(target, raw_scores),
        "hybrid_ensemble": evaluate(target, calibrated),
    }

    version = _new_version()
    meta = {
        "version": version,
        "tier": TIER_RECALIBRATED,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "recalibrates": str(base.artifact_dir),
        "metrics": metrics,
        "dataset": {"rows": int(len(frame)), "conversion_rate": float(target.mean())},
        "notes": [
            "Tier 1: base model ranking retained, probabilities fitted to this tenant. "
            "Metrics are in-sample and will read optimistically."
        ],
    }
    (artifact_dir / "recalibration_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return TrainingResult(
        version=version,
        tier=TIER_RECALIBRATED,
        artifact_dir=artifact_dir,
        metrics=metrics,
        weights={},
        leakage={"is_clean": True, "findings": []},
        training_rows=int(len(frame)),
        notes=meta["notes"],
    )


def plan_for(labelled_count: int) -> dict[str, Any]:
    """What would happen if training ran right now, without running it."""
    tier = tier_for_label_count(labelled_count)
    if tier == TIER_GENERIC:
        action = "keep_base_model"
        detail = f"{labelled_count} labelled lead(s); the shared base model keeps scoring."
    elif tier == TIER_RECALIBRATED:
        action = "recalibrate"
        detail = f"{labelled_count} labelled lead(s); base model recalibrated to this pipeline."
    else:
        action = "train_tenant_model"
        detail = f"{labelled_count} labelled lead(s); a dedicated model will be trained."

    return {
        "tier": tier,
        "action": action,
        "detail": detail,
        "labelled_count": labelled_count,
        "next_tier_at": {
            TIER_GENERIC: settings.tier1_min_labels,
            TIER_RECALIBRATED: settings.tier2_min_labels,
            TIER_TENANT: settings.tier3_min_labels,
            TIER_CONTINUOUS: None,
        }[tier],
    }
