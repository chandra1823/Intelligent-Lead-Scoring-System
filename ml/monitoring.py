"""
Drift and calibration monitoring.

Two separate failures, two separate checks:

  * drift        - the leads coming in stopped looking like the training data
  * miscalibration - the model still ranks well, but its probabilities no
                     longer mean what they say

Drift is measurable immediately. Calibration needs outcomes, so it lags.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from app.core.config import settings

EPSILON = 1e-6


@dataclass
class DriftSignal:
    feature: str
    psi: float
    severity: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _severity(psi: float) -> str:
    if psi >= settings.drift_alert_psi:
        return "alert"
    if psi >= settings.drift_warn_psi:
        return "warn"
    return "ok"


def population_stability_index(
    reference: dict[str, Any], live_values: pd.Series
) -> float | None:
    """
    PSI between a stored reference distribution and live traffic.

    Rule of thumb: below 0.1 stable, 0.1-0.25 worth watching, above 0.25 the
    population has genuinely moved.
    """
    values = live_values.dropna()
    if values.empty:
        return None

    kind = reference.get("kind")

    if kind == "numeric":
        bounds = np.asarray(reference.get("bin_edges") or [], dtype=float)
        stored = reference.get("bin_proportions")
        if len(bounds) < 3 or not stored:
            return None

        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return None

        # Expected mass comes from the reference histogram, so identical data
        # scores exactly zero however lumpy the distribution is.
        expected = np.asarray(stored, dtype=float)
        observed, _ = np.histogram(numeric, bins=bounds)
        observed = observed / max(observed.sum(), 1)
        if len(expected) != len(observed):
            return None

    elif kind in ("categorical", "numeric_discrete"):
        proportions = reference.get("proportions") or {}
        if not proportions:
            return None
        categories = list(proportions)
        expected = np.array([proportions[c] for c in categories], dtype=float)

        if kind == "numeric_discrete":
            live = pd.to_numeric(values, errors="coerce").dropna()
            keys = live.astype(float).astype(str)
        else:
            keys = values.astype(str)

        counts = keys.value_counts(normalize=True)
        observed = np.array([float(counts.get(c, 0.0)) for c in categories], dtype=float)

        # Everything outside the reference vocabulary is one "other" bucket.
        expected = np.append(expected, max(1.0 - expected.sum(), 0.0))
        observed = np.append(observed, max(1.0 - observed.sum(), 0.0))
    else:
        return None

    expected = np.clip(expected, EPSILON, None)
    observed = np.clip(observed, EPSILON, None)
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def detect_drift(
    reference_statistics: dict[str, Any],
    live_frame: pd.DataFrame,
    min_rows: int = 50,
) -> dict[str, Any]:
    """Compare live traffic against the distributions captured at training time."""
    if len(live_frame) < min_rows:
        return {
            "status": "insufficient_data",
            "detail": f"Need at least {min_rows} recent leads; have {len(live_frame)}.",
            "signals": [],
            "rows_examined": len(live_frame),
        }

    signals: list[DriftSignal] = []

    for feature, reference in reference_statistics.items():
        if feature not in live_frame.columns:
            continue

        psi = population_stability_index(reference, live_frame[feature])
        if psi is None:
            continue

        severity = _severity(psi)
        if severity == "ok":
            continue

        signals.append(
            DriftSignal(
                feature=feature,
                psi=round(psi, 4),
                severity=severity,
                detail=(
                    f"PSI {psi:.3f} — the distribution of '{feature}' has moved "
                    f"{'materially' if severity == 'alert' else 'noticeably'} since training"
                ),
            )
        )

    signals.sort(key=lambda item: item.psi, reverse=True)
    status = "alert" if any(s.severity == "alert" for s in signals) else (
        "warn" if signals else "ok"
    )

    return {
        "status": status,
        "detail": {
            "ok": "Live traffic matches the training distribution.",
            "warn": "Some features have shifted; worth watching.",
            "alert": "Input distributions have moved materially. Consider retraining.",
        }[status],
        "signals": [signal.to_dict() for signal in signals],
        "rows_examined": len(live_frame),
    }


def calibration_report(
    predictions: Iterable[float],
    outcomes: Iterable[int],
    bins: int = 10,
) -> dict[str, Any]:
    """
    How well predicted probabilities match observed conversion rates.

    Expected Calibration Error is the headline: the average gap between what
    the model promised and what actually happened.
    """
    probabilities = np.asarray(list(predictions), dtype=float)
    actuals = np.asarray(list(outcomes), dtype=int)

    if len(probabilities) == 0 or len(probabilities) != len(actuals):
        return {"status": "insufficient_data", "detail": "No matched predictions and outcomes."}

    if len(np.unique(actuals)) < 2:
        return {
            "status": "insufficient_data",
            "detail": "Outcomes are all one class; calibration needs both.",
            "sample_size": int(len(actuals)),
        }

    edges = np.linspace(0.0, 1.0, bins + 1)
    buckets: list[dict[str, float]] = []
    ece = 0.0

    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        mask = (probabilities >= low) & (
            probabilities < high if index < bins - 1 else probabilities <= high
        )
        if not mask.any():
            continue

        predicted = float(probabilities[mask].mean())
        observed = float(actuals[mask].mean())
        weight = float(mask.sum()) / len(probabilities)
        ece += weight * abs(predicted - observed)

        buckets.append(
            {
                "bin_low": round(float(low), 2),
                "bin_high": round(float(high), 2),
                "count": int(mask.sum()),
                "mean_predicted": round(predicted, 4),
                "observed_rate": round(observed, 4),
                "gap": round(observed - predicted, 4),
            }
        )

    brier = float(brier_score_loss(actuals, probabilities))
    try:
        auc = float(roc_auc_score(actuals, probabilities))
    except ValueError:  # pragma: no cover
        auc = float("nan")

    if ece <= 0.05:
        status, detail = "ok", "Predicted probabilities track observed outcomes."
    elif ece <= 0.12:
        status, detail = "warn", "Probabilities are drifting from observed rates."
    else:
        status, detail = "alert", "Probabilities no longer mean what they say. Recalibrate."

    return {
        "status": status,
        "detail": detail,
        "sample_size": int(len(actuals)),
        "expected_calibration_error": round(ece, 4),
        "brier": round(brier, 4),
        "roc_auc": round(auc, 4),
        "observed_rate": round(float(actuals.mean()), 4),
        "mean_predicted": round(float(probabilities.mean()), 4),
        "bins": buckets,
    }


def health_rollup(drift: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    """Single status for a dashboard tile, worst-of the two checks."""
    order = {"ok": 0, "insufficient_data": 1, "warn": 2, "alert": 3}
    worst = max(
        [drift.get("status", "ok"), calibration.get("status", "ok")],
        key=lambda status: order.get(status, 0),
    )

    return {
        "status": worst,
        "drift": drift.get("status"),
        "calibration": calibration.get("status"),
        "action_required": worst == "alert",
        "recommendation": {
            "ok": "No action needed.",
            "insufficient_data": "Collect more scored leads and outcomes.",
            "warn": "Monitor; schedule a retrain if the trend continues.",
            "alert": "Retrain on recent data.",
        }[worst],
    }
