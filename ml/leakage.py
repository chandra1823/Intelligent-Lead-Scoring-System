"""
Target leakage detection.

Phase 1 found this by hand: the bundled dataset's `Tags` column is filled in by
a sales rep *after* the outcome is known, so `Tags = "Closed by Horizzon"`
converts at 99.4%. Training on it produced ~92% accuracy that meant nothing.

Every CRM has a column like that. This module runs the same check
automatically, before any training run, on any customer's data.

A flagged feature is quarantined, not deleted — the report explains why, and a
human can override. Silent removal would be its own kind of wrong.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from sklearn.metrics import roc_auc_score

# A single feature carrying this much signal alone is almost never legitimate.
SOLO_AUC_LIMIT = 0.93
# A category this pure is an outcome label wearing a feature's clothes.
PURITY_LIMIT = 0.95
PURITY_FLOOR = 0.05
# Ignore tiny categories; 3 leads at 100% is noise, not leakage.
MIN_CATEGORY_SUPPORT = 25

SEVERITY_ORDER = {"critical": 0, "high": 1, "moderate": 2}

# Column names that are outcome-ish in most CRMs. Name evidence alone never
# flags a feature — it only raises the severity of a statistical finding.
SUSPICIOUS_NAME_TOKENS = (
    "tag", "status", "stage", "quality", "grade", "disposition", "outcome",
    "result", "closed", "won", "lost", "rating", "score", "profile",
)


@dataclass
class LeakageFinding:
    feature: str
    severity: str
    reason: str
    detail: str
    statistic: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LeakageReport:
    findings: list[LeakageFinding]
    quarantined: list[str]
    checked: list[str]

    @property
    def is_clean(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_clean": self.is_clean,
            "quarantined": self.quarantined,
            "checked": self.checked,
            "findings": [f.to_dict() for f in self.findings],
        }

    def summary(self) -> str:
        if self.is_clean:
            return f"No leakage detected across {len(self.checked)} feature(s)."
        lines = [
            f"{len(self.findings)} potential leak(s) found; "
            f"{len(self.quarantined)} feature(s) quarantined."
        ]
        for finding in self.findings:
            lines.append(f"  [{finding.severity}] {finding.feature}: {finding.detail}")
        return "\n".join(lines)


def _name_is_suspicious(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in SUSPICIOUS_NAME_TOKENS)


def _escalate(severity: str, feature: str) -> str:
    """A suspicious name turns a moderate statistical signal into a high one."""
    if not _name_is_suspicious(feature):
        return severity
    if severity == "moderate":
        return "high"
    if severity == "high":
        return "critical"
    return severity


def _check_numeric(name: str, values: pd.Series, target: pd.Series) -> LeakageFinding | None:
    usable = pd.to_numeric(values, errors="coerce")
    mask = usable.notna()
    if mask.sum() < MIN_CATEGORY_SUPPORT or target[mask].nunique() < 2:
        return None

    try:
        auc = roc_auc_score(target[mask], usable[mask])
    except ValueError:
        return None

    # A feature that perfectly inverts the target leaks just as badly.
    strength = max(auc, 1.0 - auc)
    if strength < SOLO_AUC_LIMIT:
        return None

    severity = "critical" if strength >= 0.98 else "high"
    return LeakageFinding(
        feature=name,
        severity=_escalate(severity, name),
        reason="solo_auc",
        detail=(
            f"alone predicts the outcome at AUC {strength:.3f}; "
            "a legitimate single feature rarely exceeds 0.90"
        ),
        statistic=round(float(strength), 4),
    )


def _check_categorical(name: str, values: pd.Series, target: pd.Series) -> LeakageFinding | None:
    frame = pd.DataFrame({"value": values.astype("object"), "target": target}).dropna()
    if frame.empty:
        return None

    grouped = frame.groupby("value")["target"].agg(["count", "mean"])
    supported = grouped[grouped["count"] >= MIN_CATEGORY_SUPPORT]
    if supported.empty:
        return None

    extreme = supported[(supported["mean"] >= PURITY_LIMIT) | (supported["mean"] <= PURITY_FLOOR)]
    if extreme.empty:
        return None

    covered = int(extreme["count"].sum())
    coverage = covered / len(frame)
    worst = extreme["mean"].apply(lambda rate: max(rate, 1 - rate)).max()

    # One tiny pure category is a quirk; a third of the rows is a label.
    if coverage < 0.02:
        return None

    severity = "critical" if coverage >= 0.25 else ("high" if coverage >= 0.08 else "moderate")
    examples = ", ".join(
        f"{value!r}={rate:.1%}" for value, rate in extreme["mean"].head(3).items()
    )
    return LeakageFinding(
        feature=name,
        severity=_escalate(severity, name),
        reason="category_purity",
        detail=(
            f"{len(extreme)} category value(s) covering {coverage:.0%} of rows are "
            f"near-perfectly separated ({examples})"
        ),
        statistic=round(float(worst), 4),
    )


def detect_leakage(
    frame: pd.DataFrame,
    target: pd.Series,
    features: list[str] | None = None,
    quarantine_at: str = "high",
) -> LeakageReport:
    """
    Scan candidate features for target leakage.

    Findings at or above `quarantine_at` severity are quarantined; the caller
    decides whether to honour that or override it with a human decision.
    """
    columns = features if features is not None else [c for c in frame.columns if c != target.name]
    target = pd.Series(target).reset_index(drop=True)
    findings: list[LeakageFinding] = []
    checked: list[str] = []

    for column in columns:
        if column not in frame.columns:
            continue
        checked.append(column)
        values = frame[column].reset_index(drop=True)

        if values.dropna().empty or values.nunique(dropna=True) < 2:
            continue

        numeric = pd.to_numeric(values, errors="coerce")
        is_numeric_column = numeric.notna().mean() > 0.9

        finding = (
            _check_numeric(column, values, target)
            if is_numeric_column
            else _check_categorical(column, values, target)
        )
        if finding is not None:
            findings.append(finding)

    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), -f.statistic))
    threshold = SEVERITY_ORDER.get(quarantine_at, 1)
    quarantined = [f.feature for f in findings if SEVERITY_ORDER.get(f.severity, 9) <= threshold]

    return LeakageReport(findings=findings, quarantined=quarantined, checked=checked)


def safe_features(
    frame: pd.DataFrame,
    target: pd.Series,
    features: list[str],
    overrides: list[str] | None = None,
) -> tuple[list[str], LeakageReport]:
    """
    Return the feature list with leaky columns removed.

    `overrides` names features a human has explicitly approved despite a flag.
    """
    report = detect_leakage(frame, target, features)
    approved = set(overrides or [])
    blocked = set(report.quarantined) - approved
    return [f for f in features if f not in blocked], report
