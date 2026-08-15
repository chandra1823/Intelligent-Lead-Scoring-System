"""
Preprocessing for the canonical lead schema.

Models train and score on canonical field names only (see ml/canonical.py).
Source-specific column names never reach this layer — the schema mapper has
already translated them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.canonical import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    PLACEHOLDER_VALUES,
    UNKNOWN_TOKEN,
)

# A numeric feature with at most this many distinct values is treated as
# discrete for monitoring purposes.
DISCRETE_VALUE_LIMIT = 20

# Built-in mapping for the bundled "Lead Scoring.csv". It is an ordinary source
# mapping with no special status — the same shape a connector produces.
X_EDUCATION_MAPPING: dict[str, str] = {
    "Prospect ID": "external_id",
    "Total Time Spent on Website": "time_on_site_seconds",
    "TotalVisits": "total_visits",
    "Page Views Per Visit": "page_views_per_visit",
    "Lead Origin": "origin",
    "Lead Source": "channel",
    "Last Activity": "last_activity",
    "Last Notable Activity": "last_notable_activity",
    "What is your current occupation": "occupation",
    "Specialization": "specialization",
    "How did you hear about X Education": "heard_about_us",
    "What matters most to you in choosing a course": "motivation",
    "City": "city",
    "Country": "country",
    "Do Not Email": "do_not_contact",
    "A free copy of Mastering The Interview": "wants_free_material",
    "Converted": "converted",
}

# Columns a sales rep fills in after contacting the lead. Kept here so the
# bundled dataset stays honest even if the automatic detector is disabled.
# See ml/leakage.py for the check applied to arbitrary customer data.
LEAKY_COLUMNS: list[str] = [
    "Tags",
    "Lead Quality",
    "Lead Profile",
    "Asymmetrique Activity Index",
    "Asymmetrique Profile Index",
    "Asymmetrique Activity Score",
    "Asymmetrique Profile Score",
]


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce a canonical-named frame into the exact feature matrix layout.

    Missing canonical columns are added as NaN so a partially mapped source
    still scores — the imputers fill the gaps from training-set statistics.
    """
    frame = df.copy()

    for column in CATEGORICAL_FEATURES:
        if column not in frame.columns:
            frame[column] = np.nan
            continue
        series = frame[column].astype("object")
        lowered = series.astype(str).str.strip().str.lower()
        frame[column] = series.where(~lowered.isin(PLACEHOLDER_VALUES), np.nan)

    for column in NUMERIC_FEATURES:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame[FEATURE_COLUMNS]


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + scale numerics, mode-impute + one-hot the categoricals."""
    numeric_pipeline = Pipeline(
        steps=[
            # keep_empty_features holds the column count stable when a source
            # supplies none of a field, so the matrix shape does not depend on
            # which optional fields a customer happens to populate.
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value=UNKNOWN_TOKEN)),
            # min_frequency folds rare levels into one bucket, which keeps the
            # one-hot matrix small and stops the trees splitting on noise.
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=25,
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def compute_baseline_row(df: pd.DataFrame) -> dict[str, object]:
    """The 'typical lead' used as the reference point for explanations."""
    frame = clean_frame(df)
    baseline: dict[str, object] = {}

    for column in NUMERIC_FEATURES:
        median = frame[column].median()
        baseline[column] = float(median) if pd.notna(median) else 0.0

    for column in CATEGORICAL_FEATURES:
        modes = frame[column].mode(dropna=True)
        baseline[column] = str(modes.iloc[0]) if not modes.empty else UNKNOWN_TOKEN

    return baseline


def category_options(df: pd.DataFrame, max_options: int = 12) -> dict[str, list[str]]:
    """Most common values per categorical field, for populating UI dropdowns."""
    frame = clean_frame(df)
    options: dict[str, list[str]] = {}

    for column in CATEGORICAL_FEATURES:
        counts = frame[column].value_counts(dropna=True)
        options[column] = [str(value) for value in counts.head(max_options).index]

    return options


def feature_statistics(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    """
    Reference distributions captured at training time.

    Drift monitoring compares live traffic against these.
    """
    frame = clean_frame(df)
    stats: dict[str, dict[str, object]] = {}

    for column in NUMERIC_FEATURES:
        values = frame[column].dropna()
        if values.empty:
            continue

        # Counts like "visits" take a handful of distinct values, and quantiles
        # describe them badly — several quantiles land on the same number, and
        # any binning built from them reports drift against the very data it
        # was derived from. Store proportions per value instead and compare
        # them exactly.
        if values.nunique() <= DISCRETE_VALUE_LIMIT:
            proportions = values.value_counts(normalize=True)
            stats[column] = {
                "kind": "numeric_discrete",
                "proportions": {str(float(k)): float(v) for k, v in proportions.items()},
                "mean": float(values.mean()),
            }
            continue

        # Store the reference histogram itself rather than just the quantile
        # edges. Deriving expected mass from quantile spacing assumes each bin
        # holds an equal share, which is false as soon as values pile up on one
        # number — and then a feature drifts against its own training data.
        edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
        if len(edges) < 3:
            continue
        counts, _ = np.histogram(values, bins=edges)
        total = max(int(counts.sum()), 1)
        stats[column] = {
            "kind": "numeric",
            "bin_edges": [float(e) for e in edges],
            "bin_proportions": [float(c) / total for c in counts],
            "mean": float(values.mean()),
        }

    for column in CATEGORICAL_FEATURES:
        values = frame[column].dropna()
        if values.empty:
            continue
        proportions = values.value_counts(normalize=True).head(25)
        stats[column] = {
            "kind": "categorical",
            "proportions": {str(k): float(v) for k, v in proportions.items()},
        }

    return stats
