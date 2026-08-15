"""
Train the shared base model from the bundled dataset.

This is the Tier 0 model: what scores a brand-new workspace before it has any
outcomes of its own. It trains on the canonical schema, so a customer whose
columns map onto the same fields gets useful scores on day one.

    python scripts/train_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from ml.features import X_EDUCATION_MAPPING  # noqa: E402
from ml.mapping import apply_mapping  # noqa: E402
from ml.registry import TIER_GENERIC  # noqa: E402
from ml.training import TrainingError, train_model  # noqa: E402

DATASET_PATH = ROOT / "Lead Scoring.csv"


def main() -> None:
    if not DATASET_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATASET_PATH}")

    raw = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(raw):,} rows x {len(raw.columns)} columns")

    # The bundled CSV goes through the same mapping layer as any CRM.
    canonical = apply_mapping(raw, X_EDUCATION_MAPPING)
    print(f"Mapped onto {len(canonical.columns)} canonical field(s)")

    artifact_dir = settings.base_artifact_dir
    print(f"Training base model into {artifact_dir}\n")

    try:
        result = train_model(canonical, artifact_dir, tier=TIER_GENERIC)
    except TrainingError as error:
        raise SystemExit(f"Training failed: {error}") from error

    for name, scores in result.metrics.items():
        label = "hybrid" if name == "hybrid_ensemble" else name
        print(
            f"{label:16s} acc={scores['accuracy']:.4f} auc={scores['roc_auc']:.4f} "
            f"pr_auc={scores['pr_auc']:.4f} brier={scores['brier']:.4f}"
        )

    print(f"\nWeights: {json.dumps({k: round(v, 4) for k, v in result.weights.items()})}")

    leakage = result.leakage or {}
    if leakage.get("findings"):
        print(f"\nLeakage screen flagged {len(leakage['findings'])} feature(s):")
        for finding in leakage["findings"]:
            print(f"  [{finding['severity']}] {finding['feature']}: {finding['detail']}")
    else:
        print("\nLeakage screen: clean.")

    for note in result.notes:
        print(f"note: {note}")

    print(f"\nBase model {result.version} written to {result.artifact_dir}")


if __name__ == "__main__":
    main()
