"""
Model registry.

Resolves which model scores a given tenant, caches loaded artifacts, and keeps
the cache coherent when a new version is promoted. Falls back down the chain —
tenant model, base model, heuristic — so scoring never hard-fails.
"""

from __future__ import annotations

import threading
from pathlib import Path

from app.core.config import settings
from ml.model_service import LeadScoringModel

# Cold-start tiers. See ROADMAP.md for the rationale.
TIER_GENERIC = "generic"
TIER_RECALIBRATED = "recalibrated"
TIER_TENANT = "tenant"
TIER_CONTINUOUS = "continuous"

TIER_LABELS = {
    TIER_GENERIC: "Generic scoring - not yet trained on your data",
    TIER_RECALIBRATED: "Tuned to your conversion rate",
    TIER_TENANT: "Trained on your pipeline",
    TIER_CONTINUOUS: "Trained on your pipeline, retrained continuously",
}


def tier_for_label_count(labelled: int) -> str:
    if labelled >= settings.tier3_min_labels:
        return TIER_CONTINUOUS
    if labelled >= settings.tier2_min_labels:
        return TIER_TENANT
    if labelled >= settings.tier1_min_labels:
        return TIER_RECALIBRATED
    return TIER_GENERIC


class ModelRegistry:
    """Process-wide cache of loaded models, keyed by artifact directory."""

    def __init__(self) -> None:
        self._cache: dict[str, LeadScoringModel] = {}
        self._lock = threading.Lock()

    def load(self, artifact_dir: Path) -> LeadScoringModel:
        key = str(Path(artifact_dir).resolve())
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        model = LeadScoringModel(artifact_dir)
        with self._lock:
            self._cache[key] = model
        return model

    def base_model(self) -> LeadScoringModel:
        return self.load(settings.base_artifact_dir)

    def invalidate(self, artifact_dir: Path | None = None) -> None:
        """Drop cached artifacts so the next score picks up a promotion."""
        with self._lock:
            if artifact_dir is None:
                self._cache.clear()
            else:
                self._cache.pop(str(Path(artifact_dir).resolve()), None)

    def resolve(self, artifact_dir: Path | None) -> LeadScoringModel:
        """
        Load a tenant model, falling back to the base model.

        A tenant directory that exists but holds no usable artifacts is treated
        as absent rather than as a failure.
        """
        if artifact_dir is not None:
            candidate = self.load(artifact_dir)
            if candidate.models:
                return candidate
        return self.base_model()

    @property
    def cached_paths(self) -> list[str]:
        with self._lock:
            return sorted(self._cache)


registry = ModelRegistry()
