"""Scoring, explanation, and metrics endpoints."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import rate_limit, resolve_tenant
from app.db.models import Tenant
from app.db.session import get_session
from app.models.schemas import (
    BatchScoreItem,
    BatchScoreRequest,
    BatchScoreResponse,
    ExplainResponse,
    LeadFeatures,
    MetricsResponse,
    PredictionResponse,
)
from app.services.leads import active_model_for
from ml.canonical import describe_schema
from ml.decisions import next_best_action
from ml.features import clean_frame
from ml.model_service import band_for

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT / "frontend"
DOCS_DIR = ROOT / "docs"

router = APIRouter()


@router.get("/")
def root() -> dict:
    return {
        "message": "Lead Scoring API is running",
        "ui": "/ui",
        "docs": "/docs",
        "roadmap": "/ui/roadmap",
    }


@router.get("/ui", include_in_schema=False)
def ui_landing() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@router.get("/ui/dashboard", include_in_schema=False)
def ui_dashboard() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "dashboard.html")


@router.get("/ui/roadmap", include_in_schema=False)
def ui_roadmap() -> FileResponse:
    """The Phase 1 / Phase 2 roadmap document."""
    return FileResponse(DOCS_DIR / "roadmap.html")


@router.get("/health")
def health(
    tenant: Tenant = Depends(resolve_tenant),
    session: Session = Depends(get_session),
) -> dict:
    model, record = active_model_for(session, tenant)
    payload = model.health()
    payload.update(
        {
            "tenant": tenant.slug,
            "model_source": "tenant" if record else "base",
        }
    )
    return payload


@router.get("/schema")
def canonical_schema() -> dict:
    """The canonical lead schema every source is mapped onto."""
    return {"fields": describe_schema()}


@router.get("/metrics", response_model=MetricsResponse)
def metrics(
    tenant: Tenant = Depends(resolve_tenant),
    session: Session = Depends(get_session),
) -> MetricsResponse:
    model, _ = active_model_for(session, tenant)
    return MetricsResponse(
        trained_models=sorted(model.models),
        weights=model.weights,
        metrics=model.metrics,
        dataset=model.dataset_info,
        model_version=model.version,
        model_tier=model.tier,
        leakage_report=model.leakage_report,
        category_options=model.category_options,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(
    payload: LeadFeatures,
    tenant: Tenant = Depends(rate_limit),
    session: Session = Depends(get_session),
) -> PredictionResponse:
    model, _ = active_model_for(session, tenant)
    result = model.predict(payload.to_columns())

    return PredictionResponse(
        prediction=result.prediction,
        probability=round(result.probability, 4),
        label=result.centralized_output["label"],
        band=result.band,
        model=result.model_name,
        centralized_output=result.centralized_output,
    )


@router.post("/predict/batch", response_model=BatchScoreResponse)
def predict_batch(
    payload: BatchScoreRequest,
    tenant: Tenant = Depends(rate_limit),
    session: Session = Depends(get_session),
) -> BatchScoreResponse:
    """Score many leads in one vectorised pass."""
    model, _ = active_model_for(session, tenant)

    rows = [lead.to_columns() for lead in payload.leads]
    if not rows:
        return BatchScoreResponse(scored=0, results=[], model_tier=model.tier)

    probabilities = model.predict_frame(clean_frame(pd.DataFrame(rows)))

    return BatchScoreResponse(
        scored=len(probabilities),
        results=[
            BatchScoreItem(
                index=index,
                probability=round(float(probability), 4),
                prediction=int(probability >= 0.5),
                band=band_for(float(probability)),
            )
            for index, probability in enumerate(probabilities)
        ],
        model_tier=model.tier,
    )


@router.post("/explain", response_model=ExplainResponse)
def explain(
    payload: LeadFeatures,
    tenant: Tenant = Depends(rate_limit),
    session: Session = Depends(get_session),
) -> ExplainResponse:
    model, _ = active_model_for(session, tenant)
    features = payload.to_columns()
    result = model.explain(features)

    return ExplainResponse(
        prediction=result.prediction,
        probability=round(result.probability, 4),
        label=result.centralized_output["label"],
        band=result.band,
        summary=model.summarize(result),
        contributions=result.contributions,
        next_best_action=next_best_action(features, result.probability),
        input_features=payload.model_dump(exclude_none=True),
    )
