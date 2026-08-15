"""Source management, sync, queue, training, and monitoring endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import resolve_tenant
from app.core.config import settings
from app.core.security import decrypt_secrets, encrypt_secrets
from app.db.models import Lead, ModelVersion, Source, SyncRun, Tenant
from app.db.session import get_session
from app.models.schemas import (
    MappingConfirm,
    OutcomeRequest,
    SourceCreate,
    SourceResponse,
    SyncRequest,
    TenantCreate,
    TrainRequest,
)
from app.services import leads as lead_service
from app.services.tenants import create_tenant, describe_tenant
from connectors.base import ConnectorError
from connectors.registry import available_kinds, build_source
from ml.mapping import MappingProposal, merge_confirmations
from ml.monitoring import calibration_report, detect_drift, health_rollup
from ml.registry import TIER_GENERIC, TIER_RECALIBRATED, registry, tier_for_label_count
from ml.training import TrainingError, plan_for, recalibrate_base, should_promote, train_model

router = APIRouter(prefix="/v1", tags=["platform"])


def _describe_source(source: Source) -> SourceResponse:
    try:
        adapter = build_source(source.kind, source.config, {})
        writeback = adapter.supports_writeback
    except ConnectorError:
        writeback = False

    return SourceResponse(
        id=source.id,
        name=source.name,
        kind=source.kind,
        config=source.config,
        mapping=source.mapping or {},
        mapping_confirmed=source.mapping_confirmed,
        supports_writeback=writeback,
        last_synced_at=source.last_synced_at.isoformat() if source.last_synced_at else None,
        last_sync_status=source.last_sync_status,
        last_sync_error=source.last_sync_error,
    )


def _get_source(session: Session, tenant: Tenant, source_id: str) -> Source:
    source = session.scalar(
        select(Source).where(Source.id == source_id, Source.tenant_id == tenant.id)
    )
    if source is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source not found.")
    return source


# ------------------------------------------------------------------- tenants


@router.get("/me")
def whoami(tenant: Tenant = Depends(resolve_tenant)) -> dict[str, Any]:
    return describe_tenant(tenant)


@router.post("/tenants", status_code=status.HTTP_201_CREATED)
def add_tenant(
    payload: TenantCreate,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """
    Create a workspace and return its first API key.

    The key is shown once and never again — only its hash is stored.
    """
    try:
        tenant, api_key = create_tenant(session, payload.name, payload.daily_capacity)
        session.commit()
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error

    return {
        **describe_tenant(tenant),
        "api_key": api_key,
        "warning": "Store this key now. It cannot be retrieved again.",
    }


# ----------------------------------------------------------------- connectors


@router.get("/connectors")
def list_connectors() -> dict[str, Any]:
    return {"connectors": available_kinds()}


@router.get("/sources", response_model=list[SourceResponse])
def list_sources(
    tenant: Tenant = Depends(resolve_tenant),
    session: Session = Depends(get_session),
) -> list[SourceResponse]:
    sources = session.scalars(select(Source).where(Source.tenant_id == tenant.id))
    return [_describe_source(source) for source in sources]


@router.post("/sources", response_model=SourceResponse, status_code=status.HTTP_201_CREATED)
def add_source(
    payload: SourceCreate,
    tenant: Tenant = Depends(resolve_tenant),
    session: Session = Depends(get_session),
) -> SourceResponse:
    """Register a lead source. Credentials are encrypted before storage."""
    try:
        adapter = build_source(payload.kind, payload.config, payload.secrets)
        check = adapter.test_connection()
    except ConnectorError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error

    if not check.ok:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not connect: {check.detail}",
        )

    existing = session.scalar(
        select(Source).where(Source.tenant_id == tenant.id, Source.name == payload.name)
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A source named '{payload.name}' already exists.",
        )

    source = Source(
        tenant_id=tenant.id,
        name=payload.name,
        kind=payload.kind,
        config=payload.config,
        secrets=encrypt_secrets(payload.secrets),
    )
    session.add(source)
    session.flush()

    lead_service.audit(session, tenant.id, "source.created", source.id, kind=payload.kind)
    session.commit()
    return _describe_source(source)


@router.delete("/sources/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_source(
    source_id: str,
    tenant: Tenant = Depends(resolve_tenant),
    session: Session = Depends(get_session),
) -> None:
    source = _get_source(session, tenant, source_id)
    session.delete(source)
    lead_service.audit(session, tenant.id, "source.deleted", source_id)
    session.commit()


@router.post("/sources/{source_id}/inspect")
def inspect_source(
    source_id: str,
    tenant: Tenant = Depends(resolve_tenant),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Sample the source and propose a canonical mapping for review."""
    source = _get_source(session, tenant, source_id)

    try:
        result = lead_service.inspect_source_schema(source)
    except ConnectorError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error

    if result.get("proposal"):
        source.pending_mapping = result["proposal"]
        session.commit()

    return result


@router.post("/sources/{source_id}/mapping")
def confirm_mapping(
    source_id: str,
    payload: MappingConfirm,
    tenant: Tenant = Depends(resolve_tenant),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """
    Confirm the mapping for a source.

    Corrections are merged over the pending proposal, so a caller only has to
    send the columns they want to change.
    """
    source = _get_source(session, tenant, source_id)

    base_mapping: dict[str, str] = {}
    if payload.accept_proposal and source.pending_mapping:
        base_mapping = dict(source.pending_mapping.get("mapping") or {})

    proposal = MappingProposal(
        mapping=base_mapping, proposals=[], unmapped_columns=[], missing_features=[],
        confident_count=0, review_count=0,
    )

    try:
        merged = merge_confirmations(proposal, payload.mapping)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error

    if not merged:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The confirmed mapping is empty; at least one column must map to a canonical field.",
        )

    source.mapping = merged
    source.mapping_confirmed = True
    source.pending_mapping = None

    lead_service.audit(session, tenant.id, "mapping.confirmed", source.id, fields=len(merged))
    session.commit()

    from ml.mapping import mapping_coverage

    return {"source_id": source.id, "mapping": merged, "coverage": mapping_coverage(merged)}


@router.post("/sources/{source_id}/sync")
def sync(
    source_id: str,
    payload: SyncRequest,
    tenant: Tenant = Depends(resolve_tenant),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Pull records, canonicalise, store, and score."""
    source = _get_source(session, tenant, source_id)
    return lead_service.sync_source(session, tenant, source, limit=payload.limit, score=payload.score)


@router.get("/sources/{source_id}/runs")
def sync_history(
    source_id: str,
    tenant: Tenant = Depends(resolve_tenant),
    session: Session = Depends(get_session),
    limit: int = Query(default=20, ge=1, le=200),
) -> dict[str, Any]:
    _get_source(session, tenant, source_id)
    runs = session.scalars(
        select(SyncRun)
        .where(SyncRun.source_id == source_id)
        .order_by(SyncRun.started_at.desc())
        .limit(limit)
    )
    return {
        "runs": [
            {
                "id": run.id,
                "status": run.status,
                "fetched": run.fetched,
                "created": run.created,
                "updated": run.updated,
                "scored": run.scored,
                "error": run.error,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            }
            for run in runs
        ]
    }


@router.post("/sources/{source_id}/push-scores")
def push_scores(
    source_id: str,
    tenant: Tenant = Depends(resolve_tenant),
    session: Session = Depends(get_session),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    """Write current scores back to the source system."""
    source = _get_source(session, tenant, source_id)
    adapter = build_source(source.kind, source.config, decrypt_secrets(source.secrets))

    if not adapter.supports_writeback:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{source.kind}' sources are read-only.",
        )

    scored = list(
        session.scalars(
            select(Lead)
            .where(Lead.source_id == source.id, Lead.latest_probability.is_not(None))
            .order_by(Lead.latest_probability.desc())
            .limit(limit)
        )
    )

    pushed, failures = 0, []
    for lead in scored:
        try:
            adapter.push_score(lead.external_id, lead.latest_probability, lead.latest_band or "")
            pushed += 1
        except ConnectorError as error:
            failures.append({"external_id": lead.external_id, "error": str(error)})

    lead_service.audit(session, tenant.id, "scores.pushed", source.id, pushed=pushed)
    session.commit()
    return {"pushed": pushed, "attempted": len(scored), "failures": failures[:10]}


# --------------------------------------------------------------------- queue


@router.get("/leads/priority")
def priority(
    tenant: Tenant = Depends(resolve_tenant),
    session: Session = Depends(get_session),
    limit: int | None = Query(default=None, ge=1, le=1000),
    strategy: str = Query(default="expected_value", pattern="^(expected_value|probability|value)$"),
    band: str | None = Query(default=None, pattern="^(hot|warm|cool|cold)$"),
    source_id: str | None = None,
) -> dict[str, Any]:
    """The ranked work queue — capacity-aware, decay-adjusted."""
    return lead_service.priority_queue(
        session, tenant, limit=limit, strategy=strategy, band=band, source_id=source_id
    )


@router.post("/leads/outcome")
def outcome(
    payload: OutcomeRequest,
    tenant: Tenant = Depends(resolve_tenant),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Record ground truth so the model can learn from it."""
    try:
        result = lead_service.record_outcome(
            session, tenant, payload.lead_id, payload.converted, payload.deal_value
        )
    except LookupError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    result["training_plan"] = plan_for(result["labelled_total"])
    return result


@router.get("/leads/stats")
def lead_stats(
    tenant: Tenant = Depends(resolve_tenant),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    total = session.scalar(select(func.count(Lead.id)).where(Lead.tenant_id == tenant.id)) or 0
    scored = (
        session.scalar(
            select(func.count(Lead.id)).where(
                Lead.tenant_id == tenant.id, Lead.latest_probability.is_not(None)
            )
        )
        or 0
    )
    labelled = (
        session.scalar(
            select(func.count(Lead.id)).where(
                Lead.tenant_id == tenant.id, Lead.converted.is_not(None)
            )
        )
        or 0
    )
    converted = (
        session.scalar(
            select(func.count(Lead.id)).where(
                Lead.tenant_id == tenant.id, Lead.converted.is_(True)
            )
        )
        or 0
    )

    return {
        "total": int(total),
        "scored": int(scored),
        "labelled": int(labelled),
        "converted": int(converted),
        "conversion_rate": round(converted / labelled, 4) if labelled else None,
        "training_plan": plan_for(int(labelled)),
    }


# ------------------------------------------------------------------ training


@router.post("/train")
def train(
    payload: TrainRequest,
    tenant: Tenant = Depends(resolve_tenant),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """
    Train or recalibrate a model for this workspace.

    The cold-start tier is chosen from how many labelled leads exist; the
    result is only promoted if it beats the incumbent.
    """
    frame = lead_service.labelled_frame(session, tenant)
    labelled = len(frame)
    tier = tier_for_label_count(labelled)

    if tier == TIER_GENERIC and not payload.force:
        return {
            "status": "skipped",
            "detail": plan_for(labelled)["detail"],
            "training_plan": plan_for(labelled),
        }

    version_dir = settings.tenant_artifact_dir(tenant.id, "pending")

    try:
        if tier == TIER_RECALIBRATED:
            result = recalibrate_base(frame, version_dir)
        else:
            result = train_model(
                frame, version_dir, tier=tier, leakage_overrides=payload.leakage_overrides
            )
    except TrainingError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error

    champion = session.scalar(
        select(ModelVersion)
        .where(ModelVersion.tenant_id == tenant.id, ModelVersion.is_active.is_(True))
        .order_by(ModelVersion.created_at.desc())
    )

    promote, reason = should_promote(result.metrics, champion.metrics if champion else None)

    final_dir = settings.tenant_artifact_dir(tenant.id, result.version)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if version_dir.exists():
        if final_dir.exists():
            import shutil

            shutil.rmtree(final_dir)
        version_dir.rename(final_dir)

    record = ModelVersion(
        tenant_id=tenant.id,
        version=result.version,
        tier=result.tier,
        artifact_dir=str(final_dir),
        metrics=result.metrics,
        weights=result.weights,
        leakage_report=result.leakage,
        training_rows=result.training_rows,
        is_active=promote,
        promoted_reason=reason,
    )
    session.add(record)

    if promote:
        if champion is not None:
            champion.is_active = False
        registry.invalidate()

    lead_service.audit(
        session, tenant.id, "model.trained", result.version, promoted=promote, tier=result.tier
    )
    session.commit()

    # Set on the result before serialising; spreading to_dict() over these keys
    # would overwrite them with the dataclass defaults.
    result.promoted = promote
    result.promotion_reason = reason

    return {"status": "trained", "reason": reason, **result.to_dict()}


@router.get("/models")
def list_models(
    tenant: Tenant = Depends(resolve_tenant),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    records = session.scalars(
        select(ModelVersion)
        .where(ModelVersion.tenant_id == tenant.id)
        .order_by(ModelVersion.created_at.desc())
    )
    return {
        "models": [
            {
                "version": record.version,
                "tier": record.tier,
                "is_active": record.is_active,
                "training_rows": record.training_rows,
                "roc_auc": record.metrics.get("hybrid_ensemble", {}).get("roc_auc"),
                "promoted_reason": record.promoted_reason,
                "created_at": record.created_at.isoformat() if record.created_at else None,
            }
            for record in records
        ]
    }


# ---------------------------------------------------------------- monitoring


@router.get("/monitoring")
def monitoring(
    tenant: Tenant = Depends(resolve_tenant),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Drift and calibration health for the active model."""
    model, _ = lead_service.active_model_for(session, tenant)

    live = lead_service.recent_feature_frame(session, tenant)
    drift = (
        detect_drift(model.feature_statistics, live)
        if model.feature_statistics and not live.empty
        else {"status": "insufficient_data", "detail": "Not enough scored leads yet.", "signals": []}
    )

    probabilities, outcomes = lead_service.matched_predictions(session, tenant)
    calibration = (
        calibration_report(probabilities, outcomes)
        if probabilities
        else {"status": "insufficient_data", "detail": "No leads with both a score and an outcome yet."}
    )

    return {
        "model_version": model.version,
        "model_tier": model.tier,
        "drift": drift,
        "calibration": calibration,
        "health": health_rollup(drift, calibration),
    }


@router.get("/monitoring/lift")
def lift(
    tenant: Tenant = Depends(resolve_tenant),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Decile lift - the number that justifies changing how a team works."""
    from ml.decisions import lift_by_decile

    probabilities, outcomes = lead_service.matched_predictions(session, tenant)
    if not probabilities:
        return {
            "status": "insufficient_data",
            "detail": "No scored leads with known outcomes yet.",
            "deciles": [],
        }

    table = lift_by_decile(list(zip(probabilities, outcomes, strict=True)))
    return {
        "status": "ok",
        "sample_size": len(probabilities),
        "top_decile_lift": table[0]["lift"] if table else None,
        "deciles": table,
    }
