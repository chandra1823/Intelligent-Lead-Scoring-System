"""
Lead ingestion, scoring, and queue services.

The API routes and the MCP server both call into here, so the two surfaces
cannot drift apart in behaviour.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decrypt_secrets, strip_pii
from app.db.models import AuditEvent, Lead, ModelVersion, Prediction, Source, SyncRun, Tenant
from connectors.base import ConnectorError
from connectors.registry import build_source
from ml.canonical import ID_COLUMN, TARGET_COLUMN, VALUE_COLUMN
from ml.decisions import queue_summary, rank_leads
from ml.features import clean_frame
from ml.mapping import apply_mapping, propose_mapping
from ml.model_service import LeadScoringModel, band_for
from ml.registry import registry

TRUTHY = {"1", "true", "yes", "won", "converted", "y"}
FALSY = {"0", "false", "no", "lost", "n"}


def audit(session: Session, tenant_id: str | None, action: str, target: str = "", **detail: Any) -> None:
    session.add(
        AuditEvent(tenant_id=tenant_id, action=action, target=target or None, detail=detail)
    )


# --------------------------------------------------------------------- models


def active_model_for(session: Session, tenant: Tenant) -> tuple[LeadScoringModel, ModelVersion | None]:
    """Resolve the model that should score this tenant, falling back to base."""
    record = session.scalar(
        select(ModelVersion)
        .where(ModelVersion.tenant_id == tenant.id, ModelVersion.is_active.is_(True))
        .order_by(ModelVersion.created_at.desc())
    )

    if record is not None:
        model = registry.resolve(record.artifact_dir)
        if model.models:
            return model, record

    return registry.base_model(), None


# ------------------------------------------------------------------- ingestion


def _coerce_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in TRUTHY:
        return True
    if text in FALSY:
        return False
    return None


def _coerce_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None  # reject NaN


def canonicalise(records: list[dict[str, Any]], mapping: dict[str, str]) -> pd.DataFrame:
    """Apply a confirmed mapping to raw source records."""
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    return apply_mapping(frame, mapping) if mapping else frame


def upsert_leads(
    session: Session,
    tenant: Tenant,
    source: Source,
    frame: pd.DataFrame,
) -> tuple[int, int]:
    """
    Insert or update leads from a canonical frame.

    Returns (created, updated). Records without an external id are skipped
    rather than duplicated on every sync.
    """
    if frame.empty:
        return 0, 0

    created = updated = 0
    rows = frame.where(pd.notna(frame), None).to_dict(orient="records")

    existing = {
        lead.external_id: lead
        for lead in session.scalars(select(Lead).where(Lead.source_id == source.id))
    }

    for index, row in enumerate(rows):
        external_id = row.get(ID_COLUMN)
        if external_id in (None, ""):
            # Stable synthetic id so re-syncing a source without ids does not
            # create a fresh copy of every row each time.
            external_id = f"{source.id}-row-{index}"
        external_id = str(external_id)

        payload = {k: v for k, v in row.items() if k not in {ID_COLUMN, "display_name"}}
        payload = strip_pii(payload)

        converted = _coerce_bool(row.get(TARGET_COLUMN))
        deal_value = _coerce_float(row.get(VALUE_COLUMN))
        display_name = row.get("display_name")

        lead = existing.get(external_id)
        if lead is None:
            lead = Lead(
                tenant_id=tenant.id,
                source_id=source.id,
                external_id=external_id,
                display_name=str(display_name) if display_name else None,
                payload=payload,
                converted=converted,
                deal_value=deal_value,
            )
            session.add(lead)
            existing[external_id] = lead
            created += 1
        else:
            lead.payload = payload
            if display_name:
                lead.display_name = str(display_name)
            if converted is not None:
                lead.converted = converted
            if deal_value is not None:
                lead.deal_value = deal_value
            lead.updated_at = datetime.now(timezone.utc)
            updated += 1

    return created, updated


def score_leads(
    session: Session,
    tenant: Tenant,
    leads: list[Lead],
    log_predictions: bool = True,
) -> int:
    """Score leads in one vectorised pass and persist the results."""
    if not leads:
        return 0

    model, version_record = active_model_for(session, tenant)
    frame = clean_frame(pd.DataFrame([lead.payload or {} for lead in leads]))
    probabilities = model.predict_frame(frame)

    now = datetime.now(timezone.utc)
    for lead, probability in zip(leads, probabilities, strict=False):
        lead.latest_probability = float(probability)
        lead.latest_band = band_for(float(probability))
        lead.scored_at = now

        if log_predictions:
            session.add(
                Prediction(
                    tenant_id=tenant.id,
                    lead_id=lead.id,
                    model_version=version_record.version if version_record else model.version,
                    model_tier=model.tier,
                    probability=float(probability),
                    prediction=int(probability >= settings.decision_threshold),
                    inputs=strip_pii(lead.payload or {}),
                    components={},
                )
            )

    return len(leads)


def sync_source(
    session: Session,
    tenant: Tenant,
    source: Source,
    limit: int = 5000,
    score: bool = True,
) -> dict[str, Any]:
    """
    Pull records from a source, canonicalise, store, and score.

    Refuses to run until the mapping is confirmed — importing thousands of rows
    under a guessed mapping is expensive to undo.
    """
    run = SyncRun(tenant_id=tenant.id, source_id=source.id, status="running")
    session.add(run)
    session.flush()

    if not source.mapping_confirmed:
        run.status = "blocked"
        run.error = "Schema mapping is not confirmed yet."
        run.finished_at = datetime.now(timezone.utc)
        source.last_sync_status = "blocked"
        session.commit()
        return {
            "status": "blocked",
            "detail": (
                "Confirm the schema mapping before syncing. "
                "Call map_schema, review the proposal, then confirm_mapping."
            ),
            "sync_run_id": run.id,
        }

    adapter = build_source(source.kind, source.config, decrypt_secrets(source.secrets))

    try:
        records: list[dict[str, Any]] = []
        cursor = source.sync_cursor
        pages = 0

        while len(records) < limit and pages < 100:
            result = adapter.fetch(cursor=cursor, limit=min(1000, limit - len(records)))
            records.extend(result.records)
            pages += 1
            if not result.has_more or not result.cursor or result.cursor == cursor:
                cursor = result.cursor or cursor
                break
            cursor = result.cursor

    except ConnectorError as error:
        run.status = "failed"
        run.error = str(error)
        run.finished_at = datetime.now(timezone.utc)
        source.last_sync_status = "failed"
        source.last_sync_error = str(error)
        audit(session, tenant.id, "sync.failed", source.id, error=str(error))
        session.commit()
        return {"status": "failed", "detail": str(error), "sync_run_id": run.id}

    frame = canonicalise(records, source.mapping)
    created, updated = upsert_leads(session, tenant, source, frame)
    session.flush()

    scored = 0
    if score:
        touched = list(
            session.scalars(
                select(Lead).where(Lead.source_id == source.id).order_by(Lead.updated_at.desc()).limit(limit)
            )
        )
        scored = score_leads(session, tenant, touched)

    run.status = "success"
    run.fetched = len(records)
    run.created = created
    run.updated = updated
    run.scored = scored
    run.finished_at = datetime.now(timezone.utc)

    source.last_synced_at = datetime.now(timezone.utc)
    source.sync_cursor = cursor
    source.last_sync_status = "success"
    source.last_sync_error = None

    audit(session, tenant.id, "sync.success", source.id, fetched=len(records), created=created)

    # Commit here rather than in dependency teardown: FastAPI runs teardown
    # after the response is sent, so a caller could act on "success" before the
    # rows were durable, and a follow-up request could miss them entirely.
    session.commit()

    return {
        "status": "success",
        "fetched": len(records),
        "created": created,
        "updated": updated,
        "scored": scored,
        "sync_run_id": run.id,
    }


# ----------------------------------------------------------------- inspection


def inspect_source_schema(source: Source, sample_size: int = 300) -> dict[str, Any]:
    """Pull a sample and propose a canonical mapping for it."""
    adapter = build_source(source.kind, source.config, decrypt_secrets(source.secrets))
    result = adapter.fetch(cursor=None, limit=sample_size)

    if not result.records:
        return {
            "status": "empty",
            "detail": "The source returned no records to inspect.",
            "proposal": None,
        }

    frame = pd.DataFrame(result.records)
    proposal = propose_mapping(frame, source.mapping if source.mapping_confirmed else None)

    return {
        "status": "ok",
        "detail": (
            f"{proposal.confident_count} column(s) mapped confidently, "
            f"{proposal.review_count} need confirmation."
        ),
        "sample_rows": len(result.records),
        "proposal": proposal.to_dict(),
    }


# ---------------------------------------------------------------------- queue


def priority_queue(
    session: Session,
    tenant: Tenant,
    limit: int | None = None,
    strategy: str = "expected_value",
    band: str | None = None,
    source_id: str | None = None,
    include_converted: bool = False,
) -> dict[str, Any]:
    """The work queue: ranked, capacity-aware, decay-adjusted."""
    query = select(Lead).where(Lead.tenant_id == tenant.id, Lead.latest_probability.is_not(None))

    if not include_converted:
        query = query.where((Lead.converted.is_(None)) | (Lead.converted.is_(False)))
    if source_id:
        query = query.where(Lead.source_id == source_id)
    if band:
        query = query.where(Lead.latest_band == band)

    # Pre-trim in SQL, then re-rank in Python where decay and value apply.
    candidates = list(
        session.scalars(query.order_by(Lead.latest_probability.desc()).limit(2000))
    )

    capacity = limit if limit is not None else tenant.daily_capacity
    base_rate = 0.38

    model, _ = active_model_for(session, tenant)
    if model.dataset_info.get("conversion_rate"):
        base_rate = float(model.dataset_info["conversion_rate"])

    ranked = rank_leads(
        candidates,
        capacity=capacity,
        strategy=strategy,
        default_deal_value=tenant.default_deal_value,
        base_rate=base_rate,
    )

    for item in ranked:
        lead = next((c for c in candidates if c.id == item.lead_id), None)
        item_dict = item.to_dict()
        item_dict["payload"] = lead.payload if lead else {}

    return {
        "leads": [item.to_dict() for item in ranked],
        "summary": queue_summary(ranked, capacity),
        "strategy": strategy,
        "model_tier": model.tier,
    }


def record_outcome(
    session: Session,
    tenant: Tenant,
    lead_id: str,
    converted: bool,
    deal_value: float | None = None,
) -> dict[str, Any]:
    """Capture ground truth. This is what makes retraining possible."""
    lead = session.scalar(
        select(Lead).where(Lead.id == lead_id, Lead.tenant_id == tenant.id)
    )
    if lead is None:
        lead = session.scalar(
            select(Lead).where(Lead.external_id == lead_id, Lead.tenant_id == tenant.id)
        )
    if lead is None:
        raise LookupError(f"No lead found for id '{lead_id}'")

    lead.converted = converted
    lead.converted_at = datetime.now(timezone.utc) if converted else None
    if deal_value is not None:
        lead.deal_value = deal_value

    audit(session, tenant.id, "outcome.recorded", lead.id, converted=converted)
    session.commit()

    labelled = session.scalar(
        select(func.count(Lead.id)).where(
            Lead.tenant_id == tenant.id, Lead.converted.is_not(None)
        )
    )

    return {
        "lead_id": lead.id,
        "external_id": lead.external_id,
        "converted": converted,
        "deal_value": lead.deal_value,
        "labelled_total": int(labelled or 0),
    }


def labelled_frame(session: Session, tenant: Tenant) -> pd.DataFrame:
    """Every lead with a known outcome, in canonical form, ready for training."""
    leads = list(
        session.scalars(
            select(Lead).where(Lead.tenant_id == tenant.id, Lead.converted.is_not(None))
        )
    )
    if not leads:
        return pd.DataFrame()

    rows = []
    for lead in leads:
        row = dict(lead.payload or {})
        row[TARGET_COLUMN] = int(bool(lead.converted))
        row[ID_COLUMN] = lead.external_id
        rows.append(row)

    return pd.DataFrame(rows)


def matched_predictions(session: Session, tenant: Tenant, limit: int = 5000) -> tuple[list[float], list[int]]:
    """Scored leads whose outcome is now known — the input to calibration checks."""
    leads = list(
        session.scalars(
            select(Lead)
            .where(
                Lead.tenant_id == tenant.id,
                Lead.converted.is_not(None),
                Lead.latest_probability.is_not(None),
            )
            .limit(limit)
        )
    )
    return (
        [float(lead.latest_probability) for lead in leads],
        [int(bool(lead.converted)) for lead in leads],
    )


def recent_feature_frame(session: Session, tenant: Tenant, limit: int = 1000) -> pd.DataFrame:
    """Recently scored leads, for drift comparison against training statistics."""
    leads = list(
        session.scalars(
            select(Lead)
            .where(Lead.tenant_id == tenant.id, Lead.scored_at.is_not(None))
            .order_by(Lead.scored_at.desc())
            .limit(limit)
        )
    )
    if not leads:
        return pd.DataFrame()
    return clean_frame(pd.DataFrame([lead.payload or {} for lead in leads]))
