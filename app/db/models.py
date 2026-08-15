"""
Persistence layer.

SQLite by default so a fresh clone runs with no infrastructure; point
LEAD_API_DATABASE_URL at Postgres for a real deployment. All types used here
are portable across both.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    """An isolated workspace. Every other row belongs to exactly one tenant."""

    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # Sales capacity per day, used by capacity-aware ranking.
    daily_capacity: Mapped[int] = mapped_column(Integer, default=50)
    # Fallback deal value when a source carries no revenue figure.
    default_deal_value: Mapped[float] = mapped_column(Float, default=1000.0)

    api_keys: Mapped[list[ApiKey]] = relationship(back_populates="tenant", cascade="all, delete-orphan")
    sources: Mapped[list[Source]] = relationship(back_populates="tenant", cascade="all, delete-orphan")


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    # Only the hash is stored; the plaintext key is shown once at creation.
    key_hash: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    prefix: Mapped[str] = mapped_column(String(16), nullable=False)
    label: Mapped[str] = mapped_column(String(120), default="default")
    scopes: Mapped[dict[str, Any]] = mapped_column(JSON, default=lambda: ["read", "write"])
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)

    tenant: Mapped[Tenant] = relationship(back_populates="api_keys")


class Source(Base):
    """A connected lead source: a CRM, a spreadsheet, a webhook endpoint."""

    __tablename__ = "sources"
    __table_args__ = (UniqueConstraint("tenant_id", "name", name="uq_source_name_per_tenant"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    kind: Mapped[str] = mapped_column(String(60), nullable=False)

    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # Credentials live here; encrypted at rest via app.core.security.
    secrets: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Confirmed source-column -> canonical-field mapping.
    mapping: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    mapping_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    pending_mapping: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Features a human approved despite a leakage flag.
    leakage_overrides: Mapped[list[str]] = mapped_column(JSON, default=list)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    sync_cursor: Mapped[str | None] = mapped_column(String(200), nullable=True)
    last_sync_status: Mapped[str | None] = mapped_column(String(40), nullable=True)
    last_sync_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    tenant: Mapped[Tenant] = relationship(back_populates="sources")
    leads: Mapped[list[Lead]] = relationship(back_populates="source", cascade="all, delete-orphan")


class Lead(Base):
    """A lead in canonical form. `payload` holds the mapped canonical fields."""

    __tablename__ = "leads"
    __table_args__ = (
        UniqueConstraint("source_id", "external_id", name="uq_lead_external_id_per_source"),
        Index("ix_leads_tenant_score", "tenant_id", "latest_probability"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), index=True)

    external_id: Mapped[str] = mapped_column(String(200), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(300), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    latest_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    latest_priority: Mapped[float | None] = mapped_column(Float, nullable=True)
    latest_band: Mapped[str | None] = mapped_column(String(20), nullable=True)
    scored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Ground truth, once known. Feeds retraining and calibration monitoring.
    converted: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    converted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deal_value: Mapped[float | None] = mapped_column(Float, nullable=True)

    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    source: Mapped[Source] = relationship(back_populates="leads")


class Prediction(Base):
    """
    Immutable audit record of every score produced.

    Required to answer "why was this lead deprioritised in March?" after the
    model has since been retrained.
    """

    __tablename__ = "predictions"
    __table_args__ = (Index("ix_predictions_tenant_time", "tenant_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    lead_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)

    model_version: Mapped[str] = mapped_column(String(80), nullable=False)
    model_tier: Mapped[str] = mapped_column(String(40), nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)

    inputs: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    contributions: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    components: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ModelVersion(Base):
    """A trained model belonging to a tenant, or the shared base model."""

    __tablename__ = "model_versions"
    __table_args__ = (Index("ix_models_tenant_active", "tenant_id", "is_active"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    # NULL tenant_id marks the shared base model shipped with the repo.
    tenant_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    version: Mapped[str] = mapped_column(String(80), nullable=False)

    tier: Mapped[str] = mapped_column(String(40), nullable=False)
    artifact_dir: Mapped[str] = mapped_column(String(500), nullable=False)

    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    weights: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    leakage_report: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    feature_statistics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    training_rows: Mapped[int] = mapped_column(Integer, default=0)

    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    promoted_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SyncRun(Base):
    """One execution of a source sync, successful or not."""

    __tablename__ = "sync_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    source_id: Mapped[str] = mapped_column(String(32), index=True, nullable=False)

    status: Mapped[str] = mapped_column(String(40), nullable=False)
    fetched: Mapped[int] = mapped_column(Integer, default=0)
    created: Mapped[int] = mapped_column(Integer, default=0)
    updated: Mapped[int] = mapped_column(Integer, default=0)
    scored: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditEvent(Base):
    """Append-only record of consequential actions."""

    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_tenant_time", "tenant_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    actor: Mapped[str] = mapped_column(String(120), default="system")
    action: Mapped[str] = mapped_column(String(80), nullable=False)
    target: Mapped[str | None] = mapped_column(String(200), nullable=True)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
