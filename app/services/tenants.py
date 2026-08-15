"""Tenant provisioning and API key management."""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import generate_api_key
from app.db.models import ApiKey, Tenant


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "tenant"


def get_or_create_default(session: Session) -> Tenant:
    """
    The workspace used when no API key is presented.

    A single-user local install never has to think about tenancy; a deployment
    sets require_api_key and this path is never taken.
    """
    tenant = session.scalar(select(Tenant).where(Tenant.slug == settings.default_tenant_slug))
    if tenant is None:
        tenant = Tenant(name="Default workspace", slug=settings.default_tenant_slug)
        session.add(tenant)
        session.commit()
    return tenant


def create_tenant(session: Session, name: str, daily_capacity: int = 50) -> tuple[Tenant, str]:
    """Create a workspace and its first API key. The key is returned once."""
    slug = slugify(name)
    existing = session.scalar(select(Tenant).where(Tenant.slug == slug))
    if existing is not None:
        raise ValueError(f"A workspace with slug '{slug}' already exists.")

    tenant = Tenant(name=name, slug=slug, daily_capacity=daily_capacity)
    session.add(tenant)
    session.flush()

    plaintext = issue_api_key(session, tenant, label="initial")
    return tenant, plaintext


def issue_api_key(session: Session, tenant: Tenant, label: str = "default",
                  scopes: list[str] | None = None) -> str:
    plaintext, key_hash, prefix = generate_api_key()
    session.add(
        ApiKey(
            tenant_id=tenant.id,
            key_hash=key_hash,
            prefix=prefix,
            label=label,
            scopes=scopes or ["read", "write"],
        )
    )
    return plaintext


def revoke_api_key(session: Session, tenant: Tenant, key_id: str) -> bool:
    key = session.scalar(
        select(ApiKey).where(ApiKey.id == key_id, ApiKey.tenant_id == tenant.id)
    )
    if key is None:
        return False
    key.revoked = True
    return True


def describe_tenant(tenant: Tenant) -> dict[str, Any]:
    return {
        "id": tenant.id,
        "name": tenant.name,
        "slug": tenant.slug,
        "daily_capacity": tenant.daily_capacity,
        "default_deal_value": tenant.default_deal_value,
        "created_at": tenant.created_at.isoformat() if tenant.created_at else None,
    }
