"""Shared FastAPI dependencies: database sessions, authentication, rate limiting."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import hash_api_key
from app.db.models import ApiKey, Tenant
from app.db.session import get_session
from app.services.tenants import get_or_create_default


def resolve_tenant(
    session: Session = Depends(get_session),
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> Tenant:
    """
    Identify the caller's workspace.

    Accepts `Authorization: Bearer <key>` or `X-API-Key: <key>`. When
    require_api_key is off, unauthenticated callers get the default workspace
    so a local install works with no setup.
    """
    presented = x_api_key
    if not presented and authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()

    if not presented:
        if settings.require_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="An API key is required. Send it as 'X-API-Key' or 'Authorization: Bearer'.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return get_or_create_default(session)

    record = session.scalar(
        select(ApiKey).where(
            ApiKey.key_hash == hash_api_key(presented), ApiKey.revoked.is_(False)
        )
    )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or revoked API key."
        )

    tenant = session.get(Tenant, record.tenant_id)
    if tenant is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Workspace no longer exists.")

    record.last_used_at = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    return tenant


class SlidingWindowLimiter:
    """
    In-process rate limiter.

    Deliberately simple: one process, one memory. A multi-worker deployment
    should put a shared limiter in front (Redis, or the ingress), which is why
    this is configurable rather than load-bearing.
    """

    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            bucket = self._hits[key]
            while bucket and now - bucket[0] > self.window:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            return True

    def retry_after(self, key: str) -> int:
        with self._lock:
            bucket = self._hits.get(key)
            if not bucket:
                return 1
            return max(1, int(self.window - (time.monotonic() - bucket[0])) + 1)


limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)


def rate_limit(request: Request, tenant: Tenant = Depends(resolve_tenant)) -> Tenant:
    """Per-tenant request budget, applied to the scoring endpoints."""
    if not settings.rate_limit_enabled:
        return tenant

    if not limiter.check(tenant.id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit of {settings.rate_limit_per_minute} requests/minute exceeded.",
            headers={"Retry-After": str(limiter.retry_after(tenant.id))},
        )
    return tenant
