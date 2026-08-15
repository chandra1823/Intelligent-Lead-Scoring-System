"""
The lead source interface.

Adding a CRM means writing one class against this protocol and registering it.
Everything downstream — mapping, scoring, storage, writeback — is shared, so an
adapter only has to know how to talk to its own API.

Contract:
  * fetch() yields raw records as plain dicts, exactly as the source returns
    them. Do not rename or clean anything; the schema mapper handles that.
  * fetch() is incremental. Honour `cursor` and return a new one so the next
    sync resumes rather than re-reading everything.
  * push_score() is optional. Raise UnsupportedOperation if the source is
    read-only.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class ConnectorError(RuntimeError):
    """A source failed in a way the caller should surface, not retry blindly."""


class UnsupportedOperation(ConnectorError):
    """The source cannot do this — e.g. writeback to a CSV upload."""


class AuthenticationError(ConnectorError):
    """Credentials are missing, wrong, or expired."""


@dataclass
class FetchResult:
    records: list[dict[str, Any]]
    cursor: str | None = None
    has_more: bool = False


@dataclass
class ConnectionStatus:
    ok: bool
    detail: str
    sample_columns: list[str] = field(default_factory=list)
    record_estimate: int | None = None


@runtime_checkable
class LeadSource(Protocol):
    """Implemented by every adapter."""

    kind: str
    supports_writeback: bool

    def test_connection(self) -> ConnectionStatus:
        """Verify credentials and report what the source looks like."""

    def fetch(self, cursor: str | None = None, limit: int = 1000) -> FetchResult:
        """Pull a page of raw records, resuming from `cursor` when given."""

    def push_score(self, external_id: str, probability: float, band: str) -> None:
        """Write a score back to the source record."""

    def describe(self) -> dict[str, Any]:
        """Non-secret configuration, safe to return over the API."""


class BaseSource:
    """Shared behaviour. Adapters subclass this rather than the Protocol."""

    kind: str = "base"
    supports_writeback: bool = False

    def __init__(self, config: dict[str, Any], secrets: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.secrets = secrets or {}

    def test_connection(self) -> ConnectionStatus:  # pragma: no cover - overridden
        raise NotImplementedError

    def fetch(self, cursor: str | None = None, limit: int = 1000) -> FetchResult:  # pragma: no cover
        raise NotImplementedError

    def push_score(self, external_id: str, probability: float, band: str) -> None:
        raise UnsupportedOperation(f"{self.kind} does not support score writeback")

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "supports_writeback": self.supports_writeback,
            "config": {k: v for k, v in self.config.items() if "secret" not in k.lower()},
        }

    def iter_all(self, limit_pages: int = 100, page_size: int = 1000) -> Iterator[dict[str, Any]]:
        """Walk every page, stopping at `limit_pages` so a bad cursor cannot loop forever."""
        cursor: str | None = None
        for _ in range(limit_pages):
            result = self.fetch(cursor=cursor, limit=page_size)
            yield from result.records
            if not result.has_more or not result.cursor or result.cursor == cursor:
                return
            cursor = result.cursor


# Declared here so adapters and the sync engine agree on the required fields.
CONFIG_SCHEMAS: dict[str, dict[str, Any]] = {}


def register_config_schema(kind: str, schema: dict[str, Any]) -> None:
    CONFIG_SCHEMAS[kind] = schema
