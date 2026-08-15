"""Adapter registry — the one place that knows which connectors exist."""

from __future__ import annotations

from typing import Any

from connectors.base import CONFIG_SCHEMAS, BaseSource, ConnectorError
from connectors.csv_source import CsvSource, GoogleSheetSource, InlineSource
from connectors.hubspot import HubSpotSource

_ADAPTERS: dict[str, type[BaseSource]] = {
    CsvSource.kind: CsvSource,
    GoogleSheetSource.kind: GoogleSheetSource,
    InlineSource.kind: InlineSource,
    HubSpotSource.kind: HubSpotSource,
}


def register(adapter: type[BaseSource]) -> None:
    """Add a connector. Third-party adapters call this at import time."""
    _ADAPTERS[adapter.kind] = adapter


def available_kinds() -> list[dict[str, Any]]:
    return [
        {
            "kind": kind,
            "supports_writeback": adapter.supports_writeback,
            "config_schema": CONFIG_SCHEMAS.get(kind, {}),
        }
        for kind, adapter in sorted(_ADAPTERS.items())
    ]


def build_source(
    kind: str,
    config: dict[str, Any] | None = None,
    secrets: dict[str, Any] | None = None,
) -> BaseSource:
    adapter = _ADAPTERS.get(kind)
    if adapter is None:
        known = ", ".join(sorted(_ADAPTERS))
        raise ConnectorError(f"Unknown source kind '{kind}'. Available: {known}")
    return adapter(config or {}, secrets or {})
