"""
CSV and Google Sheets sources.

Sheets are read through the published-CSV URL, which needs no OAuth app and no
service account — the lowest-friction way for someone to point this at real
data five minutes after cloning.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from connectors.base import (
    BaseSource,
    ConnectionStatus,
    ConnectorError,
    FetchResult,
    register_config_schema,
)


def _rows_from_text(text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text))
    return [{k: v for k, v in row.items() if k} for row in reader]


class CsvSource(BaseSource):
    """A CSV file on disk. Cursor is the row offset."""

    kind = "csv"
    supports_writeback = False

    def _path(self) -> Path:
        raw = self.config.get("path")
        if not raw:
            raise ConnectorError("csv source requires a 'path'")
        path = Path(raw).expanduser()
        if not path.exists():
            raise ConnectorError(f"file not found: {path}")
        return path

    def _load(self) -> list[dict[str, Any]]:
        return _rows_from_text(self._path().read_text(encoding="utf-8-sig"))

    def test_connection(self) -> ConnectionStatus:
        try:
            rows = self._load()
        except ConnectorError as error:
            return ConnectionStatus(ok=False, detail=str(error))

        return ConnectionStatus(
            ok=True,
            detail=f"Read {len(rows)} row(s) from {self._path().name}",
            sample_columns=list(rows[0].keys()) if rows else [],
            record_estimate=len(rows),
        )

    def fetch(self, cursor: str | None = None, limit: int = 1000) -> FetchResult:
        rows = self._load()
        offset = int(cursor) if cursor and cursor.isdigit() else 0
        window = rows[offset : offset + limit]
        next_offset = offset + len(window)
        return FetchResult(
            records=window,
            cursor=str(next_offset),
            has_more=next_offset < len(rows),
        )


class GoogleSheetSource(BaseSource):
    """
    A Google Sheet published to the web, or any HTTP-reachable CSV.

    Accepts a normal share link and rewrites it to the CSV export endpoint.
    """

    kind = "google_sheet"
    supports_writeback = False

    def _url(self) -> str:
        url = self.config.get("url")
        if not url:
            raise ConnectorError("google_sheet source requires a 'url'")

        if "docs.google.com/spreadsheets" in url and "/export" not in url:
            try:
                sheet_id = url.split("/d/")[1].split("/")[0]
            except IndexError as error:
                raise ConnectorError("could not read a sheet id from that URL") from error
            gid = self.config.get("gid", "0")
            return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        return url

    def _load(self) -> list[dict[str, Any]]:
        import httpx

        url = self._url()
        try:
            response = httpx.get(url, timeout=30.0, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ConnectorError(f"could not read sheet: {error}") from error

        if "<html" in response.text[:200].lower():
            raise ConnectorError(
                "the sheet returned HTML, not CSV — publish it to the web "
                "(File > Share > Publish to web) or make it link-readable"
            )
        return _rows_from_text(response.text)

    def test_connection(self) -> ConnectionStatus:
        try:
            rows = self._load()
        except ConnectorError as error:
            return ConnectionStatus(ok=False, detail=str(error))

        return ConnectionStatus(
            ok=True,
            detail=f"Read {len(rows)} row(s) from the sheet",
            sample_columns=list(rows[0].keys()) if rows else [],
            record_estimate=len(rows),
        )

    def fetch(self, cursor: str | None = None, limit: int = 1000) -> FetchResult:
        rows = self._load()
        offset = int(cursor) if cursor and cursor.isdigit() else 0
        window = rows[offset : offset + limit]
        next_offset = offset + len(window)
        return FetchResult(
            records=window,
            cursor=str(next_offset),
            has_more=next_offset < len(rows),
        )


class InlineSource(BaseSource):
    """
    Records supplied directly in the source config.

    Backs webhook intake and one-off uploads, and makes tests hermetic.
    """

    kind = "inline"
    supports_writeback = False

    def _records(self) -> list[dict[str, Any]]:
        return list(self.config.get("records") or [])

    def test_connection(self) -> ConnectionStatus:
        rows = self._records()
        return ConnectionStatus(
            ok=True,
            detail=f"{len(rows)} inline record(s)",
            sample_columns=list(rows[0].keys()) if rows else [],
            record_estimate=len(rows),
        )

    def fetch(self, cursor: str | None = None, limit: int = 1000) -> FetchResult:
        rows = self._records()
        offset = int(cursor) if cursor and cursor.isdigit() else 0
        window = rows[offset : offset + limit]
        next_offset = offset + len(window)
        return FetchResult(records=window, cursor=str(next_offset), has_more=next_offset < len(rows))


register_config_schema(
    "csv",
    {"path": {"type": "string", "required": True, "description": "Path to a CSV file"}},
)
register_config_schema(
    "google_sheet",
    {
        "url": {"type": "string", "required": True, "description": "Sheet share link or CSV URL"},
        "gid": {"type": "string", "required": False, "description": "Tab id, defaults to 0"},
    },
)
register_config_schema(
    "inline",
    {"records": {"type": "array", "required": True, "description": "Raw records"}},
)
