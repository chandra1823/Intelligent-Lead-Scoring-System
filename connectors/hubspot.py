"""
HubSpot CRM adapter.

Uses the v3 CRM objects API with a private app token. Demonstrates the full
contract including writeback, cursor-based paging, and rate-limit backoff — the
reference implementation for anyone adding Salesforce, Zoho, or Pipedrive.
"""

from __future__ import annotations

import time
from typing import Any

from connectors.base import (
    AuthenticationError,
    BaseSource,
    ConnectionStatus,
    ConnectorError,
    FetchResult,
    register_config_schema,
)

API_ROOT = "https://api.hubapi.com"

# Requested on every contact. HubSpot returns only what you ask for.
DEFAULT_PROPERTIES = [
    "firstname", "lastname", "email", "jobtitle", "company", "industry",
    "city", "country", "lifecyclestage", "hs_lead_status",
    "hs_analytics_num_page_views", "hs_analytics_num_visits",
    "hs_analytics_average_page_views", "hs_analytics_source",
    "hs_analytics_source_data_1", "hs_email_open", "hs_email_click",
    "num_conversion_events", "hs_object_id", "createdate",
    "notes_last_updated", "hs_email_optout",
]

MAX_RETRIES = 5


class HubSpotSource(BaseSource):
    kind = "hubspot"
    supports_writeback = True

    def _token(self) -> str:
        token = self.secrets.get("access_token") or self.config.get("access_token")
        if not token:
            raise AuthenticationError(
                "HubSpot needs a private app token. Create one under "
                "Settings > Integrations > Private Apps with crm.objects.contacts scopes."
            )
        return token

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Issue a request, backing off on 429 and 5xx rather than failing the sync."""
        import httpx

        headers = {"Authorization": f"Bearer {self._token()}", "Content-Type": "application/json"}
        url = f"{API_ROOT}{path}"

        for attempt in range(MAX_RETRIES):
            try:
                response = httpx.request(method, url, headers=headers, timeout=30.0, **kwargs)
            except httpx.HTTPError as error:
                if attempt == MAX_RETRIES - 1:
                    raise ConnectorError(f"HubSpot unreachable: {error}") from error
                time.sleep(2**attempt)
                continue

            if response.status_code in (401, 403):
                raise AuthenticationError("HubSpot rejected the token — check its scopes.")

            if response.status_code == 429 or response.status_code >= 500:
                if attempt == MAX_RETRIES - 1:
                    raise ConnectorError(f"HubSpot returned {response.status_code} after retries")
                # Honour Retry-After when HubSpot sends it.
                wait = float(response.headers.get("Retry-After", 2**attempt))
                time.sleep(min(wait, 30.0))
                continue

            if response.status_code >= 400:
                raise ConnectorError(f"HubSpot {response.status_code}: {response.text[:300]}")

            return response.json() if response.content else {}

        raise ConnectorError("HubSpot request exhausted retries")

    def _properties(self) -> list[str]:
        extra = self.config.get("extra_properties") or []
        return list(dict.fromkeys(DEFAULT_PROPERTIES + list(extra)))

    def test_connection(self) -> ConnectionStatus:
        try:
            payload = self._request(
                "GET",
                "/crm/v3/objects/contacts",
                params={"limit": 1, "properties": ",".join(self._properties())},
            )
        except ConnectorError as error:
            return ConnectionStatus(ok=False, detail=str(error))

        results = payload.get("results", [])
        columns = sorted(results[0].get("properties", {})) if results else []
        return ConnectionStatus(
            ok=True,
            detail="Connected to HubSpot",
            sample_columns=columns,
            record_estimate=payload.get("total"),
        )

    def fetch(self, cursor: str | None = None, limit: int = 100) -> FetchResult:
        params: dict[str, Any] = {
            # HubSpot caps contact pages at 100.
            "limit": min(limit, 100),
            "properties": ",".join(self._properties()),
        }
        if cursor:
            params["after"] = cursor

        payload = self._request("GET", "/crm/v3/objects/contacts", params=params)

        records: list[dict[str, Any]] = []
        for item in payload.get("results", []):
            row = dict(item.get("properties") or {})
            row["hs_object_id"] = item.get("id")
            records.append(row)

        paging = (payload.get("paging") or {}).get("next") or {}
        next_cursor = paging.get("after")

        return FetchResult(records=records, cursor=next_cursor, has_more=bool(next_cursor))

    def push_score(self, external_id: str, probability: float, band: str) -> None:
        """
        Write the score into custom contact properties.

        The properties must exist in the portal first; HubSpot rejects unknown
        property names rather than creating them.
        """
        score_property = self.config.get("score_property", "lead_score_probability")
        band_property = self.config.get("band_property", "lead_score_band")

        self._request(
            "PATCH",
            f"/crm/v3/objects/contacts/{external_id}",
            json={
                "properties": {
                    score_property: round(probability, 4),
                    band_property: band,
                }
            },
        )


register_config_schema(
    "hubspot",
    {
        "access_token": {
            "type": "secret", "required": True, "description": "Private app token",
        },
        "score_property": {
            "type": "string", "required": False,
            "description": "Contact property for the probability",
        },
        "band_property": {
            "type": "string", "required": False,
            "description": "Contact property for the band",
        },
        "extra_properties": {
            "type": "array", "required": False,
            "description": "Additional properties to pull",
        },
    },
)
