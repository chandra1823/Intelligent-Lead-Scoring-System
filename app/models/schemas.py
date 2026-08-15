"""Request and response models for the HTTP API."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ml.canonical import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Snake-case API surface -> canonical field names. They already match; the map
# exists so the API contract can stay stable if a canonical name is renamed.
FIELD_TO_COLUMN = {name: name for name in NUMERIC_FEATURES + CATEGORICAL_FEATURES}


class LeadFeatures(BaseModel):
    """
    A lead to score.

    Every field is optional except the three behavioural numbers; anything
    omitted is filled from the training-set baseline.
    """

    total_time_spent_on_website: float | None = Field(default=None, ge=0)
    page_views_per_visit: float | None = Field(default=None, ge=0)
    total_visits: float | None = Field(default=None, ge=0)

    time_on_site_seconds: float | None = Field(default=None, ge=0)
    email_opens: float | None = Field(default=None, ge=0)
    email_clicks: float | None = Field(default=None, ge=0)
    form_submissions: float | None = Field(default=None, ge=0)
    days_since_last_activity: float | None = Field(default=None, ge=0)

    channel: str | None = None
    source_detail: str | None = None
    campaign: str | None = None
    origin: str | None = None
    occupation: str | None = None
    seniority: str | None = None
    industry: str | None = None
    company_size: str | None = None
    specialization: str | None = None
    country: str | None = None
    city: str | None = None
    last_activity: str | None = None
    last_notable_activity: str | None = None
    heard_about_us: str | None = None
    motivation: str | None = None
    requested_demo: str | None = None
    wants_free_material: str | None = None
    do_not_contact: str | None = None

    deal_value: float | None = Field(default=None, ge=0)

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "time_on_site_seconds": 1200,
                "page_views_per_visit": 5.0,
                "total_visits": 8,
                "origin": "Lead Add Form",
                "channel": "Reference",
                "last_activity": "SMS Sent",
                "occupation": "Working Professional",
            }
        },
    )

    def to_columns(self) -> dict[str, Any]:
        """Canonical field dict, with pre-2.0 field names accepted as aliases."""
        dumped = self.model_dump(exclude_none=True)

        # Backwards compatibility with the Phase 1 request shape.
        if "total_time_spent_on_website" in dumped:
            dumped.setdefault("time_on_site_seconds", dumped.pop("total_time_spent_on_website"))
        dumped.pop("total_time_spent_on_website", None)

        return {key: value for key, value in dumped.items() if key in FIELD_TO_COLUMN or key == "deal_value"}


class Contribution(BaseModel):
    feature: str
    field: str | None = None
    value: Any
    baseline: Any
    impact: float
    direction: str


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    label: str
    band: str
    model: str
    centralized_output: dict[str, Any]

    model_config = ConfigDict(protected_namespaces=())


class ExplainResponse(BaseModel):
    prediction: int
    probability: float
    label: str
    band: str
    summary: str
    contributions: list[Contribution]
    next_best_action: dict[str, str]
    input_features: dict[str, Any]


class BatchScoreRequest(BaseModel):
    leads: list[LeadFeatures] = Field(..., max_length=5000)


class BatchScoreItem(BaseModel):
    index: int
    probability: float
    prediction: int
    band: str


class BatchScoreResponse(BaseModel):
    scored: int
    results: list[BatchScoreItem]
    model_tier: str

    model_config = ConfigDict(protected_namespaces=())


class MetricsResponse(BaseModel):
    trained_models: list[str]
    weights: dict[str, float]
    metrics: dict[str, dict[str, float]]
    dataset: dict[str, Any]
    model_version: str
    model_tier: str
    leakage_report: dict[str, Any]
    category_options: dict[str, list[str]]

    model_config = ConfigDict(protected_namespaces=())


# ------------------------------------------------------------------- sources


class SourceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    kind: str
    config: dict[str, Any] = Field(default_factory=dict)
    secrets: dict[str, Any] = Field(default_factory=dict)


class SourceResponse(BaseModel):
    id: str
    name: str
    kind: str
    config: dict[str, Any]
    mapping: dict[str, str]
    mapping_confirmed: bool
    supports_writeback: bool
    last_synced_at: str | None = None
    last_sync_status: str | None = None
    last_sync_error: str | None = None


class MappingConfirm(BaseModel):
    """
    Confirm or correct a proposed mapping.

    A null value rejects the proposal for that column.
    """

    mapping: dict[str, str | None]
    accept_proposal: bool = True


class SyncRequest(BaseModel):
    limit: int = Field(default=5000, ge=1, le=50000)
    score: bool = True


class OutcomeRequest(BaseModel):
    lead_id: str
    converted: bool
    deal_value: float | None = Field(default=None, ge=0)


class TrainRequest(BaseModel):
    leakage_overrides: list[str] = Field(default_factory=list)
    force: bool = False


class TenantCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    daily_capacity: int = Field(default=50, ge=1, le=10000)
