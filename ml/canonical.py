"""
The canonical lead schema.

Every lead source — a CRM, a spreadsheet, a webhook — is mapped onto these
fields before it reaches the model. Models train and score on canonical names
only, so a model trained on one customer's pipeline can be evaluated against
another's without either side knowing the other's column names.

Aliases exist so the schema mapper can propose a mapping automatically; they
are matched case- and separator-insensitively.
"""

from __future__ import annotations

from dataclasses import dataclass, field

NUMERIC = "numeric"
CATEGORICAL = "categorical"
IDENTITY = "identity"
TIMESTAMP = "timestamp"
TARGET = "target"
VALUE = "value"

ENGAGEMENT = "engagement"
PROVENANCE = "provenance"
PROFILE = "profile"
INTENT = "intent"
OUTCOME = "outcome"
META = "meta"


@dataclass(frozen=True)
class CanonicalField:
    name: str
    kind: str
    group: str
    description: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    is_feature: bool = True


CANONICAL_FIELDS: tuple[CanonicalField, ...] = (
    # ---------------------------------------------------------------- identity
    CanonicalField(
        "external_id", IDENTITY, META,
        "The lead's primary key in the source system.",
        ("id", "lead id", "record id", "prospect id", "contact id", "vid", "hs object id"),
        is_feature=False,
    ),
    CanonicalField(
        "display_name", IDENTITY, META,
        "Human-readable label for the lead. Never used as a feature.",
        ("name", "full name", "contact name", "first name", "lead name"),
        is_feature=False,
    ),
    # -------------------------------------------------------------- engagement
    CanonicalField(
        "time_on_site_seconds", NUMERIC, ENGAGEMENT,
        "Total seconds the lead has spent on the website.",
        ("total time spent on website", "time on site", "session duration",
         "hs analytics average time per session", "total time", "time spent"),
    ),
    CanonicalField(
        "total_visits", NUMERIC, ENGAGEMENT,
        "Number of distinct sessions.",
        ("totalvisits", "visits", "sessions", "number of sessions",
         "hs analytics num visits", "website visits"),
    ),
    CanonicalField(
        "page_views_per_visit", NUMERIC, ENGAGEMENT,
        "Average pages viewed per session.",
        ("page views per visit", "pages per session", "avg page views",
         "hs analytics num page views", "pageviews"),
    ),
    CanonicalField(
        "email_opens", NUMERIC, ENGAGEMENT,
        "Marketing emails opened.",
        ("emails opened", "opens", "hs email open", "email open count"),
    ),
    CanonicalField(
        "email_clicks", NUMERIC, ENGAGEMENT,
        "Links clicked in marketing emails.",
        ("emails clicked", "clicks", "hs email click", "email click count"),
    ),
    CanonicalField(
        "form_submissions", NUMERIC, ENGAGEMENT,
        "Forms submitted on the site.",
        ("num form submissions", "forms submitted", "num conversion events"),
    ),
    CanonicalField(
        "days_since_last_activity", NUMERIC, ENGAGEMENT,
        "Days since the lead last did anything. Drives score decay.",
        ("days since last activity", "recency", "days inactive", "last activity days"),
    ),
    # -------------------------------------------------------------- provenance
    CanonicalField(
        "channel", CATEGORICAL, PROVENANCE,
        "Acquisition channel: organic, paid, referral, direct, social, email.",
        ("lead source", "source", "original source", "utm medium",
         "hs analytics source", "traffic source"),
    ),
    CanonicalField(
        "source_detail", CATEGORICAL, PROVENANCE,
        "Specific origin within the channel, e.g. Google or a partner site.",
        ("source detail", "referrer", "utm source", "original source drill down 1",
         "lead source detail"),
    ),
    CanonicalField(
        "campaign", CATEGORICAL, PROVENANCE,
        "Marketing campaign the lead arrived through.",
        ("utm campaign", "campaign name", "hs analytics first url", "ad campaign"),
    ),
    CanonicalField(
        "origin", CATEGORICAL, PROVENANCE,
        "How the record was created: form, API, import, manual.",
        ("lead origin", "record source", "created by", "origin type"),
    ),
    # ----------------------------------------------------------------- profile
    CanonicalField(
        "occupation", CATEGORICAL, PROFILE,
        "Employment status or role.",
        ("what is your current occupation", "occupation", "job title", "jobtitle",
         "current occupation", "title", "role"),
    ),
    CanonicalField(
        "seniority", CATEGORICAL, PROFILE,
        "Seniority band: individual contributor, manager, director, executive.",
        ("seniority", "job level", "management level"),
    ),
    CanonicalField(
        "industry", CATEGORICAL, PROFILE,
        "Industry of the lead's employer.",
        ("industry", "vertical", "sector", "company industry"),
    ),
    CanonicalField(
        "company_size", CATEGORICAL, PROFILE,
        "Employee-count band of the lead's employer.",
        ("company size", "numberofemployees", "employee count", "employees"),
    ),
    CanonicalField(
        "specialization", CATEGORICAL, PROFILE,
        "Subject area or product interest.",
        ("specialization", "area of interest", "product interest", "department"),
    ),
    CanonicalField(
        "country", CATEGORICAL, PROFILE,
        "Country of the lead.",
        ("country", "country region", "nation"),
    ),
    CanonicalField(
        "city", CATEGORICAL, PROFILE,
        "City of the lead.",
        ("city", "town", "locality"),
    ),
    # ------------------------------------------------------------------ intent
    CanonicalField(
        "last_activity", CATEGORICAL, INTENT,
        "Most recent tracked interaction.",
        ("last activity", "last engagement", "recent activity",
         "hs last engagement type", "latest activity"),
    ),
    CanonicalField(
        "last_notable_activity", CATEGORICAL, INTENT,
        "Most recent high-signal interaction.",
        ("last notable activity", "notable activity", "last meaningful activity"),
    ),
    CanonicalField(
        "heard_about_us", CATEGORICAL, INTENT,
        "Self-reported discovery channel.",
        ("how did you hear about x education", "how did you hear about us",
         "referral source", "discovery channel"),
    ),
    CanonicalField(
        "motivation", CATEGORICAL, INTENT,
        "Self-reported reason for interest.",
        ("what matters most to you in choosing a course", "motivation",
         "reason for interest", "goal"),
    ),
    CanonicalField(
        "requested_demo", CATEGORICAL, INTENT,
        "Whether the lead asked for a demo, trial, or callback.",
        ("requested demo", "demo requested", "trial requested", "wants callback"),
    ),
    CanonicalField(
        "wants_free_material", CATEGORICAL, INTENT,
        "Opted in to a lead magnet or free resource.",
        ("a free copy of mastering the interview", "downloaded ebook",
         "wants free material", "content download"),
    ),
    CanonicalField(
        "do_not_contact", CATEGORICAL, INTENT,
        "Opted out of email or calls.",
        ("do not email", "do not call", "unsubscribed", "opted out",
         "hs email optout", "do not contact"),
    ),
    # ----------------------------------------------------------------- outcome
    CanonicalField(
        "converted", TARGET, OUTCOME,
        "Training label: 1 if the lead converted, 0 otherwise.",
        ("converted", "is converted", "won", "closed won", "outcome", "did convert"),
        is_feature=False,
    ),
    CanonicalField(
        "deal_value", VALUE, OUTCOME,
        "Monetary value of the opportunity. Powers expected-value ranking.",
        ("deal value", "amount", "revenue", "contract value", "acv", "mrr"),
        is_feature=False,
    ),
    CanonicalField(
        "converted_at", TIMESTAMP, OUTCOME,
        "When the conversion happened.",
        ("converted at", "close date", "closedate", "won date"),
        is_feature=False,
    ),
)

BY_NAME: dict[str, CanonicalField] = {f.name: f for f in CANONICAL_FIELDS}

FEATURE_FIELDS: tuple[CanonicalField, ...] = tuple(f for f in CANONICAL_FIELDS if f.is_feature)
NUMERIC_FEATURES: list[str] = [f.name for f in FEATURE_FIELDS if f.kind == NUMERIC]
CATEGORICAL_FEATURES: list[str] = [f.name for f in FEATURE_FIELDS if f.kind == CATEGORICAL]
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN = "converted"
VALUE_COLUMN = "deal_value"
ID_COLUMN = "external_id"

# Unanswered dropdowns arrive as placeholder strings in most CRMs.
PLACEHOLDER_VALUES = frozenset({"select", "", "none", "n/a", "na", "null", "unknown", "-", "--"})
UNKNOWN_TOKEN = "Unknown"


def normalize_key(raw: str) -> str:
    """Lowercase and strip separators so 'Total_Visits' matches 'total visits'."""
    cleaned = []
    for character in str(raw).strip().lower():
        cleaned.append(character if character.isalnum() else " ")
    return " ".join("".join(cleaned).split())


ALIAS_INDEX: dict[str, str] = {}
for _f in CANONICAL_FIELDS:
    ALIAS_INDEX[normalize_key(_f.name)] = _f.name
    for _alias in _f.aliases:
        ALIAS_INDEX.setdefault(normalize_key(_alias), _f.name)


def describe_schema() -> list[dict[str, object]]:
    """Serialisable description of the schema, exposed over the API and MCP."""
    return [
        {
            "name": f.name,
            "kind": f.kind,
            "group": f.group,
            "description": f.description,
            "is_feature": f.is_feature,
            "aliases": list(f.aliases),
        }
        for f in CANONICAL_FIELDS
    ]


def lookup(name: str) -> CanonicalField | None:
    return BY_NAME.get(name)
