"""
Schema mapping: arbitrary source columns -> the canonical lead schema.

The mapper proposes; a human confirms. Every proposal carries a confidence and
the evidence behind it, and anything below the review threshold is surfaced
rather than applied silently — a wrong mapping produces a confidently wrong
model, which is worse than no model.

Three signals, in order of trust:
  1. exact alias match      - the column name is a known synonym
  2. fuzzy name match       - token-set similarity against names and aliases
  3. value-shape evidence   - does the data look like what the field expects
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from ml.canonical import (
    ALIAS_INDEX,
    BY_NAME,
    CANONICAL_FIELDS,
    CATEGORICAL,
    IDENTITY,
    NUMERIC,
    PLACEHOLDER_VALUES,
    TARGET,
    TIMESTAMP,
    VALUE,
    normalize_key,
)

# Above this we auto-apply; below it the mapping is held for confirmation.
AUTO_ACCEPT = 0.85
REVIEW_FLOOR = 0.45


@dataclass
class FieldProposal:
    source_column: str
    canonical_field: str | None
    confidence: float
    method: str
    evidence: str
    alternatives: list[dict[str, Any]] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return self.canonical_field is not None and self.confidence < AUTO_ACCEPT

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["needs_review"] = self.needs_review
        return payload


@dataclass
class MappingProposal:
    mapping: dict[str, str]
    proposals: list[FieldProposal]
    unmapped_columns: list[str]
    missing_features: list[str]
    confident_count: int
    review_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mapping": self.mapping,
            "proposals": [p.to_dict() for p in self.proposals],
            "unmapped_columns": self.unmapped_columns,
            "missing_features": self.missing_features,
            "confident_count": self.confident_count,
            "review_count": self.review_count,
        }


def _clean_series(series: pd.Series) -> pd.Series:
    values = series.dropna()
    if values.dtype == object:
        lowered = values.astype(str).str.strip().str.lower()
        values = values[~lowered.isin(PLACEHOLDER_VALUES)]
    return values


def _numeric_ratio(series: pd.Series) -> float:
    values = _clean_series(series)
    if values.empty:
        return 0.0
    coerced = pd.to_numeric(values, errors="coerce")
    return float(coerced.notna().mean())


def _looks_binary(series: pd.Series) -> bool:
    values = _clean_series(series)
    if values.empty:
        return False
    distinct = {str(v).strip().lower() for v in values.unique()[:50]}
    binary_sets = (
        {"0", "1"}, {"yes", "no"}, {"true", "false"},
        {"y", "n"}, {"converted", "not converted"},
    )
    return any(distinct <= option for option in binary_sets)


def _shape_score(column_values: pd.Series, canonical_name: str) -> float:
    """
    How well the column's *data* fits the field's expected shape.

    Deliberately weak evidence — it breaks ties, it does not decide mappings.
    """
    spec = BY_NAME[canonical_name]
    values = _clean_series(column_values)
    if values.empty:
        return 0.0

    if spec.kind in (NUMERIC, VALUE):
        return _numeric_ratio(column_values)

    if spec.kind == TARGET:
        return 1.0 if _looks_binary(column_values) else 0.0

    if spec.kind == IDENTITY:
        # Identifiers are near-unique; a repeated value is evidence against.
        return float(values.nunique() / len(values))

    if spec.kind == TIMESTAMP:
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
        return float(parsed.notna().mean())

    if spec.kind == CATEGORICAL:
        distinct = values.nunique()
        # Categorical fields repeat their values, unlike free text or IDs.
        if distinct <= 1:
            return 0.2
        ratio = distinct / len(values)
        return 1.0 if ratio < 0.05 else (0.6 if ratio < 0.2 else 0.15)

    return 0.0


def _name_similarity(normalized_column: str, canonical_name: str) -> float:
    spec = BY_NAME[canonical_name]
    candidates = [normalize_key(spec.name), *(normalize_key(a) for a in spec.aliases)]
    return max(fuzz.token_set_ratio(normalized_column, c) for c in candidates) / 100.0


def propose_mapping(
    frame: pd.DataFrame,
    already_mapped: dict[str, str] | None = None,
) -> MappingProposal:
    """
    Propose a source-column -> canonical-field mapping for a sample frame.

    `already_mapped` entries are treated as confirmed and are never overridden.
    """
    confirmed = dict(already_mapped or {})
    proposals: list[FieldProposal] = []
    mapping: dict[str, str] = {}

    # Score every (column, field) pair first, then assign globally. Assigning
    # column-by-column makes the result depend on column order: an early weak
    # match could claim a field that a later, stronger column needed.
    candidates: dict[str, list[tuple[float, str, str, str]]] = {}

    for column in frame.columns:
        if column in confirmed:
            continue

        normalized = normalize_key(column)
        best_per_field: dict[str, tuple[float, str, str]] = {}

        exact = ALIAS_INDEX.get(normalized)
        if exact is not None:
            shape = _shape_score(frame[column], exact)
            # A known synonym is strong evidence, but data that contradicts the
            # field's shape should still pull the confidence down.
            confidence = 0.95 if shape >= 0.5 else 0.72
            best_per_field[exact] = (confidence, "alias", f"'{column}' is a known synonym")

        for spec in CANONICAL_FIELDS:
            name_score = _name_similarity(normalized, spec.name)
            # Short unrelated words score surprisingly high on token ratios
            # ("amount" vs "country" ~67%), so the bar sits well above chance.
            if name_score < 0.72:
                continue
            shape = _shape_score(frame[column], spec.name)
            confidence = 0.72 * name_score + 0.28 * shape
            existing = best_per_field.get(spec.name)
            if existing is None or confidence > existing[0]:
                best_per_field[spec.name] = (
                    confidence,
                    "fuzzy",
                    f"name similarity {name_score:.0%}, value shape {shape:.0%}",
                )

        ranked = sorted(
            ((score, field, method, evidence) for field, (score, method, evidence) in best_per_field.items()),
            key=lambda item: item[0],
            reverse=True,
        )
        candidates[column] = ranked

    # Confirmed mappings hold their fields unconditionally.
    claimed: set[str] = set(confirmed.values())
    for column, target in confirmed.items():
        mapping[column] = target
        proposals.append(FieldProposal(column, target, 1.0, "confirmed", "Confirmed by a human."))

    # Greedy global assignment, strongest evidence first.
    flat = sorted(
        (
            (score, column, field, method, evidence)
            for column, ranked in candidates.items()
            for score, field, method, evidence in ranked
            if score >= REVIEW_FLOOR
        ),
        key=lambda item: item[0],
        reverse=True,
    )

    assigned: dict[str, tuple[float, str, str, str]] = {}
    for score, column, target, method, evidence in flat:
        if column in assigned or target in claimed:
            continue
        assigned[column] = (score, target, method, evidence)
        claimed.add(target)

    for column, ranked in candidates.items():
        alternatives = [
            {"canonical_field": name, "confidence": round(score, 3)}
            for score, name, _, _ in ranked[:4]
        ]

        if column in assigned:
            score, field, method, evidence = assigned[column]
            mapping[column] = field
            proposals.append(
                FieldProposal(
                    column, field, round(score, 3), method, evidence,
                    [a for a in alternatives if a["canonical_field"] != field][:3],
                )
            )
        elif not ranked:
            proposals.append(
                FieldProposal(column, None, 0.0, "none", "No canonical field resembles this column.")
            )
        elif ranked[0][0] < REVIEW_FLOOR:
            proposals.append(
                FieldProposal(
                    column, None, round(ranked[0][0], 3), "below_threshold", ranked[0][3], alternatives
                )
            )
        else:
            proposals.append(
                FieldProposal(
                    column, None, 0.0, "conflict",
                    "every candidate field was claimed by a stronger match",
                    alternatives,
                )
            )

    mapped_targets = set(mapping.values())
    missing = [f.name for f in CANONICAL_FIELDS if f.is_feature and f.name not in mapped_targets]
    unmapped = [p.source_column for p in proposals if p.canonical_field is None]

    return MappingProposal(
        mapping=mapping,
        proposals=proposals,
        unmapped_columns=unmapped,
        missing_features=missing,
        confident_count=sum(1 for p in proposals if p.canonical_field and not p.needs_review),
        review_count=sum(1 for p in proposals if p.needs_review),
    )


def apply_mapping(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename source columns to canonical names, dropping anything unmapped."""
    usable = {src: dst for src, dst in mapping.items() if src in frame.columns}
    renamed = frame[list(usable)].rename(columns=usable)
    return renamed.loc[:, ~renamed.columns.duplicated()]


def mapping_coverage(mapping: dict[str, str]) -> dict[str, Any]:
    """How much of the canonical feature space a mapping actually fills."""
    from ml.canonical import FEATURE_COLUMNS

    mapped = set(mapping.values()) & set(FEATURE_COLUMNS)
    return {
        "mapped_features": sorted(mapped),
        "missing_features": sorted(set(FEATURE_COLUMNS) - mapped),
        "coverage": round(len(mapped) / len(FEATURE_COLUMNS), 3) if FEATURE_COLUMNS else 0.0,
        "has_target": "converted" in mapping.values(),
    }


def merge_confirmations(
    proposal: MappingProposal, confirmations: dict[str, str | None]
) -> dict[str, str]:
    """
    Apply a human's corrections on top of a proposal.

    A value of None explicitly rejects a proposed mapping for that column.
    """
    merged = dict(proposal.mapping)
    for column, target in confirmations.items():
        if target is None:
            merged.pop(column, None)
        else:
            if target not in BY_NAME:
                raise ValueError(f"Unknown canonical field: {target}")
            merged = {k: v for k, v in merged.items() if v != target}
            merged[column] = target
    return merged


def sample_columns(rows: Iterable[dict[str, Any]], limit: int = 500) -> pd.DataFrame:
    """Build a sample frame from raw source records for mapping inspection."""
    collected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index >= limit:
            break
        collected.append(row)
    return pd.DataFrame(collected)


def describe_proposal(proposal: MappingProposal) -> str:
    """Human-readable summary, used in MCP tool output."""
    lines = [
        f"{proposal.confident_count} column(s) mapped confidently, "
        f"{proposal.review_count} need confirmation."
    ]
    for item in proposal.proposals:
        if item.canonical_field is None:
            continue
        marker = "?" if item.needs_review else "="
        lines.append(
            f"  {item.source_column} {marker}> {item.canonical_field} "
            f"({item.confidence:.0%}, {item.method})"
        )
    if proposal.unmapped_columns:
        lines.append(f"  unmapped: {', '.join(proposal.unmapped_columns[:12])}")
    if proposal.missing_features:
        lines.append(f"  canonical fields with no source: {len(proposal.missing_features)}")
    return "\n".join(lines)


def required_confirmations(proposal: MappingProposal) -> Sequence[FieldProposal]:
    return [p for p in proposal.proposals if p.needs_review]
