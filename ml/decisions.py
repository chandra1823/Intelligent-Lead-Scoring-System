"""
Decision layer — turning a probability into a work queue.

A raw probability is not a decision. Three things change the answer:

  * capacity  - a team with 20 calls a day wants the top 20, not everything
                above an arbitrary 0.5 cutoff
  * value     - a 40% chance at a large deal outranks an 80% chance at a small
                one, so ranking on probability alone leaves money on the table
  * freshness - a score computed on three-week-old engagement should not be
                trusted like one computed this morning
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from ml.model_service import band_for


@dataclass
class RankedLead:
    lead_id: str
    external_id: str
    display_name: str | None
    probability: float
    adjusted_probability: float
    expected_value: float
    priority: float
    band: str
    days_stale: float | None
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decay_factor(days_stale: float | None, half_life: float | None = None) -> float:
    """
    Exponential confidence decay pulling a stale score toward the base rate.

    A lead that has done nothing for two half-lives keeps a quarter of its
    distance above baseline.
    """
    if not settings.score_decay_enabled or days_stale is None or days_stale <= 0:
        return 1.0

    half_life = half_life or settings.score_decay_half_life_days
    if half_life <= 0:
        return 1.0

    return float(0.5 ** (days_stale / half_life))


def apply_decay(
    probability: float,
    days_stale: float | None,
    base_rate: float = 0.38,
    half_life: float | None = None,
) -> float:
    """Shrink a score toward the population base rate as it ages."""
    factor = decay_factor(days_stale, half_life)
    if factor >= 1.0:
        return probability
    return float(base_rate + (probability - base_rate) * factor)


def expected_value(probability: float, deal_value: float | None, default_value: float) -> float:
    value = deal_value if deal_value is not None and deal_value > 0 else default_value
    return float(probability * value)


def _days_since(moment: datetime | None) -> float | None:
    if moment is None:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - moment
    return max(delta.total_seconds() / 86400.0, 0.0)


def rank_leads(
    leads: Iterable[Any],
    capacity: int | None = None,
    strategy: str = "expected_value",
    default_deal_value: float = 1000.0,
    base_rate: float = 0.38,
) -> list[RankedLead]:
    """
    Order leads into a work queue.

    strategy:
      expected_value - probability x deal value (default; respects revenue)
      probability    - highest chance of conversion first
      value          - largest deal first, among leads above the threshold
    """
    ranked: list[RankedLead] = []

    for lead in leads:
        probability = float(getattr(lead, "latest_probability", None) or 0.0)
        stale_days = _days_since(getattr(lead, "scored_at", None))
        adjusted = apply_decay(probability, stale_days, base_rate=base_rate)
        value = getattr(lead, "deal_value", None)
        ev = expected_value(adjusted, value, default_deal_value)

        if strategy == "probability":
            priority = adjusted
        elif strategy == "value":
            priority = float(value or default_deal_value) if adjusted >= settings.decision_threshold else 0.0
        else:
            priority = ev

        ranked.append(
            RankedLead(
                lead_id=getattr(lead, "id", ""),
                external_id=getattr(lead, "external_id", ""),
                display_name=getattr(lead, "display_name", None),
                probability=round(probability, 4),
                adjusted_probability=round(adjusted, 4),
                expected_value=round(ev, 2),
                priority=round(priority, 4),
                band=band_for(adjusted),
                days_stale=round(stale_days, 2) if stale_days is not None else None,
            )
        )

    ranked.sort(key=lambda item: item.priority, reverse=True)
    for index, item in enumerate(ranked, start=1):
        item.rank = index

    if capacity is not None and capacity > 0:
        ranked = ranked[:capacity]

    return ranked


def queue_summary(ranked: list[RankedLead], capacity: int | None = None) -> dict[str, Any]:
    if not ranked:
        return {
            "count": 0,
            "capacity": capacity,
            "total_expected_value": 0.0,
            "bands": {},
            "mean_probability": 0.0,
        }

    bands: dict[str, int] = {}
    for item in ranked:
        bands[item.band] = bands.get(item.band, 0) + 1

    return {
        "count": len(ranked),
        "capacity": capacity,
        "total_expected_value": round(sum(item.expected_value for item in ranked), 2),
        "bands": bands,
        "mean_probability": round(
            sum(item.adjusted_probability for item in ranked) / len(ranked), 4
        ),
        "cutoff_probability": round(min(item.adjusted_probability for item in ranked), 4),
    }


def segment_thresholds(
    outcomes: list[tuple[str, float, int]],
    min_support: int = 50,
) -> dict[str, float]:
    """
    Per-segment cutoffs that maximise F1 within each segment.

    One global 0.5 threshold is wrong whenever segments have different base
    rates — paid search and referrals do not convert at the same rate, so they
    should not share a cutoff.
    """
    grouped: dict[str, list[tuple[float, int]]] = {}
    for segment, probability, actual in outcomes:
        grouped.setdefault(segment, []).append((probability, actual))

    thresholds: dict[str, float] = {}
    for segment, rows in grouped.items():
        if len(rows) < min_support:
            continue

        best_threshold, best_f1 = settings.decision_threshold, -1.0
        for candidate in [i / 20 for i in range(2, 19)]:
            tp = sum(1 for p, a in rows if p >= candidate and a == 1)
            fp = sum(1 for p, a in rows if p >= candidate and a == 0)
            fn = sum(1 for p, a in rows if p < candidate and a == 1)
            if tp == 0:
                continue
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_threshold, best_f1 = candidate, f1

        thresholds[segment] = round(best_threshold, 3)

    return thresholds


def next_best_action(lead_payload: dict[str, Any], probability: float) -> dict[str, str]:
    """
    Suggest the next touch from the lead's own signals.

    Deliberately rule-based and labelled as such. Genuine next-best-action
    needs uplift modelling on historical touch outcomes, which needs treatment
    data this system does not collect yet.
    """
    do_not_contact = str(lead_payload.get("do_not_contact", "")).strip().lower() in {"yes", "true", "1"}
    occupation = str(lead_payload.get("occupation", "") or "").lower()
    last_activity = str(lead_payload.get("last_activity", "") or "").lower()
    time_on_site = float(lead_payload.get("time_on_site_seconds") or 0)

    if do_not_contact:
        return {
            "action": "no_outreach",
            "reason": "This lead has opted out of contact.",
            "basis": "rule",
        }

    if probability >= 0.75 and "working" in occupation:
        return {
            "action": "call_now",
            "reason": "High score and a working professional — call while intent is fresh.",
            "basis": "rule",
        }

    if probability >= 0.6 and "sms" in last_activity:
        return {
            "action": "call_now",
            "reason": "Responded to SMS and scores well; a call is likely to connect.",
            "basis": "rule",
        }

    if probability >= 0.5:
        return {
            "action": "personal_email",
            "reason": "Above threshold but no strong channel signal — start with email.",
            "basis": "rule",
        }

    if time_on_site > 300:
        return {
            "action": "nurture_sequence",
            "reason": "Engaged with the site but scores low; nurture rather than call.",
            "basis": "rule",
        }

    return {
        "action": "monitor",
        "reason": "Not enough signal to justify a rep's time yet.",
        "basis": "rule",
    }


def lift_by_decile(scored: list[tuple[float, int]]) -> list[dict[str, float]]:
    """
    Decile lift table — the number a sales lead actually cares about.

    "The top 10% of your leads convert 3.2x better than average" is the claim
    that justifies changing how a team works.
    """
    rows = sorted(scored, key=lambda item: item[0], reverse=True)
    if not rows:
        return []

    base_rate = sum(actual for _, actual in rows) / len(rows)
    if base_rate <= 0:
        return []

    bucket_size = max(len(rows) // 10, 1)
    table: list[dict[str, float]] = []

    for index in range(min(10, math.ceil(len(rows) / bucket_size))):
        chunk = rows[index * bucket_size : (index + 1) * bucket_size]
        if not chunk:
            break
        rate = sum(actual for _, actual in chunk) / len(chunk)
        table.append(
            {
                "decile": index + 1,
                "leads": len(chunk),
                "conversion_rate": round(rate, 4),
                "lift": round(rate / base_rate, 3),
                "mean_score": round(sum(p for p, _ in chunk) / len(chunk), 4),
            }
        )

    return table
