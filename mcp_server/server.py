"""
MCP server for the lead scoring platform.

Exposes scoring, triage, connector management, and training as tools an AI
assistant can compose. The point is that lead triage is a conversation — "who
should I call today, and why?" — not a dashboard someone has to learn.

Runs against the same services as the HTTP API, so behaviour cannot diverge.

    uv run lead-mcp                    # stdio, for Claude Desktop / Cursor
    uv run lead-mcp --transport http   # streamable HTTP, for a deployment
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from sqlalchemy import select

from app.core.config import settings
from app.core.security import encrypt_secrets
from app.db.models import Lead, ModelVersion, Source, Tenant
from app.db.session import init_db, session_scope
from app.services import leads as lead_service
from app.services.tenants import get_or_create_default
from connectors.base import ConnectorError
from connectors.registry import available_kinds, build_source
from ml.canonical import describe_schema
from ml.decisions import next_best_action
from ml.mapping import MappingProposal, merge_confirmations
from ml.monitoring import calibration_report, detect_drift, health_rollup
from ml.registry import TIER_GENERIC, TIER_RECALIBRATED, registry, tier_for_label_count
from ml.training import TrainingError, plan_for, recalibrate_base, should_promote, train_model

server = MCPServer(
    name="lead-scoring",
    title="Intelligent Lead Scoring",
    version="2.0.0",
    instructions=(
        "Scores sales leads on their probability of converting, explains every score, "
        "and ranks a pipeline into a work queue.\n\n"
        "Typical flow: connect_source -> map_schema -> confirm_mapping -> sync_source -> "
        "list_priority_leads. Use explain_lead to justify any individual score, and "
        "record_outcome to feed results back so the model improves.\n\n"
        "Scores are probabilities, not certainties. When presenting a queue, lead with "
        "the ranking and the reasons, not the raw decimals."
    ),
)

READ_ONLY = ToolAnnotations(readOnlyHint=True)
WRITES = ToolAnnotations(readOnlyHint=False, destructiveHint=False)
EXTERNAL_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=True)


def _tenant(session) -> Tenant:
    return get_or_create_default(session)


def _source_by_name(session, tenant: Tenant, name_or_id: str) -> Source:
    source = session.scalar(
        select(Source).where(Source.tenant_id == tenant.id, Source.id == name_or_id)
    )
    if source is None:
        source = session.scalar(
            select(Source).where(Source.tenant_id == tenant.id, Source.name == name_or_id)
        )
    if source is None:
        raise ValueError(f"No source named '{name_or_id}'. Call list_sources to see what exists.")
    return source


# ============================================================ scoring tools


@server.tool(
    title="Score a lead",
    description=(
        "Score one lead's probability of converting. Accepts any subset of canonical "
        "fields; anything omitted falls back to the typical value from training data. "
        "Call get_canonical_schema to see the field names."
    ),
    annotations=READ_ONLY,
)
def score_lead(fields: dict[str, Any]) -> dict[str, Any]:
    with session_scope() as session:
        tenant = _tenant(session)
        model, _ = lead_service.active_model_for(session, tenant)
        result = model.predict(fields)

        return {
            "probability": round(result.probability, 4),
            "prediction": result.prediction,
            "band": result.band,
            "label": result.centralized_output["label"],
            "model_tier": model.tier,
            "model_tier_meaning": {
                "generic": "Not yet trained on your data",
                "recalibrated": "Base model tuned to your conversion rate",
                "tenant": "Trained on your pipeline",
                "continuous": "Trained on your pipeline, retrained continuously",
            }.get(model.tier, model.tier),
            "per_model_scores": result.centralized_output["components"],
        }


@server.tool(
    title="Explain a lead's score",
    description=(
        "Explain why a lead scored as it did. Each field's contribution is measured by "
        "re-scoring the lead with that one value reset to the training baseline, so the "
        "numbers are real model behaviour rather than a global importance ranking."
    ),
    annotations=READ_ONLY,
)
def explain_lead(fields: dict[str, Any], top_n: int = 6) -> dict[str, Any]:
    with session_scope() as session:
        tenant = _tenant(session)
        model, _ = lead_service.active_model_for(session, tenant)
        result = model.explain(fields, top_n=top_n)

        return {
            "probability": round(result.probability, 4),
            "band": result.band,
            "summary": model.summarize(result),
            "contributions": result.contributions,
            "next_best_action": next_best_action(fields, result.probability),
        }


@server.tool(
    title="Score many leads",
    description="Score a list of leads in one pass and return them ranked, highest first.",
    annotations=READ_ONLY,
)
def score_batch(leads: list[dict[str, Any]]) -> dict[str, Any]:
    import pandas as pd

    from ml.features import clean_frame
    from ml.model_service import band_for

    if not leads:
        return {"scored": 0, "results": []}

    with session_scope() as session:
        tenant = _tenant(session)
        model, _ = lead_service.active_model_for(session, tenant)
        probabilities = model.predict_frame(clean_frame(pd.DataFrame(leads)))

    ranked = sorted(
        (
            {
                "index": index,
                "probability": round(float(probability), 4),
                "band": band_for(float(probability)),
            }
            for index, probability in enumerate(probabilities)
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )
    return {"scored": len(ranked), "results": ranked}


@server.tool(
    title="List priority leads",
    description=(
        "The work queue: stored leads ranked for follow-up. Ranking accounts for deal "
        "value and for score staleness, and respects the team's daily capacity rather "
        "than returning everything above a fixed threshold. "
        "Strategy is one of expected_value (default), probability, or value."
    ),
    annotations=READ_ONLY,
)
def list_priority_leads(
    limit: int | None = None,
    strategy: str = "expected_value",
    band: str | None = None,
) -> dict[str, Any]:
    with session_scope() as session:
        tenant = _tenant(session)
        queue = lead_service.priority_queue(
            session, tenant, limit=limit, strategy=strategy, band=band
        )

    if not queue["leads"]:
        queue["hint"] = (
            "No scored leads yet. Connect a source with connect_source, then sync_source."
        )
    return queue


@server.tool(
    title="Simulate a change to a lead",
    description=(
        "What-if analysis: re-score a lead with some fields changed and report how much "
        "the probability moves. Useful for questions like 'would this lead be worth "
        "calling if they booked a demo?'"
    ),
    annotations=READ_ONLY,
)
def simulate_lead(fields: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    with session_scope() as session:
        tenant = _tenant(session)
        model, _ = lead_service.active_model_for(session, tenant)

        before = model.predict(fields).probability
        after = model.predict({**fields, **changes}).probability

    return {
        "before": round(before, 4),
        "after": round(after, 4),
        "delta": round(after - before, 4),
        "delta_points": round((after - before) * 100, 2),
        "changes": changes,
        "verdict": "improves" if after > before else ("worsens" if after < before else "no change"),
    }


# ========================================================== connector tools


@server.tool(
    title="List available connectors",
    description="Show which lead source types can be connected, and what each one needs.",
    annotations=READ_ONLY,
)
def list_connectors() -> dict[str, Any]:
    return {"connectors": available_kinds()}


@server.tool(
    title="List connected sources",
    description="Show the lead sources already connected to this workspace.",
    annotations=READ_ONLY,
)
def list_sources() -> dict[str, Any]:
    with session_scope() as session:
        tenant = _tenant(session)
        sources = session.scalars(select(Source).where(Source.tenant_id == tenant.id))
        return {
            "sources": [
                {
                    "id": source.id,
                    "name": source.name,
                    "kind": source.kind,
                    "mapping_confirmed": source.mapping_confirmed,
                    "last_synced_at": source.last_synced_at.isoformat() if source.last_synced_at else None,
                    "last_sync_status": source.last_sync_status,
                }
                for source in sources
            ]
        }


@server.tool(
    title="Connect a lead source",
    description=(
        "Register a CRM, spreadsheet, or CSV as a lead source. Credentials are encrypted "
        "before storage. After connecting, call map_schema to work out how the source's "
        "columns line up with the canonical schema."
    ),
    annotations=WRITES,
)
def connect_source(
    name: str,
    kind: str,
    config: dict[str, Any] | None = None,
    secrets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        adapter = build_source(kind, config or {}, secrets or {})
        check = adapter.test_connection()
    except ConnectorError as error:
        return {"status": "error", "detail": str(error)}

    if not check.ok:
        return {"status": "error", "detail": check.detail}

    with session_scope() as session:
        tenant = _tenant(session)
        existing = session.scalar(
            select(Source).where(Source.tenant_id == tenant.id, Source.name == name)
        )
        if existing is not None:
            return {"status": "error", "detail": f"A source named '{name}' already exists."}

        source = Source(
            tenant_id=tenant.id,
            name=name,
            kind=kind,
            config=config or {},
            secrets=encrypt_secrets(secrets),
        )
        session.add(source)
        session.flush()
        source_id = source.id
        lead_service.audit(session, tenant.id, "source.created", source_id, kind=kind)

    return {
        "status": "connected",
        "source_id": source_id,
        "name": name,
        "detail": check.detail,
        "columns_found": check.sample_columns[:40],
        "next_step": "Call map_schema to propose how these columns map to the canonical schema.",
    }


@server.tool(
    title="Propose a schema mapping",
    description=(
        "Sample a source and propose how its columns map onto the canonical lead schema. "
        "Returns confidence per column. Anything below the confidence bar is flagged for "
        "confirmation rather than applied — a wrong mapping produces a confidently wrong "
        "model. Review the proposal, then call confirm_mapping."
    ),
    annotations=READ_ONLY,
)
def map_schema(source: str) -> dict[str, Any]:
    with session_scope() as session:
        tenant = _tenant(session)
        record = _source_by_name(session, tenant, source)

        try:
            result = lead_service.inspect_source_schema(record)
        except ConnectorError as error:
            return {"status": "error", "detail": str(error)}

        if result.get("proposal"):
            record.pending_mapping = result["proposal"]
            proposal = result["proposal"]
            needs_review = [
                p for p in proposal["proposals"] if p.get("needs_review")
            ]
            return {
                "status": "ok",
                "source_id": record.id,
                "summary": result["detail"],
                "mapping": proposal["mapping"],
                "needs_confirmation": needs_review,
                "unmapped_columns": proposal["unmapped_columns"],
                "next_step": (
                    "Call confirm_mapping to accept, optionally correcting any column. "
                    "Syncing is blocked until the mapping is confirmed."
                ),
            }

        return result


@server.tool(
    title="Confirm a schema mapping",
    description=(
        "Accept the proposed mapping for a source, with optional corrections. "
        "Pass corrections as {source_column: canonical_field}; use null to reject a "
        "proposed mapping for a column. Syncing is blocked until this is done."
    ),
    annotations=WRITES,
)
def confirm_mapping(source: str, corrections: dict[str, str | None] | None = None) -> dict[str, Any]:
    from ml.mapping import mapping_coverage

    with session_scope() as session:
        tenant = _tenant(session)
        record = _source_by_name(session, tenant, source)

        base = dict((record.pending_mapping or {}).get("mapping") or {})
        if not base and not corrections:
            return {
                "status": "error",
                "detail": "No pending proposal. Call map_schema first.",
            }

        proposal = MappingProposal(
            mapping=base, proposals=[], unmapped_columns=[], missing_features=[],
            confident_count=0, review_count=0,
        )

        try:
            merged = merge_confirmations(proposal, corrections or {})
        except ValueError as error:
            return {"status": "error", "detail": str(error)}

        if not merged:
            return {"status": "error", "detail": "The mapping is empty; map at least one column."}

        record.mapping = merged
        record.mapping_confirmed = True
        record.pending_mapping = None
        lead_service.audit(session, tenant.id, "mapping.confirmed", record.id, fields=len(merged))

        coverage = mapping_coverage(merged)

    return {
        "status": "confirmed",
        "mapping": merged,
        "coverage": coverage,
        "next_step": "Call sync_source to pull and score the leads.",
    }


@server.tool(
    title="Sync a lead source",
    description=(
        "Pull leads from a connected source, translate them to the canonical schema, "
        "store them, and score them. Resumes from the last sync position."
    ),
    annotations=WRITES,
)
def sync_source(source: str, limit: int = 5000) -> dict[str, Any]:
    with session_scope() as session:
        tenant = _tenant(session)
        record = _source_by_name(session, tenant, source)
        return lead_service.sync_source(session, tenant, record, limit=limit)


@server.tool(
    title="Push scores back to the source",
    description=(
        "Write current scores into the source CRM so reps see them where they already "
        "work. This modifies records in the external system."
    ),
    annotations=EXTERNAL_WRITE,
)
def push_scores(source: str, limit: int = 100) -> dict[str, Any]:
    from app.core.security import decrypt_secrets

    with session_scope() as session:
        tenant = _tenant(session)
        record = _source_by_name(session, tenant, source)
        adapter = build_source(record.kind, record.config, decrypt_secrets(record.secrets))

        if not adapter.supports_writeback:
            return {"status": "error", "detail": f"'{record.kind}' sources are read-only."}

        scored = list(
            session.scalars(
                select(Lead)
                .where(Lead.source_id == record.id, Lead.latest_probability.is_not(None))
                .order_by(Lead.latest_probability.desc())
                .limit(limit)
            )
        )

        pushed, failures = 0, []
        for lead in scored:
            try:
                adapter.push_score(lead.external_id, lead.latest_probability, lead.latest_band or "")
                pushed += 1
            except ConnectorError as error:
                failures.append({"external_id": lead.external_id, "error": str(error)})

        lead_service.audit(session, tenant.id, "scores.pushed", record.id, pushed=pushed)

    return {"status": "ok", "pushed": pushed, "attempted": len(scored), "failures": failures[:5]}


# ========================================================== learning tools


@server.tool(
    title="Record a lead outcome",
    description=(
        "Record whether a lead actually converted. This is the feedback loop — without "
        "outcomes the model cannot improve beyond its generic starting point."
    ),
    annotations=WRITES,
)
def record_outcome(lead_id: str, converted: bool, deal_value: float | None = None) -> dict[str, Any]:
    with session_scope() as session:
        tenant = _tenant(session)
        try:
            result = lead_service.record_outcome(session, tenant, lead_id, converted, deal_value)
        except LookupError as error:
            return {"status": "error", "detail": str(error)}

        result["training_plan"] = plan_for(result["labelled_total"])
        result["status"] = "recorded"
        return result


@server.tool(
    title="Train a model on this workspace's data",
    description=(
        "Train or recalibrate a model from recorded outcomes. The approach is chosen "
        "automatically from how much labelled data exists. Candidate features are "
        "screened for target leakage first, and the new model is only promoted if it "
        "beats the current one."
    ),
    annotations=WRITES,
)
def train_tenant_model(leakage_overrides: list[str] | None = None, force: bool = False) -> dict[str, Any]:
    with session_scope() as session:
        tenant = _tenant(session)
        frame = lead_service.labelled_frame(session, tenant)
        labelled = len(frame)
        tier = tier_for_label_count(labelled)

        if tier == TIER_GENERIC and not force:
            return {"status": "skipped", **plan_for(labelled)}

        pending = settings.tenant_artifact_dir(tenant.id, "pending")
        try:
            if tier == TIER_RECALIBRATED:
                result = recalibrate_base(frame, pending)
            else:
                result = train_model(
                    frame, pending, tier=tier, leakage_overrides=leakage_overrides or []
                )
        except TrainingError as error:
            return {"status": "error", "detail": str(error)}

        champion = session.scalar(
            select(ModelVersion)
            .where(ModelVersion.tenant_id == tenant.id, ModelVersion.is_active.is_(True))
            .order_by(ModelVersion.created_at.desc())
        )
        promote, reason = should_promote(result.metrics, champion.metrics if champion else None)

        final_dir = settings.tenant_artifact_dir(tenant.id, result.version)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        if pending.exists():
            if final_dir.exists():
                import shutil

                shutil.rmtree(final_dir)
            pending.rename(final_dir)

        session.add(
            ModelVersion(
                tenant_id=tenant.id,
                version=result.version,
                tier=result.tier,
                artifact_dir=str(final_dir),
                metrics=result.metrics,
                weights=result.weights,
                leakage_report=result.leakage,
                training_rows=result.training_rows,
                is_active=promote,
                promoted_reason=reason,
            )
        )
        if promote:
            if champion is not None:
                champion.is_active = False
            registry.invalidate()

        return {
            "status": "trained",
            "promoted": promote,
            "reason": reason,
            "tier": result.tier,
            "training_rows": result.training_rows,
            "metrics": result.metrics.get("hybrid_ensemble", {}),
            "leakage": result.leakage,
            "notes": result.notes,
        }


@server.tool(
    title="Get model metrics",
    description="Holdout performance of the model currently scoring this workspace.",
    annotations=READ_ONLY,
)
def model_metrics() -> dict[str, Any]:
    with session_scope() as session:
        tenant = _tenant(session)
        model, record = lead_service.active_model_for(session, tenant)

        return {
            "model_version": model.version,
            "model_tier": model.tier,
            "source": "tenant" if record else "base",
            "ensemble": model.model_name,
            "weights": model.weights,
            "metrics": model.metrics,
            "dataset": model.dataset_info,
            "leakage_report": model.leakage_report,
        }


@server.tool(
    title="Check model health",
    description=(
        "Check whether the model is still trustworthy: has incoming data drifted from "
        "the training distribution, and do predicted probabilities still match observed "
        "conversion rates."
    ),
    annotations=READ_ONLY,
)
def check_model_health() -> dict[str, Any]:
    with session_scope() as session:
        tenant = _tenant(session)
        model, _ = lead_service.active_model_for(session, tenant)

        live = lead_service.recent_feature_frame(session, tenant)
        drift = (
            detect_drift(model.feature_statistics, live)
            if model.feature_statistics and not live.empty
            else {"status": "insufficient_data", "detail": "Not enough scored leads yet.", "signals": []}
        )

        probabilities, outcomes = lead_service.matched_predictions(session, tenant)
        calibration = (
            calibration_report(probabilities, outcomes)
            if probabilities
            else {"status": "insufficient_data", "detail": "No leads with both a score and an outcome."}
        )

        return {"drift": drift, "calibration": calibration, "health": health_rollup(drift, calibration)}


@server.tool(
    title="Get the canonical schema",
    description=(
        "The field names every lead source is mapped onto. Use this to work out what to "
        "pass to score_lead, or to correct a schema mapping."
    ),
    annotations=READ_ONLY,
)
def get_canonical_schema() -> dict[str, Any]:
    return {"fields": describe_schema()}


# ================================================================ resources


@server.resource(
    "leads://priority",
    name="Priority lead queue",
    description="The current ranked follow-up queue.",
    mime_type="application/json",
)
def priority_resource() -> str:
    with session_scope() as session:
        tenant = _tenant(session)
        return json.dumps(lead_service.priority_queue(session, tenant), indent=2, default=str)


@server.resource(
    "model://metrics",
    name="Active model metrics",
    description="Holdout metrics for the model currently in use.",
    mime_type="application/json",
)
def metrics_resource() -> str:
    with session_scope() as session:
        tenant = _tenant(session)
        model, _ = lead_service.active_model_for(session, tenant)
        return json.dumps(
            {
                "version": model.version,
                "tier": model.tier,
                "metrics": model.metrics,
                "weights": model.weights,
            },
            indent=2,
            default=str,
        )


@server.resource(
    "schema://canonical",
    name="Canonical lead schema",
    description="Every canonical field, its type, and its aliases.",
    mime_type="application/json",
)
def schema_resource() -> str:
    return json.dumps({"fields": describe_schema()}, indent=2)


# ================================================================== prompts


@server.prompt(
    title="Triage today's pipeline",
    description="Review the priority queue and recommend who to contact today.",
)
def triage_pipeline(capacity: str = "20") -> str:
    return (
        f"Use list_priority_leads with limit={capacity} to pull today's queue.\n\n"
        "Then, for the top 5, call explain_lead to understand what is driving each score.\n\n"
        "Present the result as a short call list. For each lead give: the rank, the score "
        "as a percentage, the single most important reason it scored that way, and the "
        "suggested next action. Do not list raw field dumps. End with one sentence on "
        "what the queue as a whole looks like today."
    )


@server.prompt(
    title="Explain a lead to a rep",
    description="Turn a lead's score into something a salesperson can act on.",
)
def explain_to_rep(lead_id: str) -> str:
    return (
        f"Call explain_lead for lead {lead_id}.\n\n"
        "Write two short paragraphs for a salesperson with no ML background. The first "
        "says how promising the lead is and why, in plain language, naming the specific "
        "behaviours that mattered. The second says what to do next and what to open with. "
        "Avoid jargon: no 'features', no 'model', no probabilities beyond a single "
        "percentage."
    )


@server.prompt(
    title="Weekly pipeline review",
    description="Assess pipeline health, model health, and what to fix.",
)
def weekly_review() -> str:
    return (
        "Build a weekly review:\n"
        "1. list_priority_leads to see the queue shape.\n"
        "2. check_model_health for drift and calibration.\n"
        "3. model_metrics for current performance.\n\n"
        "Report: how many leads are worth working, whether the model is still trustworthy, "
        "and the single most useful action to take this week. If health shows an alert, "
        "lead with that. Be specific and brief — this is read in a stand-up."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Lead scoring MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse"],
        default="stdio",
        help="stdio for local desktop clients, http for a deployment",
    )
    args = parser.parse_args()

    init_db()

    transport = "streamable-http" if args.transport == "http" else args.transport
    server.run(transport=transport)


if __name__ == "__main__":
    main()
