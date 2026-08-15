# Roadmap

**Phase 1 — shipped.** A calibrated, leakage-free lead scoring model behind a working API.
**Phase 2 — implemented.** A platform anyone can clone, connect to their own CRM, and
drive through their own AI assistant.

Status at a glance: 82 tests passing · 17 MCP tools · 4 connectors · holdout ROC-AUC 0.896.
The rendered version of this document is served at `/ui/roadmap`.

---

## Phase 1 — shipped

Predicts lead conversion and explains each score field by field. Runs locally: train from
CSV, serve over FastAPI, score through a browser dashboard.

| Model | Accuracy | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| RandomForest | 0.8306 | 0.8915 | 0.8328 | 0.1268 |
| XGBoost | 0.8279 | 0.8951 | 0.8442 | 0.1251 |
| LightGBM | 0.8252 | 0.8915 | 0.8403 | 0.1277 |
| **Hybrid ensemble** | **0.8333** | **0.8957** | 0.8434 | **0.1245** |

Majority-class baseline: 61.5%.

### The finding that shaped the model

An earlier version reported ~92% accuracy by training on columns a rep fills in *after*
contacting the lead — `Tags = "Closed by Horizzon"` converts at 99.4%, `Lead Quality =
"Worst"` at 2.0%. The model was reading its own answer key. Excluding them dropped accuracy
to 83.3% and made it real. **Phase 2 turns this one-off discovery into an automated check
that runs against every customer's data.**

### Delivered

- Leakage-free feature set, enforced by a test
- Calibrated hybrid ensemble weighted by AUC lift (a coin-flip model earns zero weight)
- Self-contained pipelines — training and serving cannot drift apart
- Ablation-based explanations, measured per lead
- Graceful degradation: partial ensemble → heuristic fallback
- 24 tests covering leakage, preprocessing, weighting, ranking, API contracts, fallback

### Limits

- One dataset, one schema — only understands X Education's exact columns
- No accounts, no database; scores vanish when the request ends
- Manual entry only, one lead at a time
- No feedback loop — nothing records whether a prediction was right
- No container, no deployment path, no rate limiting

---

## The gap

"Clone it and connect your CRM" sounds like one feature. It is six problems, and the
hardest is not the MCP server — it is that every customer's data looks different.

1. **Their columns aren't your columns.** HubSpot's `hs_analytics_num_page_views` is not
   `Page Views Per Visit`. Needs a canonical schema and a mapper.
2. **Their answer key isn't yours.** An ed-tech model misprices a B2B SaaS pipeline. Each
   customer needs their own model — but on day one has no labelled history.
3. **Scores must go somewhere.** A score in a browser tab changes nothing.
4. **Real data is sensitive.** Names, emails, employers → retention, encryption, audit.
5. **Models rot.** Without drift monitoring, quality decays silently for months.
6. **Leakage will recur.** Every CRM has its own `Tags` column.

---

## Phase 2 — six tracks, all implemented

Lettered, not numbered: these are workstreams with no single required order.
Each track lists what was built and where it lives.

### Track A — MCP server

Add the server to Claude Desktop or Cursor, then ask "which leads should I call today?"
in plain language against a real pipeline.

- **Tools:** `score_lead`, `score_batch`, `explain_lead`, `list_priority_leads`,
  `connect_source`, `sync_source`, `map_schema`, `push_scores`, `train_tenant_model`,
  `model_metrics`, `simulate_lead`, `record_outcome`
- **Resources:** `leads://priority`, `model://metrics`, `schema://canonical`,
  `source://{id}/mapping`
- **Prompts:** triage today's pipeline, explain a lead to a rep, weekly pipeline review
- **Transports:** stdio for local, streamable HTTP for hosted
- **Safety:** write actions declared non-readonly so clients confirm before firing

**Built:** `mcp_server/server.py` — 17 tools, 3 resources, 3 prompts, stdio + streamable
HTTP. Write tools carry annotations so clients confirm before firing.

### Track B — Universal lead connectors

One protocol, many adapters. Adding a CRM should be one file.

- **Protocol:** `fetch(since)`, `push_score(id, score)`, `schema()`, `test_connection()`
- **Adapters:** HubSpot, Salesforce, Zoho, Pipedrive, Google Sheets, CSV, webhook.
  Ship two, document the interface, let contributors add the rest.
- **Sync engine:** cursor-based incremental pulls, backoff, partial-failure recovery
- **Credentials:** OAuth where supported, encrypted at rest, never in repo or logs
- **Writeback:** push score and band into a CRM field reps already see

**Built:** `connectors/` — the protocol plus CSV, Google Sheets, inline/webhook, and
HubSpot (writeback, cursor paging, 429 backoff). A new adapter is one file.

### Track C — Bring-your-own-data ML

The hardest track, and the one that decides whether this is a product or a demo with
connectors bolted on.

- **Canonical schema** — CRM-agnostic lead: engagement, provenance, profile, intent, outcome
- **Schema mapper** — fuzzy name + type/cardinality heuristics, LLM assist for leftovers,
  human confirmation. Never silently guesses.
- **Automated leakage detector** — flags any feature that predicts the outcome almost
  perfectly, or any category with conversion above 95% / below 5%
- **Cold-start tiers** — see below
- **Per-tenant training** — scheduled retrains, champion/challenger, automatic rollback

**Built:** `ml/canonical.py`, `ml/mapping.py`, `ml/leakage.py`, `ml/training.py`. The
detector independently rediscovers the `Tags` leak. The mapper resolves renamed CRM
columns via global greedy assignment and flags low-confidence guesses for review.

#### Cold-start tiers

| Tier | Labelled leads | What runs | What the user is told |
|---|---|---|---|
| 0 · Generic | 0 | Base model over mapped fields + heuristic | "Not yet trained on your data" |
| 1 · Recalibrated | 1–500 | Base model, isotonic recalibration on their outcomes | "Tuned to your conversion rate" |
| 2 · Tenant model | 500–5,000 | Their own ensemble, base as challenger | "Trained on your pipeline" + metrics |
| 3 · Continuous | 5,000+ | Scheduled retrains, HPO, drift-triggered refresh | Full model card + trends |

### Track D — Production platform

- **Persistence:** Postgres + Alembic — tenants, sources, leads, scores, outcomes, models, audit
- **Background jobs:** Redis + RQ for sync, batch scoring, training
- **Multi-tenancy:** scoped API keys, row-level isolation, per-tenant model registry
- **Deployment:** Dockerfile, docker-compose, GitHub Actions, health/readiness probes
- **Hardening:** rate limiting, structured logging with request IDs, graceful shutdown

**Built:** SQLAlchemy models, API keys, encrypted credentials, rate limiting, request IDs,
liveness/readiness probes, Dockerfile, docker-compose with Postgres, GitHub Actions.

### Track E — Decision intelligence

A probability is not a decision. This is where the product stops describing leads and
starts directing effort.

- **Capacity-aware ranking** — a team with 50 calls a day wants the top 50, not everything
  above 0.5. The fixed threshold is an arbitrary default worth removing.
- **Expected-value scoring** — rank by probability × deal value
- **Time-to-conversion** — survival modelling: "who is going cold?"
- **Next best action** — uplift modelling: who responds to a call vs. an email, rather
  than who converts anyway
- **Score decay** — stale engagement should visibly lose confidence
- **Segment thresholds** — per-channel cut-offs tuned to each segment's base rate

**Built:** `ml/decisions.py` — capacity ranking, expected value, score decay, segment
thresholds, decile lift. Next-best-action ships rule-based and says so in its output;
genuine uplift modelling needs treatment data the system does not collect yet.

### Track F — Trust, monitoring, privacy

- **Drift monitoring** — PSI per feature, with alerting
- **Calibration monitoring** — rolling Brier and reliability curves against outcomes
- **Prediction log** — every score with inputs, model version, explanation
- **Fairness review** — disparity checks across country and city
- **Privacy** — PII encrypted at rest, excluded from logs and features, configurable
  retention, self-hosting as default
- **Model cards** — auto-generated per tenant: training window, features, metrics, limits

**Built:** `ml/monitoring.py` — PSI drift, calibration/ECE, prediction log, audit trail,
PII stripping. **Not built:** fairness review across geography, auto-generated model cards.

---

## Milestones

Numbered because these genuinely are sequential. Estimates assume one developer.

All four milestones are implemented. Effort figures are the original estimates.

| # | Milestone | Effort | Ships |
|---|---|---|---|
| 2.1 | **Connect & Score** | 4–6 wks | Point it at a Sheet or HubSpot export, get a ranked, stored, explainable list |
| 2.2 | **Agent-Native** | 3–4 wks | The README demo becomes a conversation in Claude Desktop against a real pipeline |
| 2.3 | **Learns From You** | 5–7 wks | Stops being one model for everyone; becomes each customer's own model |
| 2.4 | **Ship It** | 3–4 wks | A repo someone clones on a Saturday and runs against their CRM before lunch |

**Total: 15–21 weeks solo.**

---

## Risks

| Risk | Mitigation |
|---|---|
| Customers train on leaky columns | Leakage detector gates every training run; suspicious features quarantined, never used silently |
| Too little data to train | Cold-start tiers with a hard minimum before promotion; base model serves until then, clearly labelled |
| Schema mapping guesses wrong | Confidence scores + human confirmation before first training; mappings versioned and reversible |
| CRM rate limits and API drift | Incremental sync, backoff, per-adapter conformance tests pinned to API versions |
| PII handling | Self-hosted by default, encryption at rest, PII out of logs and features, hard delete |
| Silent model decay | Drift and calibration monitors with alerting; scheduled retrains; automatic rollback |
| Scope outruns one developer | Milestone gates; two adapters, not seven; document the interface for contributors |

---

## Success metrics

- **Adoption:** under 15 minutes from `git clone` to a first score against the user's own CRM
- **Model quality:** tenant models beat the base model on their own holdout at Tier 2+
- **Business lift:** top-decile conversion ≥ 3× base rate
- **Calibration:** Brier < 0.15, ECE < 0.05
- **Performance:** p95 < 200 ms single score; 10,000 leads in under 30 s
- **Reliability:** a failing CRM sync never takes scoring down; degraded modes visible in `/health`

---

## Phase 3 horizon — deliberately out of scope

Named so they don't creep into Phase 2 and stall it.

- Hosted multi-tenant SaaS — billing, org management, SOC 2 posture
- Conversation intelligence — score from call transcripts and email threads
- Marketplace adapters — community connectors with certification
- Automated outreach — draft first-touch messages, human approves every send
- Account-based scoring — roll lead scores up to company level for B2B
- Causal experimentation — built-in holdout groups to measure revenue impact, not just ranking


---

## What is deliberately not built

Named honestly rather than left ambiguous:

- **Salesforce, Zoho, Pipedrive adapters.** The protocol and the reference
  implementation (HubSpot) exist; these three are a file each, left for contributors
  as the plan intended.
- **Background job queue.** Sync and training run inline. A large sync will block its
  request; Redis + RQ is the next step.
- **Uplift modelling.** Next-best-action is rule-based and labelled as such.
- **Fairness review and model cards.** Monitoring covers drift and calibration only.
- **Alembic migrations.** Tables are created from the models; a schema-change path is
  needed before anyone runs this against data they care about.
