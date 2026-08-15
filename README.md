# Intelligent Lead Scoring System

Predicts which sales leads will convert, explains every score field by field, and
ranks a pipeline into a work queue.

Connect a CRM or spreadsheet, and the platform maps it onto a canonical schema,
screens it for target leakage, scores it, and — once you record real outcomes —
trains a model on your own data. Drive it over HTTP, from the dashboard, or
through an AI assistant via the built-in **MCP server**.

```
Sources          Ingest                Core                    Consumers
────────         ──────                ────                    ─────────
HubSpot     ┐    connector layer  ┐    hybrid ensemble    ┐    MCP server
Sheets/CSV  ├──> schema mapper    ├──> training service   ├──> dashboard
webhook     ┘    canonical schema ┘    decision layer     ┘    REST API
                                                               CRM writeback
                        ▲                                            │
                        └──────────── outcomes feed back ────────────┘
```

Full plan and status: [ROADMAP.md](ROADMAP.md) · rendered at `/ui/roadmap`.

---

## Results

Base model, stratified 20% holdout of the bundled 9,240-lead dataset:

| Model | Accuracy | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| RandomForest | 0.8279 | 0.8913 | 0.8332 | 0.1264 |
| XGBoost | 0.8274 | 0.8961 | 0.8455 | 0.1242 |
| LightGBM | 0.8258 | 0.8917 | 0.8401 | 0.1275 |
| **Hybrid ensemble** | **0.8306** | **0.8962** | **0.8447** | **0.1241** |

Majority-class baseline is 61.5%, so the ensemble adds ~22 points.

### Why not 92%

Earlier versions reported ~92% accuracy by training on columns a sales rep fills in
*after* contacting the lead:

| Column value | Conversion rate |
|---|---|
| `Tags = "Closed by Horizzon"` | 99.4% |
| `Tags = "Will revert after reading the email"` | 96.9% |
| `Lead Quality = "Worst"` | 2.0% |

The model was reading its own answer key. Those columns are excluded, and the check
is now automatic — [`ml/leakage.py`](ml/leakage.py) screens every training run, on the
bundled data and on yours. It independently rediscovers the `Tags` leak and leaves
the legitimate behavioural features alone.

---

## Quick start

### Prerequisites

- **Python 3.10+** (3.12 recommended)
- **macOS only:** `brew install libomp` — the OpenMP runtime XGBoost and LightGBM need

### Install and run

```bash
git clone https://github.com/chandra1823/Intelligent-Lead-Scoring-System.git
cd Intelligent-Lead-Scoring-System
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_model.py
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Then open:

| URL | Purpose |
|---|---|
| http://127.0.0.1:8000/ui/dashboard | **Operations console** — queue, sources, training, monitoring |
| http://127.0.0.1:8000/ui | Landing page |
| http://127.0.0.1:8000/ui/scorer | Single-lead scoring form |
| http://127.0.0.1:8000/ui/roadmap | Phase 1 / Phase 2 roadmap |
| http://127.0.0.1:8000/docs | OpenAPI docs |

### With Docker

```bash
docker compose up --build
```

Brings up the API on port 8000 with Postgres behind it. The base model is trained
during the image build.

---

## The console

`/ui/dashboard` is the working surface — a left-rail app with seven views, served
as three static files with no build step.

| View | What it's for |
|---|---|
| **Overview** | Lead counts, conversion rate, queue value, top-decile lift, model tier and health |
| **Lead Queue** | The call list. Ranked, capacity-aware, filterable by band and source; click a row for the detail drawer |
| **Score a Lead** | Try the model on one record, with drivers and a next action |
| **Sources** | Connect a CRM, review and correct the schema mapping, sync, push scores back, inspect sync history |
| **Canonical Schema** | Every field a source can map onto, grouped and described |
| **Model & Training** | Cold-start tier, leakage screen, holdout metrics, version history, retrain |
| **Monitoring** | Drift signals, calibration reliability per score band, decile lift |

The lead drawer explains a score field by field, suggests a next action, records
the outcome, and shows the canonical record behind it. Recording outcomes there
is what moves a workspace up the cold-start tiers.

---

## Connect your own leads

Nothing about the pipeline is specific to the bundled dataset. Point it at a CSV,
a published Google Sheet, or HubSpot:

```bash
# 1. Register the source
curl -s -X POST localhost:8000/v1/sources \
  -H 'Content-Type: application/json' \
  -d '{"name":"My CRM","kind":"csv","config":{"path":"/path/to/leads.csv"}}'

# 2. Propose a mapping from your columns onto the canonical schema
curl -s -X POST localhost:8000/v1/sources/<id>/inspect

# 3. Confirm it (send corrections in "mapping" if anything is wrong)
curl -s -X POST localhost:8000/v1/sources/<id>/mapping \
  -H 'Content-Type: application/json' \
  -d '{"mapping":{},"accept_proposal":true}'

# 4. Pull and score
curl -s -X POST localhost:8000/v1/sources/<id>/sync \
  -H 'Content-Type: application/json' -d '{"limit":5000}'

# 5. Get today's call list
curl -s "localhost:8000/v1/leads/priority?limit=20"
```

Syncing is blocked until the mapping is confirmed — importing thousands of rows
under a guessed mapping is expensive to undo.

### Connectors

| Kind | Reads | Writes scores back |
|---|---|---|
| `csv` | a local file | — |
| `google_sheet` | a published sheet or any CSV URL | — |
| `inline` | records posted directly (webhooks, uploads) | — |
| `hubspot` | Contacts via the v3 API | yes, into custom properties |

Adding one is a single file against the protocol in
[`connectors/base.py`](connectors/base.py): `fetch`, `push_score`, `schema`,
`test_connection`. Register it in [`connectors/registry.py`](connectors/registry.py).

---

## MCP server

Lead triage is a conversation, not a dashboard. The MCP server exposes the engine
as tools an assistant can compose.

Add to `.mcp.json` or your Claude Desktop config (see [.mcp.json.example](.mcp.json.example)):

```json
{
  "mcpServers": {
    "lead-scoring": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/repo", "lead-mcp"]
    }
  }
}
```

Then ask, in plain language:

> *"Connect my HubSpot account."* → `connect_source` → `map_schema` (2 fields need confirmation)
> *"Who should I call today? I have time for 20."* → `list_priority_leads`
> *"Why is that third one ranked so high?"* → `explain_lead`
> *"Push these scores back to HubSpot."* → `push_scores` (asks first)

**17 tools:** `score_lead`, `explain_lead`, `score_batch`, `list_priority_leads`,
`simulate_lead`, `list_connectors`, `list_sources`, `connect_source`, `map_schema`,
`confirm_mapping`, `sync_source`, `push_scores`, `record_outcome`,
`train_tenant_model`, `model_metrics`, `check_model_health`, `get_canonical_schema`

**3 resources:** `leads://priority`, `model://metrics`, `schema://canonical`
**3 prompts:** triage today's pipeline, explain a lead to a rep, weekly review

Tools that write are annotated so clients prompt before firing. Run over HTTP with
`lead-mcp --transport http`.

---

## How it learns from your data

A new workspace has no outcomes of its own, so scoring degrades honestly rather
than refusing to run. The active tier is always visible in `/health`.

| Tier | Labelled leads | What runs |
|---|---|---|
| 0 · Generic | 0 | Shared base model over your mapped fields |
| 1 · Recalibrated | 1–500 | Base model's ranking, probabilities fitted to your conversion rate |
| 2 · Tenant model | 500–5,000 | Your own ensemble, base model as challenger |
| 3 · Continuous | 5,000+ | Scheduled retrains, drift-triggered refresh |

Record outcomes with `POST /v1/leads/outcome`, then `POST /v1/train`. A new model
is only promoted if it beats the incumbent on holdout ROC-AUC by a margin —
otherwise the champion stays.

---

## Beyond a probability

A raw score is not a decision. [`ml/decisions.py`](ml/decisions.py) turns scores
into a queue:

- **Capacity-aware ranking** — a team with 20 calls a day gets the top 20, not
  everything above an arbitrary 0.5 cutoff
- **Expected value** — a 40% shot at a large deal outranks an 80% shot at a small one
- **Score decay** — a score computed on three-week-old engagement loses confidence
  toward the base rate
- **Decile lift** — `GET /v1/monitoring/lift`, the number that justifies changing
  how a team works
- **Next best action** — rule-based today, and labelled as such; genuine uplift
  modelling needs historical treatment data this system does not yet collect

## Staying trustworthy

`GET /v1/monitoring` reports two independent failures:

- **Drift** — population stability index per feature against the distributions
  captured at training time
- **Calibration** — expected calibration error and Brier against real outcomes,
  so silent decay becomes visible

Every score is written to a prediction log with its inputs, model version, and
explanation. Consequential actions land in an audit trail. PII is stripped before
anything is logged or stored as a feature.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | active model, tier, weights |
| `GET /metrics` | holdout metrics and leakage report |
| `GET /schema` | the canonical lead schema |
| `POST /predict` | score one lead |
| `POST /predict/batch` | score up to 5,000 in one pass |
| `POST /explain` | per-field contributions + next action |
| `GET /v1/connectors` | available source types |
| `POST /v1/sources` | connect a source |
| `POST /v1/sources/{id}/inspect` | propose a schema mapping |
| `POST /v1/sources/{id}/mapping` | confirm the mapping |
| `POST /v1/sources/{id}/sync` | pull, store, score |
| `POST /v1/sources/{id}/push-scores` | write scores back |
| `GET /v1/leads/priority` | the ranked work queue |
| `POST /v1/leads/outcome` | record ground truth |
| `POST /v1/train` | train or recalibrate |
| `GET /v1/monitoring` | drift + calibration |
| `GET /v1/monitoring/lift` | decile lift table |

Interactive docs at `/docs`. Probes at `/healthz` and `/readyz`.

### Explanations

Each field's contribution is measured by **ablation** — the lead is re-scored with
that one value reset to the training baseline, and the shift is that field's effect
on this specific lead:

```json
{
  "probability": 0.9864,
  "summary": "This lead scores 98.6% and is prioritised for follow-up. Biggest factors versus a typical lead: Time on site (1200.0) raising it by 5.9 points; ...",
  "contributions": [
    {"feature": "Time on site", "value": 1200.0, "baseline": 248.0, "impact": 0.0587, "direction": "increases"}
  ]
}
```

Real per-lead numbers, not a global importance ranking.

---

## Configuration

Copy [.env.example](.env.example) to `.env`. Every setting is optional; defaults
suit a local install.

| Variable | Default | Notes |
|---|---|---|
| `LEAD_API_DATABASE_URL` | SQLite in `data/` | Point at Postgres for a deployment |
| `LEAD_API_SECRET_KEY` | dev placeholder | **Change this.** Encrypts connector credentials |
| `LEAD_API_REQUIRE_API_KEY` | `false` | Set `true` for anything multi-user |
| `LEAD_API_SCORE_DECAY_HALF_LIFE_DAYS` | `21` | How fast stale scores lose confidence |
| `LEAD_API_TIER2_MIN_LABELS` | `500` | When a tenant model is trained |

With `require_api_key` on, create a workspace via `POST /v1/tenants` — the key is
shown once and only its hash is stored.

---

## Tests

```bash
python -m unittest discover -s test -t . -v
```

The `-t .` matters: it makes unittest treat `test/` as a package so `test/__init__.py`
runs and points the database at a temporary file. Without it the suite refuses to start
rather than writing leads and models into your real database.

82 tests: leakage detection (including a regression test for the original `Tags`
finding), schema mapping, preprocessing, ensemble weighting, explanation ranking,
the full connect → map → sync → rank → train → monitor journey, API contracts,
transaction durability, and the heuristic fallback path.

```bash
ruff check app ml connectors mcp_server scripts test
```

---

## Project layout

```text
app/
  main.py             app factory, CORS, request IDs, probes
  api/routes.py       scoring, explanation, metrics, UI
  api/platform.py     sources, sync, queue, training, monitoring
  api/deps.py         auth, rate limiting, session
  core/               settings, API keys, credential encryption
  db/                 SQLAlchemy models and session management
  services/           lead ingestion, scoring, tenants
ml/
  canonical.py        the canonical lead schema
  mapping.py          source columns -> canonical fields
  leakage.py          target-leakage detection
  features.py         preprocessing pipeline
  model_service.py    ensemble inference + ablation explanations
  training.py         training, cold-start tiers, promotion
  decisions.py        ranking, expected value, decay, lift
  monitoring.py       drift and calibration
  registry.py         model resolution and caching
connectors/           the LeadSource protocol and adapters
mcp_server/           MCP tools, resources, prompts
frontend/             landing page, console (app.*), legacy scorer
docs/roadmap.html     Phase 1 / Phase 2 roadmap
*.ipynb               original EDA / modelling / calibration notebooks
```

## Design notes

- **The canonical schema is the contract.** Models train and score on canonical
  names only, so a model can move between workspaces whose CRMs look nothing alike.
- **The mapper proposes, a human confirms.** Anything below the confidence bar is
  surfaced rather than applied — a wrong mapping produces a confidently wrong model.
- **Leakage is caught, not deleted.** Flagged features are quarantined with an
  explanation and can be overridden by someone who knows better.
- **Graceful degradation.** Missing artifacts fall back to a partial ensemble, then
  to a heuristic. `/health` always says which path is active.
- **Weights by AUC lift.** A model at AUC 0.50 earns zero weight, not the ~1/3 that
  accuracy-normalised weights would hand it.
- **Commits happen at the service boundary**, not in dependency teardown — FastAPI
  runs teardown after the response is sent, which would let a caller act on a
  "success" that had not yet been committed.

## Troubleshooting

**`XGBoostError: libxgboost.dylib could not be loaded`** — `brew install libomp`

**`/health` shows `heuristic_fallback`** — run `python scripts/train_model.py`

**Sync returns `blocked`** — confirm the schema mapping first

**Training returns "Need at least 200 labelled leads"** — expected; the base model
keeps scoring until you have enough outcomes

**Port in use** — `uvicorn app.main:app --port 8001 --reload`

## License

Academic project (final-year demonstration).
