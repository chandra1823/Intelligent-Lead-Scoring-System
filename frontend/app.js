/* ==========================================================================
   LeadIQ console

   A small hash-routed app over the platform API. No build step on purpose —
   the backend serves these three files directly, so a clone runs with nothing
   but Python installed.
   ========================================================================== */

const state = {
  health: null,
  metrics: null,
  stats: null,
  sources: [],
  connectors: [],
  monitoring: null,
  queue: null,
  schema: null,
  queueOpts: { limit: 25, strategy: "expected_value", band: "", source_id: "" },
};

/* ─────────────────────────────── api ─────────────────────────────── */

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });

  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }

  if (!response.ok) {
    const message = data?.detail
      ? typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail)
      : `${response.status} ${response.statusText}`;
    throw new Error(message);
  }
  return data;
}

/* ─────────────────────────────── helpers ─────────────────────────────── */

const el = (id) => document.getElementById(id);
const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );

const pct = (value, digits = 1) =>
  value === null || value === undefined ? "—" : `${(value * 100).toFixed(digits)}%`;

const money = (value) =>
  value === null || value === undefined
    ? "—"
    : `$${Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

const num = (value, digits = 3) =>
  value === null || value === undefined || Number.isNaN(value) ? "—" : Number(value).toFixed(digits);

const titleCase = (value) =>
  String(value ?? "").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

const relTime = (iso) => {
  if (!iso) return "never";
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000;
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
};

function toast(message, kind = "") {
  const node = document.createElement("div");
  node.className = `toast ${kind}`;
  node.textContent = message;
  el("toasts").appendChild(node);
  setTimeout(() => node.remove(), 4600);
}

function bandChip(band) {
  return `<span class="chip chip-${esc(band)}">${esc(titleCase(band))}</span>`;
}

function scoreCell(probability, band) {
  const width = Math.max(Math.min(probability * 100, 100), 1.5);
  return `<div class="score-cell">
    <div class="score-track"><div class="score-fill b-${esc(band)}" style="width:${width}%"></div></div>
    <span class="score-num">${pct(probability, 1)}</span>
  </div>`;
}

function statusChip(status) {
  const map = { ok: "chip-ok", warn: "chip-warm", alert: "chip-bad", insufficient_data: "chip-muted" };
  return `<span class="chip ${map[status] || "chip-muted"}">${esc(titleCase(status))}</span>`;
}

const TIER_COPY = {
  generic: "Not yet trained on your data — using the shared base model",
  recalibrated: "Base model ranking, probabilities fitted to your conversion rate",
  tenant: "Trained on your own pipeline",
  continuous: "Trained on your pipeline, retrained continuously",
};

/* ─────────────────────────────── shell ─────────────────────────────── */

async function loadShell() {
  try {
    const [health, metrics] = await Promise.all([api("/health"), api("/metrics")]);
    state.health = health;
    state.metrics = metrics;

    const auc = metrics.metrics?.hybrid_ensemble?.roc_auc;
    el("modelChip").innerHTML =
      `<span>${esc(health.model)}</span> · <span>${esc(metrics.model_tier)}</span>` +
      (auc ? ` · <span>AUC ${num(auc)}</span>` : "");
  } catch (error) {
    el("modelChip").innerHTML = `<span style="color:var(--bad)">backend unavailable</span>`;
    toast(`Could not reach the API: ${error.message}`, "is-bad");
  }

  try {
    const [stats, sources] = await Promise.all([api("/v1/leads/stats"), api("/v1/sources")]);
    state.stats = stats;
    state.sources = sources;
    el("navQueueCount").textContent = stats.scored ? stats.scored.toLocaleString() : "";
    el("navSourceCount").textContent = sources.length ? sources.length : "";
  } catch {
    /* the shell still renders without counts */
  }

  try {
    const monitoring = await api("/v1/monitoring");
    state.monitoring = monitoring;
    el("navHealthDot").className = `nav-dot is-${monitoring.health.status}`;
  } catch {
    /* monitoring is optional for the shell */
  }
}

/* ─────────────────────────────── views ─────────────────────────────── */

const views = {};

/* ---------- overview ---------- */

views.overview = {
  title: "Overview",
  subtitle: "Pipeline health at a glance",
  async render() {
    const [stats, queue] = await Promise.all([
      api("/v1/leads/stats"),
      api("/v1/leads/priority?limit=8").catch(() => ({ leads: [], summary: {} })),
    ]);
    state.stats = stats;

    let lift = null;
    try {
      lift = await api("/v1/monitoring/lift");
    } catch {
      /* needs outcomes */
    }

    const monitoring = state.monitoring;
    const metrics = state.metrics;
    const hybrid = metrics?.metrics?.hybrid_ensemble;
    const plan = stats.training_plan || {};

    const tiles = `
      <div class="grid g-4">
        <div class="stat">
          <span class="stat-label">Leads</span>
          <span class="stat-value">${stats.total.toLocaleString()}</span>
          <span class="stat-note">${stats.scored.toLocaleString()} scored</span>
        </div>
        <div class="stat">
          <span class="stat-label">Conversion rate</span>
          <span class="stat-value">${stats.conversion_rate === null ? "—" : pct(stats.conversion_rate)}</span>
          <span class="stat-note">${stats.converted.toLocaleString()} of ${stats.labelled.toLocaleString()} known</span>
        </div>
        <div class="stat">
          <span class="stat-label">Queue value</span>
          <span class="stat-value">${money(queue.summary?.total_expected_value)}</span>
          <span class="stat-note">top ${queue.leads?.length || 0} by expected value</span>
        </div>
        <div class="stat">
          <span class="stat-label">Top-decile lift</span>
          <span class="stat-value">${lift?.top_decile_lift ? `${lift.top_decile_lift}×` : "—"}</span>
          <span class="stat-note">${lift?.top_decile_lift ? "vs average conversion" : "needs recorded outcomes"}</span>
        </div>
      </div>`;

    const modelCard = `
      <div class="card">
        <div class="card-head">
          <div><h3>Active model</h3><p>${esc(TIER_COPY[metrics?.model_tier] || "")}</p></div>
          <a class="btn btn-sm" href="#/model">Manage</a>
        </div>
        <div class="card-body">
          <dl class="kv">
            <dt>Tier</dt><dd><span class="chip chip-info">${esc(titleCase(metrics?.model_tier || "unknown"))}</span></dd>
            <dt>Ensemble</dt><dd>${esc(state.health?.model || "—")}</dd>
            <dt>Holdout ROC-AUC</dt><dd>${num(hybrid?.roc_auc)}</dd>
            <dt>Holdout accuracy</dt><dd>${hybrid ? pct(hybrid.accuracy) : "—"}</dd>
            <dt>Brier score</dt><dd>${num(hybrid?.brier)}</dd>
          </dl>
          ${
            plan.action && plan.action !== "keep_base_model"
              ? `<div class="callout" style="margin-top:14px">${esc(plan.detail)}
                   <a href="#/model" style="color:var(--accent-ink);font-weight:600"> Train now →</a></div>`
              : ""
          }
        </div>
      </div>`;

    const healthCard = monitoring
      ? `<div class="card">
           <div class="card-head">
             <div><h3>Model health</h3><p>Drift and calibration</p></div>
             <a class="btn btn-sm" href="#/monitoring">Details</a>
           </div>
           <div class="card-body stack">
             <div class="row-between"><span>Overall</span>${statusChip(monitoring.health.status)}</div>
             <div class="row-between"><span>Input drift</span>${statusChip(monitoring.drift.status)}</div>
             <div class="row-between"><span>Calibration</span>${statusChip(monitoring.calibration.status)}</div>
             <div class="callout ${monitoring.health.status === "alert" ? "is-bad" : monitoring.health.status === "warn" ? "is-warn" : "is-ok"}">
               ${esc(monitoring.health.recommendation)}
             </div>
           </div>
         </div>`
      : "";

    const queueRows = (queue.leads || [])
      .map(
        (lead) => `
        <tr class="clickable" data-lead='${esc(JSON.stringify(lead))}'>
          <td class="rank-cell">${lead.rank}</td>
          <td><strong>${esc(lead.display_name || lead.external_id)}</strong></td>
          <td>${scoreCell(lead.adjusted_probability, lead.band)}</td>
          <td class="num">${money(lead.expected_value)}</td>
        </tr>`
      )
      .join("");

    const queueCard = `
      <div class="card">
        <div class="card-head">
          <div><h3>Today's call list</h3><p>Ranked by expected value</p></div>
          <a class="btn btn-sm" href="#/queue">Open queue</a>
        </div>
        <div class="card-body tight">
          ${
            queueRows
              ? `<div class="table-wrap"><table class="data">
                   <thead><tr><th></th><th>Lead</th><th>Score</th><th class="num">Exp. value</th></tr></thead>
                   <tbody>${queueRows}</tbody></table></div>`
              : `<div class="empty"><h3>No scored leads yet</h3>
                   <p>Connect a source and sync it to build a queue.</p>
                   <a class="btn btn-primary" href="#/sources">Connect a source</a></div>`
          }
        </div>
      </div>`;

    return `<div class="stack">
      ${tiles}
      <div class="grid g-2">${modelCard}${healthCard}</div>
      ${queueCard}
    </div>`;
  },
  after(root) {
    bindLeadRows(root);
  },
};

/* ---------- lead queue ---------- */

views.queue = {
  title: "Lead Queue",
  subtitle: "Who to contact, in order",
  async render() {
    const options = state.queueOpts;
    const params = new URLSearchParams({ strategy: options.strategy });
    if (options.limit) params.set("limit", options.limit);
    if (options.band) params.set("band", options.band);
    if (options.source_id) params.set("source_id", options.source_id);

    const queue = await api(`/v1/leads/priority?${params}`);
    state.queue = queue;

    const summary = queue.summary || {};
    const bands = summary.bands || {};

    const sourceOptions = state.sources
      .map((s) => `<option value="${esc(s.id)}"${options.source_id === s.id ? " selected" : ""}>${esc(s.name)}</option>`)
      .join("");

    const rows = (queue.leads || [])
      .map((lead) => {
        const stale = lead.days_stale === null ? "—" : `${Math.round(lead.days_stale)}d`;
        const decayed = lead.adjusted_probability < lead.probability - 0.005;
        return `
        <tr class="clickable" data-lead='${esc(JSON.stringify(lead))}'>
          <td class="rank-cell">${lead.rank}</td>
          <td>
            <strong>${esc(lead.display_name || lead.external_id)}</strong>
            ${lead.display_name ? `<div style="font-size:11.5px;color:var(--ink-4)">${esc(lead.external_id)}</div>` : ""}
          </td>
          <td>${scoreCell(lead.adjusted_probability, lead.band)}</td>
          <td>${bandChip(lead.band)}</td>
          <td class="num">${money(lead.expected_value)}</td>
          <td class="num" title="${decayed ? `Raw score ${pct(lead.probability)}, decayed for staleness` : ""}">
            ${stale}${decayed ? ' <span style="color:var(--warn)">↓</span>' : ""}
          </td>
          <td class="num">
            <button class="btn btn-sm" data-explain>Explain</button>
          </td>
        </tr>`;
      })
      .join("");

    return `<div class="stack">
      <div class="grid g-4">
        <div class="stat"><span class="stat-label">In queue</span><span class="stat-value">${summary.count ?? 0}</span><span class="stat-note">capacity ${summary.capacity ?? "—"}</span></div>
        <div class="stat"><span class="stat-label">Pipeline value</span><span class="stat-value">${money(summary.total_expected_value)}</span><span class="stat-note">sum of expected value</span></div>
        <div class="stat"><span class="stat-label">Mean score</span><span class="stat-value">${pct(summary.mean_probability)}</span><span class="stat-note">cutoff ${pct(summary.cutoff_probability)}</span></div>
        <div class="stat"><span class="stat-label">Mix</span><span class="stat-value" style="font-size:15px;padding-top:6px">
          ${["hot", "warm", "cool", "cold"].filter((b) => bands[b]).map((b) => `${bandChip(b)} ${bands[b]}`).join(" ") || "—"}
        </span></div>
      </div>

      <div class="card">
        <div class="toolbar">
          <div class="field"><label for="qStrategy">Rank by</label>
            <select id="qStrategy">
              <option value="expected_value"${options.strategy === "expected_value" ? " selected" : ""}>Expected value</option>
              <option value="probability"${options.strategy === "probability" ? " selected" : ""}>Probability</option>
              <option value="value"${options.strategy === "value" ? " selected" : ""}>Deal size</option>
            </select>
          </div>
          <div class="field"><label for="qBand">Band</label>
            <select id="qBand">
              <option value="">All</option>
              ${["hot", "warm", "cool", "cold"].map((b) => `<option value="${b}"${options.band === b ? " selected" : ""}>${titleCase(b)}</option>`).join("")}
            </select>
          </div>
          <div class="field"><label for="qSource">Source</label>
            <select id="qSource"><option value="">All</option>${sourceOptions}</select>
          </div>
          <div class="field"><label for="qLimit">Capacity</label>
            <input type="number" id="qLimit" min="1" max="1000" value="${options.limit}" />
          </div>
          <div class="spacer"></div>
          <span class="chip chip-muted chip-plain">${esc(titleCase(queue.model_tier))} model</span>
        </div>
        <div class="card-body tight">
          ${
            rows
              ? `<div class="table-wrap"><table class="data">
                  <thead><tr>
                    <th></th><th>Lead</th><th>Score</th><th>Band</th>
                    <th class="num">Exp. value</th><th class="num">Age</th><th class="num"></th>
                  </tr></thead>
                  <tbody>${rows}</tbody></table></div>`
              : `<div class="empty"><h3>Nothing to call</h3>
                   <p>No scored leads match these filters. Connect a source, sync it, or widen the filters.</p>
                   <a class="btn btn-primary" href="#/sources">Manage sources</a></div>`
          }
        </div>
      </div>
    </div>`;
  },
  after(root) {
    bindLeadRows(root);

    const apply = (key, value) => {
      state.queueOpts[key] = value;
      route();
    };
    root.querySelector("#qStrategy")?.addEventListener("change", (e) => apply("strategy", e.target.value));
    root.querySelector("#qBand")?.addEventListener("change", (e) => apply("band", e.target.value));
    root.querySelector("#qSource")?.addEventListener("change", (e) => apply("source_id", e.target.value));
    root.querySelector("#qLimit")?.addEventListener("change", (e) =>
      apply("limit", Math.max(1, Math.min(1000, Number(e.target.value) || 25)))
    );
  },
};

/* ---------- score a lead ---------- */

views.score = {
  title: "Score a Lead",
  subtitle: "Try the model on a single record",
  async render() {
    const metrics = state.metrics || (await api("/metrics"));
    const options = metrics.category_options || {};

    const selects = Object.entries(options)
      .filter(([, values]) => values.length)
      .map(
        ([field, values]) => `
        <div class="field">
          <label for="f_${esc(field)}">${esc(titleCase(field))}</label>
          <select id="f_${esc(field)}" data-field="${esc(field)}">
            <option value="">— typical value —</option>
            ${values.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join("")}
          </select>
        </div>`
      )
      .join("");

    return `<div class="grid g-2" style="align-items:start">
      <div class="card">
        <div class="card-head"><div><h3>Lead details</h3><p>Only the behaviour metrics are required</p></div></div>
        <div class="card-body stack">
          <div class="grid g-3">
            <div class="field"><label for="f_time">Time on site (seconds)</label><input type="number" id="f_time" min="0" placeholder="1200" /></div>
            <div class="field"><label for="f_views">Page views per visit</label><input type="number" id="f_views" min="0" step="0.1" placeholder="5" /></div>
            <div class="field"><label for="f_visits">Total visits</label><input type="number" id="f_visits" min="0" placeholder="8" /></div>
          </div>
          <div class="grid g-3">${selects}</div>
          <div class="field" style="max-width:220px"><label for="f_value">Deal value (optional)</label><input type="number" id="f_value" min="0" placeholder="5000" /></div>
          <div class="row">
            <button class="btn btn-primary" id="scoreBtn">Score lead</button>
            <button class="btn" id="sampleBtn">Load sample</button>
          </div>
        </div>
      </div>

      <div class="card" id="scoreResult">
        <div class="card-head"><div><h3>Result</h3><p>Score, drivers, and next action</p></div></div>
        <div class="card-body"><div class="empty" style="padding:34px 12px"><p>Enter a lead and press <strong>Score lead</strong>.</p></div></div>
      </div>
    </div>`;
  },
  after(root) {
    const collect = () => {
      const payload = {};
      const time = root.querySelector("#f_time").value;
      const views_ = root.querySelector("#f_views").value;
      const visits = root.querySelector("#f_visits").value;
      const value = root.querySelector("#f_value").value;
      if (time !== "") payload.time_on_site_seconds = Number(time);
      if (views_ !== "") payload.page_views_per_visit = Number(views_);
      if (visits !== "") payload.total_visits = Number(visits);
      if (value !== "") payload.deal_value = Number(value);
      root.querySelectorAll("select[data-field]").forEach((s) => {
        if (s.value) payload[s.dataset.field] = s.value;
      });
      return payload;
    };

    root.querySelector("#sampleBtn").addEventListener("click", () => {
      root.querySelector("#f_time").value = 1200;
      root.querySelector("#f_views").value = 5;
      root.querySelector("#f_visits").value = 8;
      root.querySelector("#f_value").value = 5000;
      const occupation = root.querySelector('select[data-field="occupation"]');
      if (occupation) {
        const match = Array.from(occupation.options).find((o) => /working/i.test(o.value));
        if (match) occupation.value = match.value;
      }
    });

    root.querySelector("#scoreBtn").addEventListener("click", async (event) => {
      const button = event.currentTarget;
      const payload = collect();
      if (!Object.keys(payload).length) return toast("Enter at least one value", "is-bad");

      button.disabled = true;
      const panel = root.querySelector("#scoreResult .card-body");
      panel.innerHTML = `<div class="loading"><span class="spinner-sm"></span></div>`;

      try {
        const data = await api("/explain", { method: "POST", body: payload });
        panel.innerHTML = renderExplanation(data);
      } catch (error) {
        panel.innerHTML = `<div class="callout is-bad">${esc(error.message)}</div>`;
      } finally {
        button.disabled = false;
      }
    });
  },
};

/* ---------- sources ---------- */

views.sources = {
  title: "Sources",
  subtitle: "Connected CRMs and files",
  async render() {
    const [sources, connectors] = await Promise.all([api("/v1/sources"), api("/v1/connectors")]);
    state.sources = sources;
    state.connectors = connectors.connectors;

    const cards = sources
      .map((source) => {
        const status = source.last_sync_status;
        const chip = !source.mapping_confirmed
          ? `<span class="chip chip-warm">Mapping needed</span>`
          : status === "success"
          ? `<span class="chip chip-ok">Synced ${esc(relTime(source.last_synced_at))}</span>`
          : status === "failed"
          ? `<span class="chip chip-bad">Sync failed</span>`
          : `<span class="chip chip-muted">Never synced</span>`;

        return `
        <div class="card" data-source="${esc(source.id)}">
          <div class="card-head">
            <div>
              <h3>${esc(source.name)}</h3>
              <p><code class="inline">${esc(source.kind)}</code> ${source.supports_writeback ? "· writeback supported" : "· read-only"}</p>
            </div>
            ${chip}
          </div>
          <div class="card-body stack">
            ${
              source.last_sync_error
                ? `<div class="callout is-bad">${esc(source.last_sync_error)}</div>`
                : ""
            }
            ${
              source.mapping_confirmed
                ? `<div class="section-label">Field mapping (${Object.keys(source.mapping).length})</div>
                   <div style="font-size:12px;color:var(--ink-2);font-family:var(--mono);line-height:1.8">
                     ${Object.entries(source.mapping)
                       .slice(0, 6)
                       .map(([k, v]) => `${esc(k)} <span style="color:var(--ink-4)">→</span> ${esc(v)}`)
                       .join("<br>")}
                     ${Object.keys(source.mapping).length > 6 ? `<br><span style="color:var(--ink-4)">+${Object.keys(source.mapping).length - 6} more</span>` : ""}
                   </div>`
                : `<div class="callout is-warn">This source cannot sync until its columns are mapped onto the canonical schema.</div>`
            }
            <div class="row">
              <button class="btn btn-sm" data-act="inspect">${source.mapping_confirmed ? "Review mapping" : "Map columns"}</button>
              <button class="btn btn-sm btn-primary" data-act="sync" ${source.mapping_confirmed ? "" : "disabled"}>Sync now</button>
              ${source.supports_writeback ? `<button class="btn btn-sm" data-act="push">Push scores</button>` : ""}
              <button class="btn btn-sm" data-act="runs">History</button>
              <div class="spacer"></div>
              <button class="btn btn-sm btn-danger" data-act="delete">Remove</button>
            </div>
            <div data-panel hidden></div>
          </div>
        </div>`;
      })
      .join("");

    const kinds = state.connectors
      .map((c) => `<option value="${esc(c.kind)}">${esc(c.kind)}${c.supports_writeback ? " (writeback)" : ""}</option>`)
      .join("");

    return `<div class="stack">
      <div class="card">
        <div class="card-head"><div><h3>Connect a source</h3><p>Point the platform at a CRM, a sheet, or a file</p></div></div>
        <div class="card-body stack">
          <div class="grid g-3">
            <div class="field"><label for="newName">Name</label><input type="text" id="newName" placeholder="Acme CRM" /></div>
            <div class="field"><label for="newKind">Type</label><select id="newKind">${kinds}</select></div>
          </div>
          <div id="kindFields" class="grid g-3"></div>
          <div class="row"><button class="btn btn-primary" id="connectBtn">Connect</button>
            <span class="hint" style="color:var(--ink-4);font-size:12px">Credentials are encrypted before storage.</span></div>
        </div>
      </div>
      ${sources.length ? `<div class="grid g-2" style="align-items:start">${cards}</div>` : `
        <div class="card"><div class="empty">
          <h3>No sources connected</h3>
          <p>Connect a CSV, a published Google Sheet, or HubSpot to start scoring real leads.</p>
        </div></div>`}
    </div>`;
  },
  after(root) {
    const kindSelect = root.querySelector("#newKind");
    const fields = root.querySelector("#kindFields");

    const renderKindFields = () => {
      const connector = state.connectors.find((c) => c.kind === kindSelect.value);
      const schema = connector?.config_schema || {};
      fields.innerHTML =
        Object.entries(schema)
          .map(
            ([key, spec]) => `
          <div class="field">
            <label for="cfg_${esc(key)}">${esc(titleCase(key))}${spec.required ? " *" : ""}</label>
            <input type="${spec.type === "secret" ? "password" : "text"}" id="cfg_${esc(key)}"
                   data-key="${esc(key)}" data-secret="${spec.type === "secret"}"
                   placeholder="${esc(spec.description || "")}" />
          </div>`
          )
          .join("") || `<p class="hint" style="color:var(--ink-4)">No configuration needed.</p>`;
    };
    kindSelect.addEventListener("change", renderKindFields);
    renderKindFields();

    root.querySelector("#connectBtn").addEventListener("click", async (event) => {
      const name = root.querySelector("#newName").value.trim();
      if (!name) return toast("Give the source a name", "is-bad");

      const config = {};
      const secrets = {};
      fields.querySelectorAll("input[data-key]").forEach((input) => {
        if (!input.value) return;
        (input.dataset.secret === "true" ? secrets : config)[input.dataset.key] = input.value;
      });

      event.currentTarget.disabled = true;
      try {
        await api("/v1/sources", {
          method: "POST",
          body: { name, kind: kindSelect.value, config, secrets },
        });
        toast(`Connected "${name}"`, "is-ok");
        await loadShell();
        route();
      } catch (error) {
        toast(error.message, "is-bad");
        event.currentTarget.disabled = false;
      }
    });

    root.querySelectorAll("[data-source]").forEach((card) => {
      const id = card.dataset.source;
      const panel = card.querySelector("[data-panel]");

      const show = (html) => {
        panel.hidden = false;
        panel.innerHTML = html;
      };

      card.querySelector('[data-act="inspect"]')?.addEventListener("click", async () => {
        show(`<div class="loading"><span class="spinner-sm"></span></div>`);
        try {
          const result = await api(`/v1/sources/${id}/inspect`, { method: "POST" });
          if (!result.proposal) return show(`<div class="callout is-warn">${esc(result.detail)}</div>`);
          show(renderMapping(id, result));
          bindMapping(panel, id);
        } catch (error) {
          show(`<div class="callout is-bad">${esc(error.message)}</div>`);
        }
      });

      card.querySelector('[data-act="sync"]')?.addEventListener("click", async (event) => {
        event.currentTarget.disabled = true;
        show(`<div class="loading"><span class="spinner-sm"></span></div>`);
        try {
          const result = await api(`/v1/sources/${id}/sync`, { method: "POST", body: { limit: 5000 } });
          if (result.status === "success") {
            toast(`Synced ${result.fetched} records · ${result.created} new · ${result.scored} scored`, "is-ok");
            await loadShell();
            route();
          } else {
            show(`<div class="callout is-warn">${esc(result.detail || result.status)}</div>`);
          }
        } catch (error) {
          show(`<div class="callout is-bad">${esc(error.message)}</div>`);
        } finally {
          event.currentTarget.disabled = false;
        }
      });

      card.querySelector('[data-act="push"]')?.addEventListener("click", async () => {
        show(`<div class="loading"><span class="spinner-sm"></span></div>`);
        try {
          const result = await api(`/v1/sources/${id}/push-scores`, { method: "POST" });
          show(`<div class="callout is-ok">Pushed ${result.pushed} of ${result.attempted} scores back to the source.</div>`);
        } catch (error) {
          show(`<div class="callout is-bad">${esc(error.message)}</div>`);
        }
      });

      card.querySelector('[data-act="runs"]')?.addEventListener("click", async () => {
        show(`<div class="loading"><span class="spinner-sm"></span></div>`);
        try {
          const { runs } = await api(`/v1/sources/${id}/runs`);
          show(
            runs.length
              ? `<div class="table-wrap"><table class="data">
                  <thead><tr><th>Status</th><th class="num">Fetched</th><th class="num">New</th><th class="num">Scored</th><th>When</th></tr></thead>
                  <tbody>${runs
                    .map(
                      (r) => `<tr>
                        <td>${r.status === "success" ? '<span class="chip chip-ok">Success</span>' : r.status === "failed" ? '<span class="chip chip-bad">Failed</span>' : `<span class="chip chip-muted">${esc(titleCase(r.status))}</span>`}</td>
                        <td class="num">${r.fetched}</td><td class="num">${r.created}</td><td class="num">${r.scored}</td>
                        <td>${esc(relTime(r.started_at))}</td></tr>`
                    )
                    .join("")}</tbody></table></div>`
              : `<p class="hint" style="color:var(--ink-4)">No sync runs yet.</p>`
          );
        } catch (error) {
          show(`<div class="callout is-bad">${esc(error.message)}</div>`);
        }
      });

      card.querySelector('[data-act="delete"]')?.addEventListener("click", async () => {
        show(`<div class="callout is-warn">
            Remove this source and every lead imported from it? This cannot be undone.
            <div class="row" style="margin-top:10px">
              <button class="btn btn-sm btn-danger" data-confirm>Remove source</button>
              <button class="btn btn-sm" data-cancel>Cancel</button>
            </div></div>`);
        panel.querySelector("[data-cancel]").addEventListener("click", () => (panel.hidden = true));
        panel.querySelector("[data-confirm]").addEventListener("click", async () => {
          try {
            await api(`/v1/sources/${id}`, { method: "DELETE" });
            toast("Source removed", "is-ok");
            await loadShell();
            route();
          } catch (error) {
            toast(error.message, "is-bad");
          }
        });
      });
    });
  },
};

function renderMapping(sourceId, result) {
  const proposal = result.proposal;
  const canonical = (state.schema?.fields || []).map((f) => f.name);

  const rows = proposal.proposals
    .map((item) => {
      const confidence = item.canonical_field
        ? `${Math.round(item.confidence * 100)}%`
        : "—";
      const tone = !item.canonical_field
        ? "chip-muted"
        : item.needs_review
        ? "chip-warm"
        : "chip-ok";

      const options = ["<option value=\"\">— not mapped —</option>"]
        .concat(
          canonical.map(
            (name) =>
              `<option value="${esc(name)}"${item.canonical_field === name ? " selected" : ""}>${esc(name)}</option>`
          )
        )
        .join("");

      return `<div class="map-row">
        <div class="map-src" title="${esc(item.source_column)}">${esc(item.source_column)}</div>
        <div class="map-arrow">→</div>
        <select data-col="${esc(item.source_column)}">${options}</select>
        <div class="map-conf"><span class="chip ${tone}">${confidence}</span></div>
      </div>`;
    })
    .join("");

  const needsReview = proposal.proposals.filter((p) => p.needs_review);

  return `<div class="stack" style="margin-top:12px">
    <div class="callout ${needsReview.length ? "is-warn" : "is-ok"}">
      ${esc(result.detail)}
      ${needsReview.length ? " Confirm or correct the highlighted rows before syncing — a wrong mapping produces a confidently wrong model." : ""}
    </div>
    <div>${rows}</div>
    <div class="row">
      <button class="btn btn-primary btn-sm" data-save-mapping>Confirm mapping</button>
      <span class="hint" style="color:var(--ink-4);font-size:12px">${proposal.unmapped_columns.length} column(s) will be ignored</span>
    </div>
  </div>`;
}

function bindMapping(panel, sourceId) {
  panel.querySelector("[data-save-mapping]").addEventListener("click", async (event) => {
    const mapping = {};
    panel.querySelectorAll("select[data-col]").forEach((select) => {
      mapping[select.dataset.col] = select.value || null;
    });

    event.currentTarget.disabled = true;
    try {
      const result = await api(`/v1/sources/${sourceId}/mapping`, {
        method: "POST",
        body: { mapping, accept_proposal: false },
      });
      toast(`Mapping confirmed — ${Math.round(result.coverage.coverage * 100)}% feature coverage`, "is-ok");
      await loadShell();
      route();
    } catch (error) {
      toast(error.message, "is-bad");
      event.currentTarget.disabled = false;
    }
  });
}

/* ---------- canonical schema ---------- */

views.schema = {
  title: "Canonical Schema",
  subtitle: "The field names every source is mapped onto",
  async render() {
    const schema = state.schema || (await api("/schema"));
    state.schema = schema;

    const groups = {};
    schema.fields.forEach((field) => {
      (groups[field.group] ||= []).push(field);
    });

    const cards = Object.entries(groups)
      .map(
        ([group, fields]) => `
      <div class="card">
        <div class="card-head"><div><h3>${esc(titleCase(group))}</h3><p>${fields.length} field(s)</p></div></div>
        <div class="card-body tight">
          <div class="table-wrap"><table class="data">
            <thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead>
            <tbody>${fields
              .map(
                (f) => `<tr>
                  <td><code class="inline">${esc(f.name)}</code>${f.is_feature ? "" : ' <span class="chip chip-muted chip-plain">not a feature</span>'}</td>
                  <td><span class="chip chip-muted chip-plain">${esc(f.kind)}</span></td>
                  <td style="color:var(--ink-2)">${esc(f.description)}</td>
                </tr>`
              )
              .join("")}</tbody>
          </table></div>
        </div>
      </div>`
      )
      .join("");

    return `<div class="stack">
      <div class="callout">Sources are mapped onto these fields before scoring, so a model trained on one pipeline
        can score another whose CRM looks nothing alike. Aliases let the mapper propose a mapping automatically.</div>
      ${cards}
    </div>`;
  },
};

/* ---------- model & training ---------- */

views.model = {
  title: "Model & Training",
  subtitle: "Performance, versions, and retraining",
  async render() {
    const [metrics, models, stats] = await Promise.all([
      api("/metrics"),
      api("/v1/models"),
      api("/v1/leads/stats"),
    ]);
    state.metrics = metrics;

    const rows = Object.entries(metrics.metrics || {})
      .map(([name, scores]) => {
        const isHybrid = name === "hybrid_ensemble";
        return `<tr${isHybrid ? ' style="background:var(--accent-soft);font-weight:600"' : ""}>
          <td>${esc(isHybrid ? "Hybrid ensemble" : titleCase(name))}</td>
          <td class="num">${pct(scores.accuracy, 2)}</td>
          <td class="num">${num(scores.roc_auc, 4)}</td>
          <td class="num">${num(scores.pr_auc, 4)}</td>
          <td class="num">${num(scores.brier, 4)}</td>
          <td class="num">${metrics.weights[name] ? pct(metrics.weights[name], 0) : "—"}</td>
        </tr>`;
      })
      .join("");

    const leakage = metrics.leakage_report || {};
    const findings = leakage.findings || [];

    const versionRows = (models.models || [])
      .map(
        (m) => `<tr>
        <td><code class="inline">${esc(m.version)}</code></td>
        <td><span class="chip chip-info">${esc(titleCase(m.tier))}</span></td>
        <td class="num">${m.training_rows.toLocaleString()}</td>
        <td class="num">${num(m.roc_auc, 4)}</td>
        <td>${m.is_active ? '<span class="chip chip-ok">Active</span>' : '<span class="chip chip-muted">Superseded</span>'}</td>
        <td style="color:var(--ink-3);font-size:12px">${esc(m.promoted_reason || "")}</td>
      </tr>`
      )
      .join("");

    const plan = stats.training_plan || {};

    return `<div class="stack">
      <div class="grid g-2" style="align-items:start">
        <div class="card">
          <div class="card-head"><div><h3>Cold-start tier</h3><p>How much of this is your data</p></div>
            <span class="chip chip-info">${esc(titleCase(metrics.model_tier))}</span></div>
          <div class="card-body stack">
            <p style="color:var(--ink-2)">${esc(TIER_COPY[metrics.model_tier] || "")}</p>
            <dl class="kv">
              <dt>Labelled leads</dt><dd>${stats.labelled.toLocaleString()}</dd>
              <dt>Next tier at</dt><dd>${plan.next_tier_at ? plan.next_tier_at.toLocaleString() : "—"}</dd>
              <dt>Model version</dt><dd><code class="inline">${esc(metrics.model_version)}</code></dd>
            </dl>
            <div class="callout">${esc(plan.detail || "")}</div>
            <div class="row">
              <button class="btn btn-primary" id="trainBtn">Train on my data</button>
              <span class="hint" style="color:var(--ink-4);font-size:12px">Promoted only if it beats the current model</span>
            </div>
            <div id="trainOut"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-head"><div><h3>Leakage screen</h3><p>Runs before every training job</p></div>
            ${findings.length ? `<span class="chip chip-bad">${findings.length} flagged</span>` : '<span class="chip chip-ok">Clean</span>'}</div>
          <div class="card-body stack">
            <p style="color:var(--ink-2)">Columns a rep fills in <em>after</em> the outcome is known make a model look
              excellent and perform terribly. Anything that predicts the outcome almost perfectly is quarantined here.</p>
            ${
              findings.length
                ? findings
                    .map(
                      (f) => `<div class="callout is-bad"><strong>${esc(f.feature)}</strong>
                        <span class="chip chip-bad" style="margin-left:6px">${esc(f.severity)}</span>
                        <div style="margin-top:4px">${esc(f.detail)}</div></div>`
                    )
                    .join("")
                : `<div class="callout is-ok">No feature shows suspicious separation against the outcome.</div>`
            }
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-head"><div><h3>Holdout performance</h3><p>Measured on data the model never saw</p></div></div>
        <div class="card-body tight"><div class="table-wrap"><table class="data">
          <thead><tr><th>Model</th><th class="num">Accuracy</th><th class="num">ROC-AUC</th><th class="num">PR-AUC</th><th class="num">Brier</th><th class="num">Weight</th></tr></thead>
          <tbody>${rows || `<tr><td colspan="6" style="color:var(--ink-3)">No metrics recorded.</td></tr>`}</tbody>
        </table></div></div>
      </div>

      <div class="card">
        <div class="card-head"><div><h3>Version history</h3><p>Champion / challenger decisions</p></div></div>
        <div class="card-body tight">
          ${
            versionRows
              ? `<div class="table-wrap"><table class="data">
                  <thead><tr><th>Version</th><th>Tier</th><th class="num">Rows</th><th class="num">ROC-AUC</th><th>State</th><th>Decision</th></tr></thead>
                  <tbody>${versionRows}</tbody></table></div>`
              : `<div class="empty" style="padding:30px"><p>No models trained on your data yet — the shared base model is scoring.</p></div>`
          }
        </div>
      </div>
    </div>`;
  },
  after(root) {
    root.querySelector("#trainBtn")?.addEventListener("click", async (event) => {
      const button = event.currentTarget;
      const out = root.querySelector("#trainOut");
      button.disabled = true;
      out.innerHTML = `<div class="loading"><span class="spinner-sm"></span></div>`;

      try {
        const result = await api("/v1/train", { method: "POST", body: {} });
        if (result.status === "skipped") {
          out.innerHTML = `<div class="callout is-warn">${esc(result.detail)}</div>`;
        } else {
          const auc = result.metrics?.hybrid_ensemble?.roc_auc;
          out.innerHTML = `<div class="callout ${result.promoted ? "is-ok" : "is-warn"}">
            <strong>${result.promoted ? "Promoted" : "Not promoted"}</strong> — ${esc(result.reason)}
            <div style="margin-top:5px">Tier ${esc(result.tier)} · ${result.training_rows.toLocaleString()} rows${auc ? ` · holdout AUC ${num(auc, 4)}` : ""}</div>
          </div>`;
          toast(result.promoted ? "New model promoted" : "Model trained but not promoted", result.promoted ? "is-ok" : "");
          await loadShell();
        }
      } catch (error) {
        out.innerHTML = `<div class="callout is-bad">${esc(error.message)}</div>`;
      } finally {
        button.disabled = false;
      }
    });
  },
};

/* ---------- monitoring ---------- */

views.monitoring = {
  title: "Monitoring",
  subtitle: "Is the model still trustworthy?",
  async render() {
    const monitoring = await api("/v1/monitoring");
    state.monitoring = monitoring;

    let lift = null;
    try {
      lift = await api("/v1/monitoring/lift");
    } catch {
      /* needs outcomes */
    }

    const drift = monitoring.drift;
    const calibration = monitoring.calibration;

    const driftBody = (drift.signals || []).length
      ? `<div class="table-wrap"><table class="data">
          <thead><tr><th>Feature</th><th class="num">PSI</th><th>Severity</th><th>What it means</th></tr></thead>
          <tbody>${drift.signals
            .map(
              (s) => `<tr>
              <td><code class="inline">${esc(s.feature)}</code></td>
              <td class="num">${num(s.psi, 3)}</td>
              <td>${s.severity === "alert" ? '<span class="chip chip-bad">Alert</span>' : '<span class="chip chip-warm">Watch</span>'}</td>
              <td style="color:var(--ink-2)">${esc(s.detail)}</td></tr>`
            )
            .join("")}</tbody></table></div>`
      : `<div class="card-body"><div class="callout is-ok">${esc(drift.detail)}</div></div>`;

    const bins = calibration.bins || [];
    const calBody = bins.length
      ? `<div class="bars">${bins
          .map((b) => {
            const gap = b.observed_rate - b.mean_predicted;
            return `<div class="bar-row">
              <span class="bar-label">${pct(b.bin_low, 0)}–${pct(b.bin_high, 0)}</span>
              <div class="bar-track" title="${b.count} leads">
                <div class="bar-fill" style="width:${Math.max(b.observed_rate * 100, 1)}%;background:${Math.abs(gap) > 0.15 ? "var(--bad)" : Math.abs(gap) > 0.08 ? "var(--warn)" : "var(--ok)"}"></div>
              </div>
              <span class="bar-value">${pct(b.observed_rate, 0)} <span style="color:var(--ink-4);font-weight:400">/ ${pct(b.mean_predicted, 0)}</span></span>
            </div>`;
          })
          .join("")}</div>
        <p class="hint" style="color:var(--ink-4);font-size:11.5px;margin-top:10px">Observed conversion rate / what the model predicted, per score band.</p>`
      : `<div class="callout">${esc(calibration.detail)}</div>`;

    const liftBody = lift?.deciles?.length
      ? `<div class="bars">${lift.deciles
          .map(
            (d) => `<div class="bar-row">
            <span class="bar-label">Decile ${d.decile}</span>
            <div class="bar-track"><div class="bar-fill" style="width:${Math.min((d.lift / 3.5) * 100, 100)}%"></div></div>
            <span class="bar-value">${d.lift}×</span>
          </div>`
          )
          .join("")}</div>`
      : `<div class="callout">Record outcomes on scored leads to measure lift.</div>`;

    return `<div class="stack">
      <div class="grid g-3">
        <div class="stat"><span class="stat-label">Overall</span>
          <span class="stat-value" style="font-size:17px;padding-top:6px">${statusChip(monitoring.health.status)}</span>
          <span class="stat-note">${esc(monitoring.health.recommendation)}</span></div>
        <div class="stat"><span class="stat-label">Calibration error</span>
          <span class="stat-value">${calibration.expected_calibration_error ?? "—"}</span>
          <span class="stat-note">${calibration.sample_size ? `${calibration.sample_size.toLocaleString()} matched outcomes` : "no outcomes yet"}</span></div>
        <div class="stat"><span class="stat-label">Live ROC-AUC</span>
          <span class="stat-value">${num(calibration.roc_auc)}</span>
          <span class="stat-note">measured against real outcomes</span></div>
      </div>

      <div class="grid g-2" style="align-items:start">
        <div class="card">
          <div class="card-head"><div><h3>Input drift</h3><p>Have incoming leads changed shape?</p></div>${statusChip(drift.status)}</div>
          ${driftBody}
        </div>
        <div class="card">
          <div class="card-head"><div><h3>Calibration</h3><p>Do the probabilities mean what they say?</p></div>${statusChip(calibration.status)}</div>
          <div class="card-body">${calBody}</div>
        </div>
      </div>

      <div class="card">
        <div class="card-head"><div><h3>Decile lift</h3><p>${lift?.top_decile_lift ? `Top decile converts ${lift.top_decile_lift}× the average` : "How much better than random is the ranking?"}</p></div></div>
        <div class="card-body">${liftBody}</div>
      </div>
    </div>`;
  },
};

/* ─────────────────────────── shared renderers ─────────────────────────── */

function renderExplanation(data) {
  const contributions = data.contributions || [];
  const largest = Math.max(...contributions.map((c) => Math.abs(c.impact)), 0.001);

  return `<div class="stack">
    <div class="drawer-score">
      <span class="big">${pct(data.probability)}</span>
      ${bandChip(data.band)}
    </div>
    <p style="color:var(--ink-2)">${esc(data.summary)}</p>

    ${
      data.next_best_action
        ? `<div class="callout ${data.next_best_action.action === "no_outreach" ? "is-warn" : "is-ok"}">
             <strong>${esc(titleCase(data.next_best_action.action))}</strong> — ${esc(data.next_best_action.reason)}
           </div>`
        : ""
    }

    <div>
      <div class="section-label">What moved this score</div>
      ${
        contributions.length
          ? `<div class="contrib">${contributions
              .map((c) => {
                const positive = c.impact > 0;
                const width = Math.max((Math.abs(c.impact) / largest) * 100, 3);
                return `<div>
                  <div class="contrib-row">
                    <span class="contrib-label"><b>${esc(c.feature)}</b> <span>${esc(c.value)}</span></span>
                    <span class="contrib-val ${positive ? "pos" : "neg"}">${positive ? "+" : "−"}${(Math.abs(c.impact) * 100).toFixed(1)}</span>
                  </div>
                  <div class="contrib-bar"><i class="${positive ? "pos" : "neg"}" style="width:${width}%"></i></div>
                </div>`;
              })
              .join("")}</div>
             <p class="hint" style="color:var(--ink-4);font-size:11.5px;margin-top:9px">
               Percentage points versus a typical lead, measured by re-scoring with each field reset to its baseline.</p>`
          : `<p style="color:var(--ink-3)">This lead sits close to a typical one — no single field stands out.</p>`
      }
    </div>
  </div>`;
}

function bindLeadRows(root) {
  root.querySelectorAll("tr[data-lead]").forEach((row) => {
    row.addEventListener("click", (event) => {
      if (event.target.closest("button") && !event.target.closest("[data-explain]")) return;
      openLeadDrawer(JSON.parse(row.dataset.lead));
    });
  });
}

async function openLeadDrawer(lead) {
  el("drawer").hidden = false;
  el("drawerScrim").hidden = false;
  el("drawerTitle").textContent = lead.display_name || lead.external_id;
  el("drawerSub").innerHTML =
    `Rank ${lead.rank} · ${esc(lead.external_id)} · expected value ${money(lead.expected_value)}`;
  el("drawerBody").innerHTML = `<div class="loading"><span class="spinner-sm"></span></div>`;

  try {
    const explanation = await api("/explain", { method: "POST", body: lead.payload || {} });

    el("drawerBody").innerHTML = `
      ${renderExplanation(explanation)}
      <div>
        <div class="section-label">Record the outcome</div>
        <p style="color:var(--ink-3);font-size:12.5px;margin-bottom:9px">
          Outcomes are what let the model learn your pipeline instead of a generic one.</p>
        <div class="row">
          <button class="btn btn-sm btn-primary" data-outcome="true">Mark converted</button>
          <button class="btn btn-sm" data-outcome="false">Mark not converted</button>
        </div>
        <div id="outcomeOut" style="margin-top:10px"></div>
      </div>
      <div>
        <div class="section-label">Canonical record</div>
        <pre class="code">${esc(JSON.stringify(lead.payload || {}, null, 2))}</pre>
      </div>`;

    el("drawerBody")
      .querySelectorAll("[data-outcome]")
      .forEach((button) =>
        button.addEventListener("click", async () => {
          const converted = button.dataset.outcome === "true";
          const out = el("outcomeOut");
          out.innerHTML = `<div class="loading" style="padding:8px"><span class="spinner-sm"></span></div>`;
          try {
            const result = await api("/v1/leads/outcome", {
              method: "POST",
              body: { lead_id: lead.lead_id, converted },
            });
            out.innerHTML = `<div class="callout is-ok">Recorded. ${result.labelled_total.toLocaleString()} labelled lead(s) total.
              <div style="margin-top:4px">${esc(result.training_plan.detail)}</div></div>`;
            toast("Outcome recorded", "is-ok");
            await loadShell();
          } catch (error) {
            out.innerHTML = `<div class="callout is-bad">${esc(error.message)}</div>`;
          }
        })
      );
  } catch (error) {
    el("drawerBody").innerHTML = `<div class="callout is-bad">${esc(error.message)}</div>`;
  }
}

function closeDrawer() {
  el("drawer").hidden = true;
  el("drawerScrim").hidden = true;
}

/* ─────────────────────────────── router ─────────────────────────────── */

async function route() {
  const name = (location.hash.replace(/^#\//, "") || "overview").split("?")[0];
  const view = views[name] || views.overview;

  document.querySelectorAll(".nav-item").forEach((item) =>
    item.classList.toggle("is-active", item.dataset.view === (views[name] ? name : "overview"))
  );
  el("viewTitle").textContent = view.title;
  el("viewSubtitle").textContent = view.subtitle;

  const root = el("view");
  root.innerHTML = `<div class="loading"><span class="spinner-sm"></span></div>`;

  try {
    root.innerHTML = await view.render();
    view.after?.(root);
  } catch (error) {
    root.innerHTML = `<div class="card"><div class="empty">
      <h3>Could not load this view</h3><p>${esc(error.message)}</p>
      <button class="btn" onclick="location.reload()">Reload</button></div></div>`;
  }
}

window.addEventListener("hashchange", route);
el("drawerClose").addEventListener("click", closeDrawer);
el("drawerScrim").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeDrawer();
});
el("refreshBtn").addEventListener("click", async () => {
  await loadShell();
  route();
});

(async function start() {
  try {
    state.schema = await api("/schema");
  } catch {
    /* the mapping editor falls back to a free-text list */
  }
  await loadShell();
  await route();
})();
