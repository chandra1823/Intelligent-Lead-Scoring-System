const form = document.getElementById("predictForm");
const statusText = document.getElementById("statusText");
const resultBlock = document.getElementById("resultBlock");
const categoricalFields = document.getElementById("categoricalFields");
const modelBadge = document.getElementById("modelBadge");
const sampleBtn = document.getElementById("sampleBtn");

const predictionEl = document.getElementById("prediction");
const probabilityEl = document.getElementById("probability");
const labelEl = document.getElementById("label");
const modelEl = document.getElementById("model");
const explainEl = document.getElementById("explain");
const componentsEl = document.getElementById("components");
const contributionsEl = document.getElementById("contributions");
const scoreFill = document.getElementById("scoreFill");

// Canonical field -> label shown in the form. Names come from ml/canonical.py;
// /metrics supplies the option values for each one.
const CATEGORICAL_FIELDS = [
  ["origin", "Lead origin"],
  ["channel", "Channel / lead source"],
  ["last_activity", "Last activity"],
  ["last_notable_activity", "Last notable activity"],
  ["occupation", "Current occupation"],
  ["specialization", "Specialization"],
  ["heard_about_us", "Heard about us via"],
  ["motivation", "Main motivation"],
  ["city", "City"],
  ["country", "Country"],
  ["do_not_contact", "Do not contact"],
  ["wants_free_material", "Wants free material"],
];

const SAMPLE_LEAD = {
  timeSpent: 1200,
  pageViews: 5,
  totalVisits: 8,
  origin: "Lead Add Form",
  channel: "Reference",
  last_activity: "SMS Sent",
  occupation: "Working Professional",
};

async function loadMetadata() {
  try {
    const [metricsRes, healthRes] = await Promise.all([
      fetch("/metrics"),
      fetch("/health"),
    ]);

    const metrics = await metricsRes.json();
    const health = await healthRes.json();

    const hybrid = metrics.metrics?.hybrid_ensemble;
    const tier = metrics.model_tier ? ` · ${metrics.model_tier}` : "";
    modelBadge.textContent = hybrid
      ? `${health.model}${tier} · AUC ${hybrid.roc_auc.toFixed(3)}`
      : `${health.model}${tier}`;

    renderCategoricalFields(metrics.category_options || {});
  } catch (error) {
    modelBadge.textContent = "backend unavailable";
    categoricalFields.innerHTML = `<p class="muted-center">Could not load options: ${error.message}</p>`;
  }
}

function renderCategoricalFields(options) {
  categoricalFields.innerHTML = "";

  CATEGORICAL_FIELDS.forEach(([field, labelText]) => {
    const values = options[field] || [];
    if (!values.length) return;

    const label = document.createElement("label");
    label.textContent = labelText;

    const select = document.createElement("select");
    select.id = field;

    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = "— use typical value —";
    select.appendChild(blank);

    values.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      select.appendChild(option);
    });

    label.appendChild(select);
    categoricalFields.appendChild(label);
  });
}

function collectPayload() {
  const payload = {
    time_on_site_seconds: Number(document.getElementById("timeSpent").value),
    page_views_per_visit: Number(document.getElementById("pageViews").value),
    total_visits: Number(document.getElementById("totalVisits").value),
  };

  CATEGORICAL_FIELDS.forEach(([field]) => {
    const element = document.getElementById(field);
    if (element && element.value) {
      payload[field] = element.value;
    }
  });

  return payload;
}

function renderComponents(centralized) {
  const components = centralized.components || {};
  const weights = centralized.weights || {};
  componentsEl.innerHTML = "";

  const names = Object.keys(components);
  if (!names.length) {
    componentsEl.innerHTML = '<p class="muted-small">No trained models loaded — heuristic fallback in use.</p>';
    return;
  }

  names.forEach((name) => {
    const row = document.createElement("div");
    row.className = "component-row";
    const weight = weights[name] !== undefined ? ` · weight ${(weights[name] * 100).toFixed(0)}%` : "";
    row.innerHTML = `<span>${name}${weight}</span><strong>${components[name].toFixed(3)}</strong>`;
    componentsEl.appendChild(row);
  });
}

function renderContributions(contributions) {
  contributionsEl.innerHTML = "";

  if (!contributions || !contributions.length) {
    contributionsEl.innerHTML = '<p class="muted-small">No single field moved the score far from a typical lead.</p>';
    return;
  }

  const largest = Math.max(...contributions.map((item) => Math.abs(item.impact)));

  contributions.forEach((item) => {
    const points = (item.impact * 100).toFixed(1);
    const positive = item.impact > 0;
    const width = Math.max((Math.abs(item.impact) / largest) * 100, 4);

    const row = document.createElement("div");
    row.className = "contribution-row";
    row.innerHTML = `
      <div class="contribution-head">
        <span>${item.feature}: <em>${item.value}</em></span>
        <strong class="${positive ? "pos" : "neg"}">${positive ? "+" : ""}${points} pts</strong>
      </div>
      <div class="contribution-bar">
        <div class="contribution-fill ${positive ? "pos" : "neg"}" style="width:${width}%"></div>
      </div>`;
    contributionsEl.appendChild(row);
  });
}

sampleBtn.addEventListener("click", () => {
  document.getElementById("timeSpent").value = SAMPLE_LEAD.timeSpent;
  document.getElementById("pageViews").value = SAMPLE_LEAD.pageViews;
  document.getElementById("totalVisits").value = SAMPLE_LEAD.totalVisits;

  Object.entries(SAMPLE_LEAD).forEach(([key, value]) => {
    const element = document.getElementById(key);
    if (element && element.tagName === "SELECT") {
      const match = Array.from(element.options).find((option) => option.value === value);
      if (match) element.value = value;
    }
  });
});

function renderNextAction(action) {
  const target = document.getElementById("nextAction");
  if (!target) return;
  if (!action || !action.action) {
    target.textContent = "";
    return;
  }
  const name = action.action.replace(/_/g, " ");
  target.innerHTML = `<strong>${name}</strong> — ${action.reason}`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = collectPayload();
  statusText.textContent = "Running model prediction...";
  resultBlock.classList.add("hidden");

  try {
    // /explain returns the score plus per-field attributions in one round trip.
    const response = await fetch("/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`${response.status} ${detail.slice(0, 120)}`);
    }

    const data = await response.json();

    const predictResponse = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const predictData = await predictResponse.json();

    const percent = (data.probability * 100).toFixed(1);
    probabilityEl.textContent = `${percent}%`;
    scoreFill.style.width = `${percent}%`;
    scoreFill.className = `score-fill ${data.prediction === 1 ? "pos" : "neg"}`;
    labelEl.textContent = data.label.replace(/_/g, " ");
    predictionEl.textContent = data.prediction;
    modelEl.textContent = predictData.model;
    explainEl.textContent = data.summary;

    renderComponents(predictData.centralized_output || {});
    renderContributions(data.contributions);
    renderNextAction(data.next_best_action);

    statusText.textContent = "Prediction complete.";
    resultBlock.classList.remove("hidden");
  } catch (error) {
    statusText.textContent = `Failed to run prediction: ${error.message}`;
  }
});

loadMetadata();
