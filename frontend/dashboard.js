const form = document.getElementById("predictForm");
const statusText = document.getElementById("statusText");
const resultBlock = document.getElementById("resultBlock");

const predictionEl = document.getElementById("prediction");
const probabilityEl = document.getElementById("probability");
const labelEl = document.getElementById("label");
const modelEl = document.getElementById("model");
const explainEl = document.getElementById("explain");

async function runPrediction(payload) {
  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Prediction failed (${response.status})`);
  }

  return response.json();
}

async function runExplanation(payload) {
  const response = await fetch("/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    return { summary: "Explanation unavailable." };
  }

  return response.json();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    total_time_spent_on_website: Number(document.getElementById("timeSpent").value),
    page_views_per_visit: Number(document.getElementById("pageViews").value),
    total_visits: Number(document.getElementById("totalVisits").value),
  };

  statusText.textContent = "Running model prediction...";
  resultBlock.classList.add("hidden");

  try {
    const [predictionData, explainData] = await Promise.all([
      runPrediction(payload),
      runExplanation(payload),
    ]);

    predictionEl.textContent = predictionData.prediction;
    probabilityEl.textContent = predictionData.probability;
    labelEl.textContent = predictionData.label;
    modelEl.textContent = predictionData.model;
    explainEl.textContent = explainData.summary;

    statusText.textContent = "Prediction complete.";
    resultBlock.classList.remove("hidden");
  } catch (error) {
    statusText.textContent = `Failed to run prediction: ${error.message}`;
  }
});
