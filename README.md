# Intelligent Lead Scoring System

A final-year college project that predicts whether a lead is likely to convert using engagement-based inputs.

This repository includes:
- a **FastAPI backend** for prediction APIs,
- a **web UI** (landing + dashboard),
- a **training script** for generating hybrid ML artifacts (RandomForest + XGBoost + LightGBM),
- and **unit tests**.

---

## 1) Prerequisites

Install these tools first:

- **Git** (for cloning the project)
- **Python 3.10+**
- **VS Code**
- (Recommended) **Python extension in VS Code**

You can verify Python and Git:

```bash
python --version
git --version
```

---

## 2) Clone the project in VS Code

### Option A: using VS Code GUI
1. Open VS Code.
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS).
3. Search: **Git: Clone**.
4. Paste your repository URL.
5. Choose a folder.
6. Click **Open** when cloning is complete.

### Option B: using terminal

```bash
git clone <YOUR_REPOSITORY_URL>
cd Intelligent-Lead-Scoring-System
code .
```

> Replace `<YOUR_REPOSITORY_URL>` with your repo link.

---

## 3) Create and activate virtual environment

From the project root (`Intelligent-Lead-Scoring-System`):

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Windows (CMD)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

When activated, your terminal should show `(.venv)`.

---

## 4) Install dependencies

```bash
pip install -r requirements.txt
```

If `pip` is outdated, update it first:

```bash
python -m pip install --upgrade pip
```

---

## 5) Run the project

Start FastAPI server:

```bash
uvicorn app.main:app --reload
```

After startup, open:

- **Landing page:** http://127.0.0.1:8000/ui
- **Dashboard:** http://127.0.0.1:8000/ui/dashboard
- **Swagger API docs:** http://127.0.0.1:8000/docs

---

## 6) How to use the dashboard

1. Open `/ui/dashboard`.
2. Enter:
   - Total Time Spent on Website
   - Page Views Per Visit
   - Total Visits
3. Click **Run Prediction**.
4. You will see:
   - prediction (0/1)
   - probability
   - label
   - model name
   - explanation summary

---

## 7) Train your own hybrid model artifacts (optional but recommended)

By default, if trained artifacts are missing, hybrid service falls back to deterministic heuristic scoring for demo continuity.

To train baseline RandomForest model:

```bash
python scripts/train_model.py
```

This creates (depending on available libraries):

```text
artifacts/random_forest.pkl
artifacts/xgboost.pkl
artifacts/lightgbm.pkl
artifacts/hybrid_meta.json
```

After that, restart server:

```bash
uvicorn app.main:app --reload
```

Now `/predict` uses hybrid ensemble output when artifacts are available.

---

## 8) Run tests

```bash
python -m unittest discover -s test -v
```

Notes:
- `test_model_service.py` validates fallback behavior.
- `test_api_routes.py` validates API routes.
- In limited environments without FastAPI test dependencies, API tests may auto-skip.

---

## 9) Project structure

```text
app/
  main.py                 # FastAPI app setup
  api/routes.py           # API + UI routes
  core/config.py          # app settings
  models/schemas.py       # request/response schemas
frontend/
  index.html              # landing page
  dashboard.html          # dashboard page
  dashboard.js            # frontend-backend integration
  styles.css              # UI styling
ml/
  model_service.py        # model loading + prediction logic
scripts/
  train_model.py          # model training utility
test/
  test_model_service.py   # model service tests
  test_api_routes.py      # API route tests
```

---

## 10) Common issues & fixes

### A) `ModuleNotFoundError`
- Ensure virtual environment is active.
- Reinstall dependencies:

```bash
pip install -r requirements.txt
```

### B) Port already in use (`8000`)
Run on another port:

```bash
uvicorn app.main:app --reload --port 8001
```

Then open http://127.0.0.1:8001/ui

### C) Dashboard not updating results
- Check backend terminal for errors.
- Verify backend is running at same host/port where frontend is served (`/ui/dashboard`).
- Test API directly from `/docs`.

---

## 11) For evaluators (quick run)

```bash
git clone <YOUR_REPOSITORY_URL>
cd Intelligent-Lead-Scoring-System
python -m venv .venv
# activate .venv
pip install -r requirements.txt
python scripts/train_model.py
uvicorn app.main:app --reload
```

Open: http://127.0.0.1:8000/ui

---

## License / Usage

This project is created for academic demonstration (final-year project).
