# Intelligent Lead Scoring System

A final-year project that predicts lead conversion probability using a hybrid model architecture:
- RandomForest
- XGBoost
- LightGBM
- Weighted hybrid ensemble + fallback mode

This guide takes you from **cloning** to **training models** to **running backend + frontend**.

---

## 1) Prerequisites

Install before starting:
- **Git**
- **Python 3.10+**
- **pip**
- **VS Code** (recommended)

Check versions:

```bash
python --version
git --version
```

---

## 2) Clone the repository

```bash
git clone <YOUR_REPO_URL>
cd Intelligent-Lead-Scoring-System
```

Optional (open in VS Code):

```bash
code .
```

---

## 3) Create and activate virtual environment

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

---

## 4) Install dependencies (Backend + Model training)

> This project uses a static HTML/CSS/JS frontend served by FastAPI, so there is **no separate npm/yarn frontend install** required.

Install Python packages:

```bash
pip install -r requirements.txt
```

---

## 5) Train models and create `.pkl` artifacts

Run:

```bash
python scripts/train_model.py
```

Generated artifacts (depending on available libraries):

```text
artifacts/random_forest.pkl
artifacts/xgboost.pkl
artifacts/lightgbm.pkl
artifacts/hybrid_meta.json
```

If XGBoost/LightGBM libraries are unavailable, the script still trains available models and writes metadata.

---

## 6) Run backend API

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Backend URLs:
- API Docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health
- Predict: `POST /predict`
- Explain: `POST /explain`

---

## 7) Run frontend

Frontend is served by the same FastAPI server:
- Landing page: http://127.0.0.1:8000/ui
- Dashboard: http://127.0.0.1:8000/ui/dashboard

No separate frontend server command is needed.

---

## 8) Verify frontend-backend connectivity

With backend running, test quickly:

```bash
curl -s http://127.0.0.1:8000/health
```

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"total_time_spent_on_website":180,"page_views_per_visit":2.4,"total_visits":5}'
```

If this returns JSON prediction output, frontend can consume the same endpoint.

---

## 9) Run tests

```bash
python -m unittest discover -s test -v
```

Notes:
- `test_model_service.py` validates hybrid/fallback behavior.
- `test_api_routes.py` validates API endpoints and UI routes.
- API tests may auto-skip in environments missing TestClient runtime deps.

---

## 10) Quick full run (copy-paste)

```bash
git clone <YOUR_REPO_URL>
cd Intelligent-Lead-Scoring-System
python -m venv .venv
# activate .venv
pip install -r requirements.txt
python scripts/train_model.py
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Then open:
- http://127.0.0.1:8000/ui

---

## 11) Troubleshooting

### A) `ModuleNotFoundError`
```bash
pip install -r requirements.txt
```

### B) Port already in use
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

### C) No model artifacts found
Run training again:
```bash
python scripts/train_model.py
```

### D) Frontend opens but no prediction
- Confirm backend is running.
- Test `/health` and `/predict` using curl.
- Check browser dev tools (Network tab) for request errors.

---

## License

Academic project (final-year demonstration).
