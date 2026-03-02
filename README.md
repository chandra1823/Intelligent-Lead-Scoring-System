# Intelligent Lead Scoring System (College Project)

This repository contains lead scoring dataset notebooks and a full **FastAPI backend + frontend UI** for a final-year project demo.

## Features

- FastAPI backend with prediction endpoints.
- Rule-based fallback when trained artifact is unavailable.
- Training script for baseline RandomForest model.
- Minimal professional frontend:
  - Landing page (`/ui`)
  - Dashboard page (`/ui/dashboard`)
  - Dashboard calls backend `/predict` and shows model output.
- Unit tests under `test/`.

## Project structure

```text
app/
  api/routes.py
  core/config.py
  models/schemas.py
  main.py
ml/
  model_service.py
scripts/
  train_model.py
frontend/
  index.html
  dashboard.html
  dashboard.js
  styles.css
test/
  test_api_routes.py
  test_model_service.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train model

```bash
python scripts/train_model.py
```

## Run app

```bash
uvicorn app.main:app --reload
```

- API docs: `http://127.0.0.1:8000/docs`
- Landing page: `http://127.0.0.1:8000/ui`
- Dashboard: `http://127.0.0.1:8000/ui/dashboard`

## Run tests

```bash
python -m unittest discover -s test -v
```

> In restricted environments without FastAPI dependencies, API tests auto-skip.
