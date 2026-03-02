# Intelligent Lead Scoring System (College Project)

This repository contains your lead scoring dataset work and now includes a backend API built with **FastAPI**.

## What was fixed

- Added a runnable backend structure (instead of notebook-only project).
- Added train script and model service for loading ML model safely.
- Added fallback prediction logic so the API works even before training.
- Added clear dependency file (`requirements.txt`).

## Project structure

```text
app/
  api/routes.py          # API endpoints
  core/config.py         # app settings
  models/schemas.py      # request/response models
  main.py                # FastAPI app entrypoint
ml/
  model_service.py       # model loading + prediction service
scripts/
  train_model.py         # train and save model artifact
artifacts/
  lead_model.pkl         # generated after training
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

This will create `artifacts/lead_model.pkl`.

## Run backend

```bash
uvicorn app.main:app --reload
```

Open docs at: `http://127.0.0.1:8000/docs`

## API endpoints

- `GET /` : basic message
- `GET /health` : app + model status
- `POST /predict` : conversion prediction
- `POST /explain` : simple explanation text for college demo

### Example payload

```json
{
  "total_time_spent_on_website": 180,
  "page_views_per_visit": 2.4,
  "total_visits": 5
}
```

## Future integration notes (React)

For future React integration:
- call `POST /predict` from your frontend form,
- display probability and label,
- optionally call `POST /explain` for explanation card.

This backend is intentionally simple and educational for a final-year college project.
