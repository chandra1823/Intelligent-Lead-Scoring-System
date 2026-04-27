from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.core.config import settings
from app.models.schemas import ExplainResponse, LeadFeatures, PredictionResponse
from ml.model_service import LeadScoringModel

router = APIRouter()
model_service = LeadScoringModel(settings.model_path)


@router.get("/")
def root() -> dict:
    return {"message": "Lead Scoring API is running", "ui": "/ui"}


@router.get("/ui")
def ui_landing() -> FileResponse:
    return FileResponse("frontend/index.html")


@router.get("/ui/dashboard")
def ui_dashboard() -> FileResponse:
    return FileResponse("frontend/dashboard.html")


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": model_service.model_name,
        "hybrid_weights": model_service.weights,
    }


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: LeadFeatures) -> PredictionResponse:
    result = model_service.predict(
        total_time_spent_on_website=payload.total_time_spent_on_website,
        page_views_per_visit=payload.page_views_per_visit,
        total_visits=payload.total_visits,
    )

    label = "likely_to_convert" if result.prediction == 1 else "unlikely_to_convert"

    return PredictionResponse(
        prediction=result.prediction,
        probability=round(result.probability, 4),
        label=label,
        model=result.model_name,
        centralized_output=result.centralized_output,
    )


@router.post("/explain", response_model=ExplainResponse)
def explain(payload: LeadFeatures) -> ExplainResponse:
    result = model_service.predict(
        total_time_spent_on_website=payload.total_time_spent_on_website,
        page_views_per_visit=payload.page_views_per_visit,
        total_visits=payload.total_visits,
    )

    if result.prediction == 1:
        summary = (
            "Hybrid model indicates strong conversion potential. "
            "Random Forest, XGBoost, and LightGBM ensemble score is above threshold."
        )
    else:
        summary = (
            "Hybrid model indicates lower conversion potential right now. "
            "Consider additional engagement before follow-up prioritization."
        )

    return ExplainResponse(summary=summary, input_features=payload.model_dump())
