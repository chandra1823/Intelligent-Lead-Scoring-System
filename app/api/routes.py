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
    return {"status": "ok", "model": model_service.model_name}


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
            "Lead appears promising based on website engagement and visit depth. "
            "For a college project, treat this as a supportive indicator, not a final business decision."
        )
    else:
        summary = (
            "Lead appears less likely to convert from the current engagement signals. "
            "Consider nurturing campaigns and follow-up before final judgment."
        )

    return ExplainResponse(summary=summary, input_features=payload.model_dump())
