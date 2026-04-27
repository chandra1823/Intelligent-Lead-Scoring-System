from typing import Any, Dict

from pydantic import BaseModel, Field


class LeadFeatures(BaseModel):
    total_time_spent_on_website: float = Field(..., ge=0)
    page_views_per_visit: float = Field(..., ge=0)
    total_visits: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "total_time_spent_on_website": 180.0,
                "page_views_per_visit": 2.4,
                "total_visits": 5.0,
            }
        }


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    label: str
    model: str
    centralized_output: Dict[str, Any]


class ExplainResponse(BaseModel):
    summary: str
    input_features: Dict[str, Any]
