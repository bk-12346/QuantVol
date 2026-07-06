# request/response models
from pydantic import BaseModel, Field
from typing import Optional, List


class ForecastRequest(BaseModel):
    as_of_date: Optional[str] = Field(
        default=None,
        description="Forecast as of this date (YYYY-MM-DD). If omitted, uses the latest available data.",
    )


class TopDriver(BaseModel):
    feature: str
    shap_value: float


class ForecastResponse(BaseModel):
    as_of_date: str
    har_rv_forecast: float
    xgboost_forecast: float
    xgboost_top_drivers: List[TopDriver]