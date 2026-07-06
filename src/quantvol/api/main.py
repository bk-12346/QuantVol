from fastapi import FastAPI, HTTPException

from quantvol.api.schemas import ForecastRequest, ForecastResponse
from quantvol.api.inference import predict_volatility

app = FastAPI(
    title="QuantVol API",
    description="Volatility forecasting API using HAR-RV and XGBoost, with SHAP-based interpretability.",
    version="0.1.0",
)


@app.get("/")
def root():
    return {"message": "QuantVol API is running. See /docs for usage."}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    try:
        result = predict_volatility(as_of_date=request.as_of_date)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))