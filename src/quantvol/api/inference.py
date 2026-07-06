# loads model, runs prediction
import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path
from datetime import datetime

from quantvol.models.baselines.har_rv_model import build_har_features
from quantvol.models.ml.lgbm_model import FEATURE_COLS as XGB_FEATURE_COLS, TARGET_COL


def load_models():
    """Load both saved final models."""
    har_model = joblib.load("models/har_rv_final.joblib")
    xgb_model = joblib.load("models/xgboost_final.joblib")
    return har_model, xgb_model


def load_latest_features(as_of_date: str = None) -> pd.DataFrame:
    """
    Load the processed feature data, optionally filtered to a specific date.
    If as_of_date is None, uses the most recent available date.
    """
    ml_df = pd.read_csv("data/processed/ml_features.csv", parse_dates=["Date"])

    if as_of_date is not None:
        target_date = pd.Timestamp(as_of_date)
        ml_df = ml_df[ml_df["Date"] <= target_date]
        if ml_df.empty:
            raise ValueError(f"No data available on or before {as_of_date}")

    return ml_df.iloc[[-1]]  # last available row up to (and including) as_of_date

def predict_volatility(as_of_date: str = None) -> dict:
    """
    Generate a volatility forecast using HAR-RV (primary) and XGBoost (supplementary,
    with SHAP-based top drivers). Uses data as of as_of_date, or the latest available
    data if as_of_date is None.
    """
    har_model, xgb_model = load_models()

    # --- HAR-RV features ---
    processed_df = pd.read_csv("data/processed/gspc_features.csv", parse_dates=["Date"])
    processed_df = build_har_features(processed_df, vol_col="realized_vol_21d")

    if as_of_date is not None:
        target_date = pd.Timestamp(as_of_date)
        processed_df = processed_df[processed_df["Date"] <= target_date]
        if processed_df.empty:
            raise ValueError(f"No data available on or before {as_of_date}")

    har_row = processed_df.iloc[[-1]]
    har_features = har_row[["har_daily", "har_weekly", "har_monthly"]]
    har_prediction = float(har_model.predict(har_features)[0])

    # --- XGBoost features ---
    xgb_row = load_latest_features(as_of_date)
    xgb_features = xgb_row[XGB_FEATURE_COLS]
    xgb_prediction = float(xgb_model.predict(xgb_features)[0])

    # --- SHAP explanation for this specific prediction ---
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(xgb_features)

    shap_dict = dict(zip(XGB_FEATURE_COLS, shap_values[0]))
    top_drivers = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    top_drivers_formatted = [
        {"feature": name, "shap_value": round(float(value), 5)} for name, value in top_drivers
    ]

    forecast_date = har_row["Date"].iloc[0]

    return {
        "as_of_date": str(forecast_date.date()),
        "har_rv_forecast": round(har_prediction, 5),
        "xgboost_forecast": round(xgb_prediction, 5),
        "xgboost_top_drivers": top_drivers_formatted,
    }