import streamlit as st
import requests
from datetime import date

st.set_page_config(page_title="QuantVol", page_icon="📈", layout="centered")

st.title("QuantVol — Volatility Forecasting")
st.caption("HAR-RV and XGBoost forecasts for S&P 500 realized volatility, with SHAP-based interpretability.")

API_URL = "http://127.0.0.1:8000/forecast"

use_latest = st.checkbox("Use latest available data", value=True)

as_of_date = None
if not use_latest:
    selected_date = st.date_input("Forecast as of date", value=date(2020, 3, 13))
    as_of_date = str(selected_date)

if st.button("Get Forecast"):
    with st.spinner("Fetching forecast..."):
        try:
            response = requests.post(API_URL, json={"as_of_date": as_of_date})
            response.raise_for_status()
            result = response.json()
            st.session_state["result"] = result
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the API: {e}")

if "result" in st.session_state:
    result = st.session_state["result"]

    st.subheader(f"Forecast as of {result['as_of_date']}")

    col1, col2 = st.columns(2)
    col1.metric("HAR-RV Forecast (Primary)", f"{result['har_rv_forecast']:.4f}")
    col2.metric("XGBoost Forecast", f"{result['xgboost_forecast']:.4f}")

    st.markdown("**Top drivers behind the XGBoost prediction (SHAP):**")

    import pandas as pd
    drivers_df = pd.DataFrame(result["xgboost_top_drivers"])
    drivers_df = drivers_df.set_index("feature")

    st.bar_chart(drivers_df["shap_value"])