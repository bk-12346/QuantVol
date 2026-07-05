# lags, rolling stats, GARCH-implied vol

import pandas as pd
import numpy as np
from pathlib import Path


def build_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a richer feature set for ML models, extending beyond HAR-RV's
    3 features. All features use .shift(1) before any rolling calculation,
    ensuring only information available at t-1 or earlier is used to
    predict the target at t.
    """
    df = df.copy()

    # --- Lagged realized volatility at multiple horizons ---
    for window in [1, 5, 10, 21, 63]:
        if window == 1:
            df[f"rv_lag_{window}d"] = df["realized_vol_21d"].shift(1)
        else:
            df[f"rv_lag_{window}d"] = (
                df["realized_vol_21d"].shift(1).rolling(window=window, min_periods=window).mean()
            )
    
    # --- Lagged alternative volatility estimators (Parkinson, Garman-Klass) ---
    for col in ["parkinson_vol_21d", "garman_klass_vol_21d"]:
        df[f"{col}_lag1"] = df[col].shift(1)

    # --- Return-based features ---
    df["return_lag1"] = df["log_return"].shift(1)
    df["abs_return_lag1"] = df["log_return"].shift(1).abs()

    # Rolling skewness and kurtosis of returns - captures changing tail risk,
    # not just the vol level. Computed on lagged returns only.
    df["return_skew_21d"] = (
        df["log_return"].shift(1).rolling(window=21, min_periods=21).skew()
    )
    df["return_kurt_21d"] = (
        df["log_return"].shift(1).rolling(window=21, min_periods=21).kurt()
    )

    # --- Target: current realized volatility (what we're predicting) ---
    df["target"] = df["realized_vol_21d"]

    return df

def main():
    from quantvol.data.ingest import load_config

    config = load_config()
    processed_path = Path(config["output_paths"]["processed"])

    df = pd.read_csv(processed_path, parse_dates=["Date"])
    df = build_ml_features(df)

    output_path = Path("data/processed/ml_features.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved ML feature set to {output_path} ({len(df)} rows)")
    print()
    feature_cols = [c for c in df.columns if c not in ["Date", "Close", "High", "Low", "Open", "Volume", "regime"]]
    print("NaN counts per feature:")
    print(df[feature_cols].isna().sum())
    print()
    print(f"Rows with complete features (no NaN): {df[feature_cols].dropna().shape[0]}")


if __name__ == "__main__":
    main()