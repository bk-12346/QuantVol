import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import xgboost as xgb

from quantvol.data.ingest import load_config
from quantvol.models.baselines.har_rv_model import build_har_features
from quantvol.models.ml.lgbm_model import FEATURE_COLS as XGB_FEATURE_COLS, TARGET_COL


def train_final_har_rv(df: pd.DataFrame, save_path: str = "models/har_rv_final.joblib"):
    """Train HAR-RV on all available data and save it."""
    df = build_har_features(df, vol_col="realized_vol_21d")
    df = df.dropna(subset=["har_daily", "har_weekly", "har_monthly", "har_target"])

    model = LinearRegression()
    model.fit(df[["har_daily", "har_weekly", "har_monthly"]], df["har_target"])

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Saved HAR-RV final model to {save_path}")

    return model

def train_final_xgboost(df: pd.DataFrame, save_path: str = "models/xgboost_final.joblib"):
    """Train XGBoost on all available data and save it."""
    df = df.dropna(subset=XGB_FEATURE_COLS + [TARGET_COL])

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        verbosity=0,
    )
    model.fit(df[XGB_FEATURE_COLS], df[TARGET_COL])

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Saved XGBoost final model to {save_path}")

    return model

def main():
    config = load_config()

    # HAR-RV uses the processed features file (has realized_vol_21d)
    processed_path = Path(config["output_paths"]["processed"])
    har_df = pd.read_csv(processed_path, parse_dates=["Date"])
    train_final_har_rv(har_df)

    # XGBoost uses the richer ml_features file
    ml_df = pd.read_csv("data/processed/ml_features.csv", parse_dates=["Date"])
    train_final_xgboost(ml_df)

    print("\nBoth final models trained and saved to models/")


if __name__ == "__main__":
    main()