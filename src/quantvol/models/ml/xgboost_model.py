import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb

from quantvol.models.ml.lgbm_model import FEATURE_COLS, TARGET_COL


def fit_predict_xgboost(train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict = None) -> np.ndarray:
    """Fit an XGBoost regressor on train_df and predict on test_df."""
    if params is None:
        params = {
            "objective": "reg:squarederror",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_weight": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "verbosity": 0,
        }

    model = xgb.XGBRegressor(**params)
    model.fit(train_df[FEATURE_COLS], train_df[TARGET_COL])

    preds = model.predict(test_df[FEATURE_COLS])
    return preds

def walk_forward_xgboost(
    df: pd.DataFrame,
    min_train_size: int = 500,
    step_size: int = 21,
    params: dict = None,
) -> pd.DataFrame:
    """Walk-forward validation for XGBoost, same expanding-window approach as LightGBM."""
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    n = len(df)

    predictions = []

    train_end = min_train_size
    while train_end < n:
        test_end = min(train_end + step_size, n)

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]

        preds = fit_predict_xgboost(train_df, test_df, params=params)

        result_chunk = test_df[["Date", TARGET_COL]].copy()
        result_chunk["prediction"] = preds
        predictions.append(result_chunk)

        train_end = test_end

    return pd.concat(predictions, ignore_index=True)


def main():
    df = pd.read_csv("data/processed/ml_features.csv", parse_dates=["Date"])

    results = walk_forward_xgboost(df)

    output_path = Path("data/processed/xgboost_predictions.csv")
    results.to_csv(output_path, index=False)

    print(f"Saved XGBoost walk-forward predictions to {output_path} ({len(results)} rows)")
    print(results.head(10))


if __name__ == "__main__":
    main()