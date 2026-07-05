import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb


FEATURE_COLS = [
    "rv_lag_1d", "rv_lag_5d", "rv_lag_10d", "rv_lag_21d", "rv_lag_63d",
    "parkinson_vol_21d_lag1", "garman_klass_vol_21d_lag1",
    "return_lag1", "abs_return_lag1",
    "return_skew_21d", "return_kurt_21d",
]

TARGET_COL = "target"


def fit_predict_lgbm(train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict = None) -> np.ndarray:
    """Fit a LightGBM regressor on train_df and predict on test_df."""
    if params is None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 5,
            "num_leaves": 15,
            "min_child_samples": 20,
            "verbose": -1,
        }

    model = lgb.LGBMRegressor(**params)
    model.fit(train_df[FEATURE_COLS], train_df[TARGET_COL])

    preds = model.predict(test_df[FEATURE_COLS])
    return preds

def walk_forward_lgbm(
    df: pd.DataFrame,
    min_train_size: int = 500,
    step_size: int = 21,
    params: dict = None,
) -> pd.DataFrame:
    """
    Walk-forward validation for LightGBM, using the same expanding-window
    approach as HAR-RV and GARCH for a fair, apples-to-apples comparison.
    """
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    n = len(df)

    predictions = []

    train_end = min_train_size
    while train_end < n:
        test_end = min(train_end + step_size, n)

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]

        preds = fit_predict_lgbm(train_df, test_df, params=params)

        result_chunk = test_df[["Date", TARGET_COL]].copy()
        result_chunk["prediction"] = preds
        predictions.append(result_chunk)

        train_end = test_end

    return pd.concat(predictions, ignore_index=True)

def main():
    df = pd.read_csv("data/processed/ml_features.csv", parse_dates=["Date"])

    # Note: Optuna tuning (see quantvol.tuning.optuna_search) was tested but
    # did not generalize well across the full walk-forward period - likely
    # due to regime-dependent optimal complexity in a single train/val split.
    # Using untuned defaults, which performed better across the full walk-forward evaluation.
    results = walk_forward_lgbm(df, params=None)

    output_path = Path("data/processed/lgbm_predictions.csv")
    results.to_csv(output_path, index=False)

    print(f"Saved LightGBM walk-forward predictions to {output_path} ({len(results)} rows)")
    print(results.head(10))


if __name__ == "__main__":
    main()