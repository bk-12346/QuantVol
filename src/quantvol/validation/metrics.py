# QLIKE, RMSE, regime-segmented eval

import numpy as np
import pandas as pd


def compute_rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((actual - predicted) ** 2))


def compute_qlike(actual: pd.Series, predicted: pd.Series) -> float:
    """
    QLIKE loss - the standard evaluation metric in volatility forecasting
    literature. Penalizes under-prediction more heavily than over-prediction,
    which aligns with real-world risk management priorities.

    Note: operates on variance, not vol directly - squares both inputs first.
    """
    actual_var = actual ** 2
    predicted_var = predicted ** 2

    ratio = actual_var / predicted_var
    qlike = ratio - np.log(ratio) - 1

    return np.mean(qlike)

def evaluate_predictions(pred_path: str, model_name: str, target_col: str) -> dict:
    """Load a saved predictions CSV and compute RMSE + QLIKE against the target."""
    df = pd.read_csv(pred_path, parse_dates=["Date"])

    actual = df[target_col]
    predicted = df["prediction"]

    rmse = compute_rmse(actual, predicted)
    qlike = compute_qlike(actual, predicted)

    return {
        "model": model_name,
        "rmse": rmse,
        "qlike": qlike,
        "n_predictions": len(df),
    }


def main():
    har_results = evaluate_predictions(
        "data/processed/har_rv_predictions.csv",
        model_name="HAR-RV",
        target_col="har_target",
    )

    garch_results = evaluate_predictions(
        "data/processed/garch_predictions.csv",
        model_name="GARCH(1,1)",
        target_col="realized_vol_21d",
    )

    lgbm_results = evaluate_predictions(
        "data/processed/lgbm_predictions.csv",
        model_name="LightGBM",
        target_col="target",
    )

    xgb_results = evaluate_predictions(
        "data/processed/xgboost_predictions.csv",
        model_name="XGBoost",
        target_col="target",
    )

    comparison = pd.DataFrame([har_results, garch_results, lgbm_results, xgb_results])
    print(comparison.to_string(index=False))

    output_path = "data/processed/baseline_comparison.csv"
    comparison.to_csv(output_path, index=False)
    print(f"\nSaved comparison to {output_path}")


if __name__ == "__main__":
    main()