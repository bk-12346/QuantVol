import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression


def build_har_features(df: pd.DataFrame, vol_col: str = "realized_vol_21d") -> pd.DataFrame:
    """
    Build HAR-RV features: daily, weekly, monthly lagged realized volatility.

    All features at time t use only information available at time t
    (i.e., they are lagged by construction - no future data used).
    The target is the vol_col value shifted forward, so the model
    trained on features up to t-1 predicts the value at t.
    """
    df = df.copy()

    # Daily component: yesterday's RV (1-day lag)
    df["har_daily"] = df[vol_col].shift(1)

    # Weekly component: average RV over the past 5 days (lagged, so days t-5 to t-1)
    df["har_weekly"] = df[vol_col].shift(1).rolling(window=5, min_periods=5).mean()

    # Monthly component: average RV over the past 21 days (lagged)
    df["har_monthly"] = df[vol_col].shift(1).rolling(window=21, min_periods=21).mean()

    # Target: the actual RV value at time t (what we're trying to predict)
    df["har_target"] = df[vol_col]

    return df

def walk_forward_har_rv(
    df: pd.DataFrame,
    feature_cols: list[str] = ["har_daily", "har_weekly", "har_monthly"],
    target_col: str = "har_target",
    min_train_size: int = 500,
    step_size: int = 21,
) -> pd.DataFrame:
    """
    Walk-forward validation for HAR-RV.

    Starts with `min_train_size` rows of training data, fits a model,
    predicts the next `step_size` rows, then expands the training
    window to include those rows and repeats - until the data runs out.

    This mimics how the model would actually be used in practice:
    retrained periodically on all data available up to that point.
    """
    df = df.dropna(subset=feature_cols + [target_col]).reset_index()
    n = len(df)

    predictions = []

    train_end = min_train_size
    while train_end < n:
        test_end = min(train_end + step_size, n)

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]

        model = LinearRegression()
        model.fit(train_df[feature_cols], train_df[target_col])

        preds = model.predict(test_df[feature_cols])

        result_chunk = test_df[["Date", target_col]].copy()
        result_chunk["prediction"] = preds
        predictions.append(result_chunk)

        train_end = test_end

    return pd.concat(predictions, ignore_index=True)

def main():
    from quantvol.data.ingest import load_config

    config = load_config()
    processed_path = Path(config["output_paths"]["processed"])

    df = pd.read_csv(processed_path, parse_dates=["Date"])

    df = build_har_features(df, vol_col="realized_vol_21d")

    results = walk_forward_har_rv(df)

    output_path = Path("data/processed/har_rv_predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"Saved HAR-RV walk-forward predictions to {output_path} ({len(results)} rows)")
    print(results.head(10))


if __name__ == "__main__":
    main()