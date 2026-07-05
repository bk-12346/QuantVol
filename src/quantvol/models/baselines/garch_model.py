import pandas as pd
import numpy as np
from pathlib import Path
from arch import arch_model


def fit_garch_forecast(returns_train: pd.Series, horizon: int = 1) -> float:
    """
    Fit a GARCH(1,1) model on training returns and forecast
    the next-period conditional variance, converted to annualized vol.
    """
    # arch_model expects returns in percentage points (not decimals) for numerical stability
    returns_pct = returns_train * 100

    model = arch_model(returns_pct, vol="Garch", p=1, q=1, dist="normal")
    fitted = model.fit(disp="off")

    forecast = fitted.forecast(horizon=horizon, reindex=False)
    variance_pct2 = forecast.variance.values[-1, -1]  # forecasted variance, in (pct)^2 units

    # Convert back from (percent)^2 to decimal^2, then annualize
    variance_decimal = variance_pct2 / (100 ** 2)
    annualized_vol = np.sqrt(variance_decimal * 252)

    return annualized_vol

def walk_forward_garch(
    df: pd.DataFrame,
    return_col: str = "log_return",
    target_col: str = "realized_vol_21d",
    min_train_size: int = 500,
    step_size: int = 21,
) -> pd.DataFrame:
    """
    Walk-forward validation for GARCH(1,1).

    At each step, fits GARCH on all returns available up to that point,
    forecasts the next `step_size` days one at a time (refitting isn't
    repeated within a step - the model forecasts iteratively forward
    using its own recursive variance equation), then expands the window.
    """
    df = df.dropna(subset=[return_col, target_col]).reset_index()
    n = len(df)

    predictions = []

    train_end = min_train_size
    while train_end < n:
        test_end = min(train_end + step_size, n)

        train_returns = df[return_col].iloc[:train_end]

        # Fit once per step, forecast multiple steps ahead using the fitted model
        returns_pct = train_returns * 100
        model = arch_model(returns_pct, vol="Garch", p=1, q=1, dist="normal")
        fitted = model.fit(disp="off")

        n_ahead = test_end - train_end
        forecast = fitted.forecast(horizon=n_ahead, reindex=False)
        variance_pct2 = forecast.variance.values[-1, :]  # array of n_ahead variance forecasts

        variance_decimal = variance_pct2 / (100 ** 2)
        annualized_vol = np.sqrt(variance_decimal * 252)

        test_df = df.iloc[train_end:test_end]
        result_chunk = test_df[["Date", target_col]].copy()
        result_chunk["prediction"] = annualized_vol

        predictions.append(result_chunk)
        train_end = test_end

    return pd.concat(predictions, ignore_index=True)

def main():
    from quantvol.data.ingest import load_config

    config = load_config()
    processed_path = Path(config["output_paths"]["processed"])

    df = pd.read_csv(processed_path, parse_dates=["Date"])

    results = walk_forward_garch(df)

    output_path = Path("data/processed/garch_predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"Saved GARCH walk-forward predictions to {output_path} ({len(results)} rows)")
    print(results.head(10))


if __name__ == "__main__":
    main()