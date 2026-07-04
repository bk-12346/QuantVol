import numpy as np
import pandas as pd
from pathlib import Path

def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """ Compute daily log returns from the Close price. """

    df = df.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))   # shift(1) gives you yesterday's close, so log_return on day t = ln(Close_t / Close_{t-1}) — this uses only past and current-day information, no future leakage

    return df

### Key design decision — annualization and window size:
# Using a 21-trading-day rolling window (roughly one calendar month) — this is a very standard choice in vol forecasting literature (HAR-RV models specifically use daily/weekly/monthly windows: 1, 5, 21 days).
# Computing it as the rolling standard deviation of log returns, then annualize by multiplying by sqrt(252) (252 = approximate trading days per year) — annualizing is the market convention, so a "20% volatility" number means something comparable across assets and to things like the VIX.

def compute_realized_volatility(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Compute rolling realized volatility (annualized) from log returns.

    Uses a trailing window -> the value at time t only uses returns from t-window+1 through t.
    No future information is used.
    """

    df = df.copy()
    trading_days_year = 252

    df[f"realized_vol_{window}d"] = (
        df["log_return"]
        .rolling(window=window, min_periods=window)     # ensures a NaN for the first 20 rows (with a 21-day window) rather than a partially-computed, noisy estimate from only a handful of days
        .std()                                          # standard deviation of returns within that window = volatility
        *np.sqrt(trading_days_year)
    )
    
    return df

def save_processed_data(df: pd.DataFrame, config: dict) -> None:
    """ Save the processed data (returns + realized vol) to the processed path. """

    output_path = Path(config["output_paths"].get("processed", "data/processed/gspc_features.csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path)
    print(f"Saved processed data to {output_path} ({len(df)} rows)")

def main():
    from quantvol.data.ingest import load_config

    config = load_config()

    interim_path = Path(config["output_paths"]["interim"])
    df = pd.read_csv(interim_path, parse_dates=["Date"], index_col="Date")

    df = compute_log_returns(df)
    df = compute_realized_volatility(df, window=21)

    save_processed_data(df, config)


if __name__ == "__main__":
    main()