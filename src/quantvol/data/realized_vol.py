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

# The concept: Parkinson's estimator uses the daily High/Low range instead of close-to-close returns. Intuitively: if a stock only trades between $100.10 and $100.20 all day, that's a much calmer day than one where it swings between $95 and $105 — even if it happens to close near where it opened. Close-to-close returns miss this intraday information entirely; Parkinson captures it.
# Formula (per day): Parkinson variance = (1 / (4 * ln(2))) * (ln(High/Low))^2

def compute_parkinson_volatility(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Compute rolling Parkinson volatility estimator (annualized).

    Uses the daily High/Low range, which captures intraday movement
    that close-to-close returns miss. More statistically efficient
    than close-to-close vol for the same sample size.
    """
    df = df.copy()
    trading_days_per_year = 252

    daily_estimate = (1.0 / (4.0 * np.log(2))) * (np.log(df["High"] / df["Low"])) ** 2

    df[f"parkinson_vol_{window}d"] = np.sqrt(
        daily_estimate.rolling(window=window, min_periods=window).mean()
        * trading_days_per_year
    )

    return df

# The concept: Garman-Klass extends Parkinson by also incorporating Open and Close, not just High/Low. It's considered even more statistically efficient than Parkinson because it uses all four price points available in daily OHLC data, capturing both intraday range and the close-to-open jump.
# Formula (per day): GK variance = 0.5 * (ln(High/Low))^2 - (2*ln(2) - 1) * (ln(Close/Open))^2

def compute_garman_klass_volatility(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Compute rolling Garman-Klass volatility estimator (annualized).

    Extends Parkinson by also using Open/Close, capturing both
    intraday range and the close-to-open jump. Generally the most
    statistically efficient of the simple range-based estimators.
    """
    df = df.copy()
    trading_days_per_year = 252

    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Close"] / df["Open"])

    daily_estimate = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)

    df[f"garman_klass_vol_{window}d"] = np.sqrt(
        daily_estimate.rolling(window=window, min_periods=window).mean()
        * trading_days_per_year
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
    df = compute_parkinson_volatility(df, window=21)
    df = compute_garman_klass_volatility(df, window=21)

    save_processed_data(df, config)


if __name__ == "__main__":
    main()