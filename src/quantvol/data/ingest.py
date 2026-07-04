# price data (yfinance etc.)

import yaml
import yfinance as yf
import pandas as pd
from pathlib import Path

def load_config(config_path: str = "configs/data_config.yaml") -> dict:
    """ Load the data configuration YAML file """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)      # reads the YAML file into a plain Python dict, so config["asset"]["ticker"] gives "^GSPC"
    
    return config

# Data-pulling function
def fetch_price_data(config: dict) -> pd.DataFrame:
    """ Download historical OHLCV price data based on the config """
    ticker = config["asset"]["ticker"]
    start = config["date_range"]["start"]
    end = config["date_range"]["end"]

    df = yf.download(
        ticker,
        start = start,
        end = end,
        interval = config["frequency"],
        auto_adjust = False,             #keep raw Close separate from Adj Close
        progress = False
    )

    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}. Check the ticker symbol or date range.")
    
    # yfinance returns multi-level columns (field, ticker) even for a single ticker.
    # Flatten to just the field name (Open, High, Low, Close, Adj Close, Volume).
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    
    return df

def save_raw_data(df: pd.DataFrame, config: dict) -> None:
    """ Save the raw downloaded data to the configured output path """
    output_path = Path(config["output_paths"]["raw"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"Saved the raw data to {output_path} ({len(df)}) rows")

def main():
    print("STARTING SCRIPT")
    config = load_config()
    df = fetch_price_data(config)
    save_raw_data(df, config)

if __name__ == "__main__":
    main()