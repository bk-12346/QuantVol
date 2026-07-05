# tags known regime periods

import pandas as pd
from pathlib import Path


def tag_regimes(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add a 'regime' column based on known historical periods defined in config.
    Any date not falling into a defined regime is labeled 'normal'.
    """
    df = df.copy()
    df["regime"] = "normal"

    for regime in config["regimes"]:
        mask = (df.index >= regime["start"]) & (df.index <= regime["end"])
        df.loc[mask, "regime"] = regime["name"]

    return df

def main():
    from quantvol.data.ingest import load_config

    config = load_config()

    processed_path = Path(config["output_paths"]["processed"])
    df = pd.read_csv(processed_path, parse_dates=["Date"], index_col="Date")

    df = tag_regimes(df, config)

    df.to_csv(processed_path)
    print(f"Updated {processed_path} with regime labels ({len(df)} rows)")
    print()
    print("Regime counts:")
    print(df["regime"].value_counts())


if __name__ == "__main__":
    main()