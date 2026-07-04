import pandas as pd
from pathlib import Path


def load_raw_data(config: dict) -> pd.DataFrame:
    """ Load the DataFrame and parse the Date column properly """

    raw_path = Path(config["output_paths"]["raw"])
    df = pd.read_csv(raw_path, parse_dates=["Date"])    # convert the Date column to actual datetime64 format on read, not leave it as a string
    df = df.set_index("Date")

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sorts chronologically, checks for duplication and unexpected gaps """

    # 1. Enforce chronological order
    df = df.sort_index()

    # 2. Check for duplicate dates
    n_duplicates = df.index.duplicated().sum()
    if n_duplicates > 0:
        raise ValueError(f"Found {n_duplicates} duplicate dates in the data.")

    # 3. Check for unexpected gaps in the data
    # 1-3 days is normal, check for anything greater
    date_diffs = df.index.to_series().diff().dt.days
    large_gaps = date_diffs[date_diffs > 5]

    if not large_gaps.empty:
        print("Warning! Found larger-than-expected gaps in the data.")
        print(large_gaps)

    # 4. Drop AdjClose as it is identical to Close for an index - no dividents or splits
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])

    return df

def save_interim_data(df: pd.DataFrame, config: dict) -> None:
    """ Save the cleaned data to the configured interim path. """

    output_path = Path(config["output_paths"]["interim"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path)
    print(f"Saved the cleaned data to {output_path} {len(df)} rows.")

def main():
    from quantvol.data.ingest import load_config

    config = load_config()
    df = load_raw_data(config)
    df = clean_data(df)

    save_interim_data(df, config)

if __name__ == "__main__":
    main()

