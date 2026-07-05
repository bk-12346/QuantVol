import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from quantvol.data.realized_vol import compute_log_returns, compute_realized_volatility


def make_dummy_price_df(n=50, seed=42):
    """Build a small synthetic price series for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n)))
    df = pd.DataFrame({"Close": prices}, index=dates)
    df.index.name = "Date"
    return df


def test_realized_vol_no_leakage():
    """
    Changing a price AFTER day t should not change the realized_vol
    value computed AT day t. If it does, the rolling window is
    leaking future information.
    """
    df = make_dummy_price_df(n=50)
    df = compute_log_returns(df)
    df = compute_realized_volatility(df, window=21)

    # Take the vol value at day 30 (well past the 21-day warm-up)
    original_vol_at_30 = df["realized_vol_21d"].iloc[30]

    # Now corrupt all prices AFTER day 30 with huge noise
    df_modified = make_dummy_price_df(n=50)
    df_modified.iloc[31:, df_modified.columns.get_loc("Close")] *= 5.0  # huge shock, future only

    df_modified = compute_log_returns(df_modified)
    df_modified = compute_realized_volatility(df_modified, window=21)

    new_vol_at_30 = df_modified["realized_vol_21d"].iloc[30]

    assert np.isclose(original_vol_at_30, new_vol_at_30), (
        "Realized volatility at day 30 changed after modifying only future prices — "
        "this indicates look-ahead leakage in the rolling window calculation."
    )

from quantvol.features.regime_labels import tag_regimes

def test_regime_tagging_boundaries():
    """
    Verify regime tagging correctly includes boundary dates and
    correctly excludes dates just outside the defined range.
    """
    dates = pd.date_range("2020-02-10", "2020-05-05", freq="B")
    df = pd.DataFrame({"Close": np.linspace(100, 110, len(dates))}, index=dates)
    df.index.name = "Date"

    config = {
        "regimes": [
            {"name": "2020_covid_crash", "start": "2020-02-15", "end": "2020-04-30"}
        ]
    }

    df = tag_regimes(df, config)

    # Just before the regime starts -> should be "normal"
    assert df.loc["2020-02-14":"2020-02-14", "regime"].eq("normal").all() if "2020-02-14" in df.index else True

    # Exactly on the start boundary -> should be tagged
    start_date = pd.Timestamp("2020-02-17")  # first business day on/after 2020-02-15 (a Saturday)
    assert df.loc[start_date, "regime"] == "2020_covid_crash"

    # Exactly on the end boundary -> should still be tagged (inclusive)
    end_date = pd.Timestamp("2020-04-30")
    assert df.loc[end_date, "regime"] == "2020_covid_crash"

    # Just after the regime ends -> should be "normal"
    after_date = pd.Timestamp("2020-05-01")
    assert df.loc[after_date, "regime"] == "normal"