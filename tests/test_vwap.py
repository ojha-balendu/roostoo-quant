import numpy as np
import pandas as pd

from quantbot.analysis.vwap import bar_vwap, resample_vwap


def test_vwap_is_dollars_over_coins_and_aggregates_by_totals():
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    coins = pd.DataFrame({"A": [1.0, 3.0, 0.0, 2.0]}, index=idx)
    dollars = pd.DataFrame({"A": [100.0, 330.0, 0.0, 240.0]}, index=idx)
    v = bar_vwap(dollars, coins)
    assert list(v["A"].iloc[[0, 1, 3]]) == [100.0, 110.0, 120.0] and np.isnan(v["A"].iloc[2])
    two = resample_vwap(dollars, coins, "2h")
    assert two["A"].iloc[0] == 430.0 / 4.0                               # not the mean of 100 and 110
    assert two["A"].iloc[1] == 120.0                                     # the empty hour adds nothing
