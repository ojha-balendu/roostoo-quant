"""Volume-weighted average prices from klines: VWAP = USD traded / coins traded."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bar_vwap(quote_volume: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """VWAP of each bar; NaN where nothing traded."""
    return (quote_volume / volume.where(volume > 0)).replace([np.inf, -np.inf], np.nan)


def resample_vwap(quote_volume: pd.DataFrame, volume: pd.DataFrame, rule: str) -> pd.DataFrame:
    """VWAP over longer bars: total USD traded / total coins traded in each bar (not an average of VWAPs)."""
    q = quote_volume.resample(rule).sum(min_count=1)
    v = volume.resample(rule).sum(min_count=1)
    return bar_vwap(q, v)
