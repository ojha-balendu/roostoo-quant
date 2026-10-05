"""Per-coin features for the agent, computed only from past prices.

For each coin at each 15-minute bar t (using closes up to and including bar t):
  * trend over 15m, 1h, 4h, 1d, 4d: log return over L bars / (7-day volatility * sqrt(L)), clipped to +/-5
  * volatility regime: log(1-day volatility / 7-day volatility), clipped to +/-3
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACKS = (1, 4, 16, 96, 384)
VOL_BARS = 96 * 7
SHORT_VOL_BARS = 96
N_FEATURES = len(LOOKBACKS) + 1
WARMUP_BARS = max(LOOKBACKS) + VOL_BARS


def build_features(closes: pd.DataFrame, lookbacks: tuple[int, ...] = LOOKBACKS) -> tuple[np.ndarray, np.ndarray]:
    """Returns (features [T, N, len(lookbacks) + 1] float32, valid [T, N] bool). Invalid entries are zero."""
    lp = np.log(closes)
    r = lp.diff()
    vol = r.rolling(VOL_BARS, min_periods=VOL_BARS // 2).std()
    feats = []
    for L in lookbacks:
        feats.append(((lp - lp.shift(L)) / (vol * np.sqrt(L))).clip(-5, 5))
    short_vol = r.rolling(SHORT_VOL_BARS, min_periods=SHORT_VOL_BARS // 2).std()
    feats.append(np.log(short_vol / vol).clip(-3, 3))
    stacked = np.stack([f.to_numpy(dtype=float) for f in feats], axis=-1)
    valid = np.isfinite(stacked).all(axis=-1) & np.isfinite(closes.to_numpy(dtype=float))
    stacked = np.where(valid[..., None], np.nan_to_num(stacked), 0.0)
    return stacked.astype(np.float32), valid
