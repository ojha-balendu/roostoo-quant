"""Composite signal: weighted sum of normalised indicators, then an affine map to [-1, 1] across coins.

Per bar t, over the coins eligible at t:
  1. each indicator is normalised across coins (default: z-score, clipped at +/-3; "rank": percentile
     rank - 0.5; "raw": untouched, which lets large-unit indicators such as volume dominate)
  2. raw signal s = sum_k coeff_k * normalised_k
  3. s' = 2 * (s - min s) / (max s - min s) - 1, so the lowest coin is -1 and the highest +1
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COEFFS: dict[str, float] = {
    "relative_strength_index": -1.0, "williams_r": -1.0, "rsi": -1.0, "volatility_20": 1.0, "volatility_60": 1.0,
    "trend_1_3": 1.0, "trend_5_20": 1.0, "trend_20_60": -1.0, "average_true_range": -1.0, "macd": 1.0,
    "macd_signal": 1.0, "macd_histogram": -1.0, "trix": 1.0, "commodity_channel_index": 7.3160,
    "chande_momentum_oscillator": -1.0, "ichimoku_conversion": -1.0, "ichimoku_base": -1.0,
    "ichimoku_leading_a": -1.0, "ichimoku_leading_b": -1.0, "know_sure_thing": 1.0, "ultimate_oscillator": -1.0,
    "aroon_up": -1.0, "aroon_down": 1.0, "aroon_oscillator": -1.0, "stochastic_k": -1.0, "stochastic_d": -1.0,
    "on_balance_volume": 1.0, "ease_of_movement": 1.0, "chaikin_money_flow": 1.1280,
    "accumulation_distribution_index": -1.0, "volume": -8.2977,
}


def normalise(x: pd.DataFrame, eligible: pd.DataFrame, method: str = "zscore", clip: float = 3.0) -> pd.DataFrame:
    x = x.where(eligible)
    if method == "raw":
        return x
    if method == "rank":
        return x.rank(axis=1, pct=True) - 0.5
    if method != "zscore":
        raise ValueError("normalise method must be zscore, rank or raw")
    sd = x.std(axis=1)
    z = x.sub(x.mean(axis=1), axis=0).div(sd.where(sd > 0), axis=0)
    return z.clip(-clip, clip)


def composite(named: dict[str, pd.DataFrame], eligible: pd.DataFrame, coeffs: dict[str, float] = COEFFS,
              method: str = "zscore") -> pd.DataFrame:
    """Raw weighted signal; an indicator missing for a coin counts as average (0) when normalised."""
    missing = sorted(set(coeffs) - set(named))
    if missing:
        raise ValueError(f"indicators missing for the composite: {missing}")
    total = None
    for k, w in coeffs.items():
        term = normalise(named[k], eligible, method)
        term = term.fillna(0.0) if method != "raw" else term
        total = w * term if total is None else total + w * term
    return total.where(eligible)


def affine_unit(s: pd.DataFrame) -> pd.DataFrame:
    """Map each row linearly so its min is -1 and its max is +1 (rows with < 2 distinct values -> NaN)."""
    mn, mx = s.min(axis=1), s.max(axis=1)
    rng = (mx - mn).where(mx > mn)
    return s.sub(mn, axis=0).div(rng, axis=0) * 2 - 1


def term_shares(named: dict[str, pd.DataFrame], eligible: pd.DataFrame, coeffs: dict[str, float] = COEFFS,
                method: str = "zscore", every: int = 24) -> pd.Series:
    """How much of the composite's cross-sectional variation each weighted term drives (shares sum to 1).
    Measured as the average absolute covariance of each term with the total, on a sample of bars."""
    rows = eligible.index[::every]
    elig = eligible.loc[rows]
    terms = {k: w * normalise(named[k].loc[rows], elig, method).fillna(0.0 if method != "raw" else np.nan)
             for k, w in coeffs.items()}
    total = sum(terms.values())
    tot_c = total.sub(total.mean(axis=1), axis=0)
    cov = {k: (t.sub(t.mean(axis=1), axis=0) * tot_c).mean(axis=1).abs().mean() for k, t in terms.items()}
    s = pd.Series(cov)
    return (s / s.sum()).sort_values(ascending=False)
