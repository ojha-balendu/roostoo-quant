"""Signals from futures positioning: funding crowding and liquidation flushes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantbot.data.futures import asof

OI_TOLERANCE = pd.Timedelta(minutes=10)


def trailing_funding(funding: pd.DataFrame, index: pd.DatetimeIndex, window: str = "7D") -> pd.DataFrame:
    """Mean funding rate over the trailing window, using only settlements at or before each time."""
    mean = pd.DataFrame({c: funding[c].dropna().rolling(window).mean() for c in funding.columns})
    return asof(mean, index)


def residualise(signal: pd.DataFrame, regressors: list[pd.DataFrame], eligible: pd.DataFrame,
                min_coins: int = 8) -> pd.DataFrame:
    """Per row, the part of `signal` across coins not explained by the regressors (cross-sectional OLS)."""
    out = np.full(signal.shape, np.nan)
    Y = signal.where(eligible).to_numpy(dtype=float)
    Xs = [r.reindex(index=signal.index, columns=signal.columns).to_numpy(dtype=float) for r in regressors]
    for i in range(len(Y)):
        ok = np.isfinite(Y[i])
        for X in Xs:
            ok &= np.isfinite(X[i])
        if ok.sum() < max(min_coins, len(Xs) + 3):
            continue
        A = np.column_stack([np.ones(ok.sum())] + [X[i, ok] for X in Xs])
        coef, *_ = np.linalg.lstsq(A, Y[i, ok], rcond=None)
        out[i, ok] = Y[i, ok] - A @ coef
    return pd.DataFrame(out, index=signal.index, columns=signal.columns)


def bar_oi_change(oi: pd.DataFrame, index: pd.DatetimeIndex, bar: pd.Timedelta,
                  tolerance: pd.Timedelta = OI_TOLERANCE) -> pd.DataFrame:
    """Open-interest change over each bar [t, t + bar): OI(t + bar) / OI(t) - 1 (snapshots within tolerance)."""
    start = asof(oi, index, tolerance)
    end = asof(oi, index + bar, tolerance)
    end.index = index
    return end / start - 1


def flush_events(close: pd.DataFrame, oi_change: pd.DataFrame, funding_before: pd.DataFrame,
                 eligible: pd.DataFrame, *, sigma_mult: float = 3.0, oi_drop: float = 0.05,
                 need_positive_funding: bool = True, vol_bars: int = 720) -> dict[str, pd.DataFrame]:
    """Boolean panels per bar t (all known at the close of bar t):
      big_drop  bar return below -sigma_mult x trailing volatility (volatility excludes bar t)
      flush     big_drop AND open interest fell >= oi_drop during the bar AND (optionally) funding > 0 before it
      no_flush  big_drop AND open interest did NOT fall (change > -1%): the comparison group"""
    r = close.pct_change(fill_method=None)
    sigma = r.rolling(vol_bars, min_periods=vol_bars // 4).std().shift(1)
    big = (r < -sigma_mult * sigma) & eligible
    flush = big & (oi_change <= -oi_drop)
    if need_positive_funding:
        flush &= funding_before > 0
    no_flush = big & (oi_change > -0.01)
    return {"big_drop": big.fillna(False), "flush": flush.fillna(False), "no_flush": no_flush.fillna(False)}


def event_returns(events: pd.DataFrame, close: pd.DataFrame, btc: pd.Series, beta: pd.DataFrame,
                  hold: int, cost: float) -> pd.DataFrame:
    """One row per event, entering at the close of the event bar and exiting `hold` bars later. Events of
    the same coin that start before the previous one has ended are skipped (one trade at a time)."""
    fwd = close.shift(-hold) / close - 1
    btc_fwd = btc.shift(-hold) / btc - 1
    rows = []
    E = events.to_numpy(dtype=bool)
    for j, coin in enumerate(events.columns):
        next_free = -1
        for i in np.flatnonzero(E[:, j]):
            if i <= next_free:
                continue
            gross = fwd.iat[i, j]
            if not np.isfinite(gross):
                continue
            b = beta[coin].iat[i] if coin in beta and np.isfinite(beta[coin].iat[i]) else 1.0
            rows.append({"time": events.index[i], "coin": coin, "gross": gross, "net": gross - cost,
                         "market_adj": gross - b * btc_fwd.iat[i]})
            next_free = i + hold
    return pd.DataFrame(rows)


def day_clustered(ev: pd.DataFrame, col: str = "net") -> dict[str, float]:
    """Mean per event, and a t-statistic that treats each UTC day as one observation (events on the same
    crash day are not independent)."""
    if ev.empty:
        return {"events": 0, "days": 0, "mean": np.nan, "t_day": np.nan, "hit": np.nan}
    daily = ev.groupby(ev["time"].dt.floor("D"))[col].mean()
    n = len(daily)
    t = daily.mean() / (daily.std() / np.sqrt(n)) if n > 2 and daily.std() > 0 else np.nan
    return {"events": len(ev), "days": n, "mean": float(ev[col].mean()), "t_day": float(t),
            "hit": float((ev[col] > 0).mean())}
