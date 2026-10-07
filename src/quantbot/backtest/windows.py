"""Evaluate a strategy over many fixed-length windows (e.g. every 14 days of the past year).

Each window is an independent competition: starts flat with the same capital, pays its own
entry fees, and is scored on its own equity curve. Weights are computed once on the full
history beforehand, which is safe because strategies only use past data (tested per strategy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantbot.backtest.engine import run_backtest


def window_bounds(index: pd.DatetimeIndex, window_days: float, step_days: float | None = None,
                  start: pd.Timestamp | None = None) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Windows ending at the last bar and stepping backwards; returned oldest first."""
    window = pd.Timedelta(days=window_days)
    step = pd.Timedelta(days=step_days or window_days)
    first = max(index.min(), start) if start is not None else index.min()
    bounds, w_end = [], index.max() + (index[1] - index[0] if len(index) > 1 else pd.Timedelta(0))
    while w_end - window >= first:
        bounds.append((w_end - window, w_end))
        w_end -= step
    return bounds[::-1]


def _hourly_beta(equity: pd.Series, bench: pd.Series) -> float:
    df = pd.concat([equity.resample("1h").last().pct_change(), bench.resample("1h").last().pct_change()],
                   axis=1).dropna()
    if len(df) < 10 or df.iloc[:, 1].var() == 0:
        return float("nan")
    return float(df.iloc[:, 0].cov(df.iloc[:, 1]) / df.iloc[:, 1].var())


def evaluate_windows(
    closes: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    window_days: float = 14,
    step_days: float | None = None,
    start: pd.Timestamp | None = None,
    trade_on: pd.Series | None = None,
    benchmark: str = "BTC/USD",
    **backtest_kwargs,
) -> pd.DataFrame:
    """One row of metrics per window."""
    rows = []
    for ws, we in window_bounds(closes.index, window_days, step_days, start):
        sel = (closes.index >= ws) & (closes.index < we)
        c, w = closes.loc[sel], weights.loc[sel]
        if len(c) < 10:
            continue
        mask = None if trade_on is None else np.asarray(trade_on, dtype=bool)[sel]
        res = run_backtest(c, w, trade_on=mask, **backtest_kwargs)
        m = res.metrics
        rows.append({
            "start": ws, "end": we,
            "return": m.get("total_return", np.nan),
            "max_dd": m.get("max_drawdown", np.nan),
            "sharpe": m.get("sharpe", np.nan),
            "sortino": m.get("sortino", np.nan),
            "calmar": m.get("calmar", np.nan),
            "score": m.get("score", np.nan),
            "fees": m.get("fees_paid", 0.0),
            "trades": m.get("n_trades", 0),
            "avg_gross": float(w.abs().sum(axis=1).mean()),
            "beta_btc": _hourly_beta(res.equity, c[benchmark]) if benchmark in c else np.nan,
        })
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, versus: pd.DataFrame | None = None) -> dict[str, float]:
    """Distribution of window outcomes; `versus` (same windows) adds a head-to-head win rate."""
    r = results
    out = {
        "windows": len(r),
        "median_return": r["return"].median(),
        "mean_return": r["return"].mean(),
        "worst_return": r["return"].min(),
        "best_return": r["return"].max(),
        "pct_positive": (r["return"] > 0).mean(),
        "median_max_dd": r["max_dd"].median(),
        "worst_max_dd": r["max_dd"].min(),
        "median_sharpe": r["sharpe"].median(),
        "median_sortino": r["sortino"].median(),
        "median_calmar": r["calmar"].median() if "calmar" in r else np.nan,
        "median_score": r["score"].median() if "score" in r else np.nan,
        "mean_score": r["score"].mean() if "score" in r else np.nan,
        "avg_fees": r["fees"].mean(),
        "avg_gross": r["avg_gross"].mean(),
        "median_beta_btc": r["beta_btc"].median(),
    }
    if versus is not None and len(versus) == len(r):
        out["pct_beats_benchmark"] = float((r["return"].to_numpy() > versus["return"].to_numpy()).mean())
    return out
