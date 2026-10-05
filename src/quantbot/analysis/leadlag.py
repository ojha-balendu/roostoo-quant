"""Lead-lag measurement: does a leader's (BTC's) return predict other coins' next returns?"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


def to_returns(closes: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Log returns at `freq` from a (finer) close panel. '1m' on 1m data returns them as-is."""
    px = closes if freq in ("1m", "1min") else closes.resample(freq).last()
    return np.log(px).diff()


def cross_correlations(r: pd.DataFrame, leader: str, max_lag: int = 5) -> pd.DataFrame:
    """corr(r_coin[t], r_leader[t - lag]) per coin (rows) and lag (columns).
    Positive lag = leader's PAST return vs coin's current return (leader leads)."""
    b = r[leader]
    out = {lag: r.corrwith(b.shift(lag)) for lag in range(-max_lag, max_lag + 1)}
    return pd.DataFrame(out).drop(index=leader)


def lead_regression(r: pd.DataFrame, leader: str) -> pd.DataFrame:
    """Per coin: r_coin[t] = a + b * r_leader[t-1] + c * r_coin[t-1].
    Returns b (lead coefficient), its heteroskedasticity-robust (HC0) t-stat, and c."""
    rows = {}
    lead = r[leader].shift(1)
    for coin in r.columns.drop(leader):
        df = pd.concat([r[coin], lead, r[coin].shift(1)], axis=1).dropna()
        if len(df) < 100:
            continue
        y = df.iloc[:, 0].to_numpy()
        X = np.column_stack([np.ones(len(df)), df.iloc[:, 1].to_numpy(), df.iloc[:, 2].to_numpy()])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ coef
        xtx_inv = np.linalg.inv(X.T @ X)
        meat = (X * (e**2)[:, None]).T @ X
        se = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv))
        rows[coin] = {"lead_b": coef[1], "lead_t": coef[1] / se[1], "own_c": coef[2], "n": len(df)}
    return pd.DataFrame(rows).T


def catch_up(r: pd.DataFrame, leader: str, quantile: float = 0.95, horizon: int = 1) -> pd.DataFrame:
    """After the leader's largest moves (top 1-quantile by size), how far does each coin
    move in the leader's direction over the next `horizon` bars? Returned in basis points,
    with the hit rate (share of events where the coin followed)."""
    b = r[leader]
    big = b.abs() >= b.abs().quantile(quantile)
    direction = np.sign(b)
    rows = {}
    for coin in r.columns.drop(leader):
        fwd = r[coin].rolling(horizon).sum().shift(-horizon)
        signed = (direction * fwd)[big].dropna()
        if len(signed) < 20:
            continue
        rows[coin] = {"catchup_bps": signed.mean() * 1e4, "hit_rate": (signed > 0).mean(), "events": len(signed)}
    return pd.DataFrame(rows).T


def stale_share(closes_1m: pd.DataFrame) -> pd.Series:
    """Share of 1-minute bars whose close equals the previous close (price did not update)."""
    d = closes_1m.diff()
    return (d == 0).sum() / d.notna().sum()


# --------------------------------------------------------------------------- tradability
def move_threshold(leader_returns: pd.Series, quantile: float, lookback_bars: int) -> pd.Series:
    """Size of a 'big move': the quantile of |return| over the previous lookback_bars (no lookahead)."""
    return (leader_returns.abs().rolling(lookback_bars, min_periods=lookback_bars // 2)
            .quantile(quantile).shift(1))


def event_returns(
    r: pd.DataFrame,
    leader: str,
    coin: str,
    quantile: float,
    horizon: int,
    lookback_bars: int,
    threshold: pd.Series | None = None,
) -> pd.Series:
    """Signed forward returns of `coin` after big `leader` moves, one trade at a time.

    A 'big move' is a bar whose absolute leader return exceeds the `quantile` of the
    previous `lookback_bars` bars (no future information). The trade is in the leader's
    direction, from the close of the move bar to `horizon` bars later. Events that would
    overlap an open trade are skipped.
    """
    b = r[leader]
    if threshold is None:
        threshold = move_threshold(b, quantile, lookback_bars)
    fwd = r[coin].rolling(horizon).sum().shift(-horizon)
    signed = np.sign(b) * fwd
    candidates = np.flatnonzero((b.abs() > threshold).to_numpy() & signed.notna().to_numpy())
    keep, next_free = [], -1
    for i in candidates:
        if i > next_free:
            keep.append(i)
            next_free = i + horizon - 1
    return signed.iloc[keep]


@dataclass(frozen=True)
class TradeRules:
    cost_bps: float = 25.0       # round trip: 2 x 0.1% fee + ~5 bps spread
    min_net_bps: float = 5.0     # required profit per trade after cost
    min_t: float = 3.0           # statistical confidence
    min_events: int = 30
    min_events_per_half: int = 10
    max_stale: float = 0.10      # share of 1m bars with no price change


def tradability(
    r: pd.DataFrame,
    leader: str,
    *,
    freq_label: str,
    horizons: Sequence[int],
    quantiles: Sequence[float],
    lookback_bars: int,
    split: pd.Timestamp,
    stale: pd.Series,
    rules: TradeRules,
) -> pd.DataFrame:
    """One row per (coin, quantile, horizon) with the stats and pass/fail of each rule."""
    years = (r.index.max() - r.index.min()).total_seconds() / (365 * 86400)
    rows = []
    thresholds = {q: move_threshold(r[leader], q, lookback_bars) for q in quantiles}
    for coin in r.columns.drop(leader):
        for q in quantiles:
            for h in horizons:
                ev = event_returns(r, leader, coin, q, h, lookback_bars, threshold=thresholds[q]) * 1e4
                if len(ev) < 2:
                    continue
                first, second = ev[ev.index < split], ev[ev.index >= split]
                sd = ev.std()
                net = ev.mean() - rules.cost_bps
                row = {
                    "coin": coin, "freq": freq_label, "btc_move": f"top {100 * (1 - q):g}%", "hold_bars": h,
                    "trades": len(ev), "trades_per_year": len(ev) / years if years > 0 else np.nan,
                    "gross_bps": ev.mean(), "net_bps": net,
                    "t": (net / (sd / np.sqrt(len(ev)))) if sd > 0 else np.nan,
                    "hit_rate": float((ev > rules.cost_bps).mean()),
                    "net_first_half": first.mean() - rules.cost_bps if len(first) else np.nan,
                    "net_second_half": second.mean() - rules.cost_bps if len(second) else np.nan,
                    "stale": float(stale.get(coin, np.nan)),
                }
                row["pass_profit"] = row["net_bps"] >= rules.min_net_bps
                row["pass_confidence"] = row["t"] >= rules.min_t
                row["pass_sample"] = (len(ev) >= rules.min_events and len(first) >= rules.min_events_per_half
                                      and len(second) >= rules.min_events_per_half)
                row["pass_stable"] = row["net_first_half"] > 0 and row["net_second_half"] > 0
                row["pass_liquid"] = row["stale"] <= rules.max_stale
                row["tradeable"] = all(row[k] for k in ("pass_profit", "pass_confidence", "pass_sample",
                                                        "pass_stable", "pass_liquid"))
                rows.append(row)
    return pd.DataFrame(rows)
