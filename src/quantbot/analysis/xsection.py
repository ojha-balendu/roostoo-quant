"""Cross-sectional over/under-reaction test on beta-neutral (residual) returns.

Hypothesis: a coin's move relative to the market over the last F hours predicts its
relative move over the next H hours, either reversing (over-reaction) or continuing
(momentum). Tested as a long/short portfolio rebalanced every H hours:

  residual e[i,t] = r[i,t] - beta[i,t-1] * r[BTC,t]          (market move removed)
  signal   s[i,t] = sum of e[i] over (t-F, t] / (residual vol * sqrt(F))   (vol-scaled)
  outcome  y[i,t] = sum of e[i] over (t, t+H]
  spread        = mean y of top `quantile` of coins by signal - mean y of bottom `quantile`
                  ("momentum" orientation; reversal is the negative)
  cost          = turnover (coins entering/leaving each leg + BTC hedge changes) x one-way cost

Returns are per $1 in the long leg and $1 in the short leg. Everything at row t uses only
data up to the close of row t; trades happen at that close.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def residual_returns(closes: pd.DataFrame, leader: str = "BTC/USD", beta_hours: int = 720,
                     shrink: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hourly residual returns (leader removed) and the shrunk betas used (known at each row)."""
    r = np.log(closes).diff()
    b = r[leader]
    roll = dict(window=beta_hours, min_periods=beta_hours // 2)
    raw = r.rolling(**roll).cov(b).div(b.rolling(**roll).var(), axis=0)
    others = [c for c in r.columns if c != leader]
    prior = raw[others].mean(axis=1)
    beta = raw.mul(shrink).add(prior * (1 - shrink), axis=0)[others]
    resid = r[others] - beta.shift(1).mul(b, axis=0)
    return resid, beta


def eligibility(closes: pd.DataFrame, quote_volume: pd.DataFrame | None, *, min_history_hours: int = 720,
                min_usd_per_hour: float = 200_000, leader: str = "BTC/USD") -> pd.DataFrame:
    """Coins with enough history and enough trading volume at each hour (trailing data only)."""
    others = [c for c in closes.columns if c != leader]
    ok = closes[others].notna().rolling(min_history_hours, min_periods=1).sum() >= min_history_hours
    if quote_volume is not None:
        vol = quote_volume.reindex(index=closes.index, columns=others)
        ok &= vol.rolling(168, min_periods=24).median() >= min_usd_per_hour
    return ok


def spread_periods(resid: pd.DataFrame, beta: pd.DataFrame, eligible: pd.DataFrame, *, formation: int,
                   horizon: int, quantile: float = 0.2, cost_one_way_bps: float = 12.5, vol_hours: int = 168,
                   min_coins: int = 10, start: pd.Timestamp | None = None,
                   end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Past relative move (vol-scaled) as the signal; see signal_periods."""
    sigma = resid.rolling(vol_hours, min_periods=vol_hours // 2).std()
    signal = resid.rolling(formation, min_periods=formation).sum() / (sigma * np.sqrt(formation))
    return signal_periods(signal, resid, beta, eligible, horizon=horizon, quantile=quantile,
                          cost_one_way_bps=cost_one_way_bps, min_coins=min_coins, start=start, end=end)


def signal_periods(signal: pd.DataFrame, resid: pd.DataFrame, beta: pd.DataFrame, eligible: pd.DataFrame, *,
                   horizon: int, quantile: float = 0.2, cost_one_way_bps: float = 12.5, min_coins: int = 10,
                   start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Any signal, ranked across coins every `horizon` bars: one row per non-overlapping holding
    period with the momentum-oriented spread (top minus bottom), its cost, and the rank IC
    (Spearman correlation between signal and next-period residual return across coins)."""
    signal = signal.reindex(index=resid.index, columns=resid.columns)
    outcome = resid.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    rows_idx = np.arange(0, len(resid), horizon)
    times = resid.index[rows_idx]
    keep = np.ones(len(times), dtype=bool)
    if start is not None:
        keep &= times >= start
    if end is not None:
        keep &= times < end
    S, Y, E, B = (signal.to_numpy(dtype=float), outcome.to_numpy(dtype=float), eligible.to_numpy(dtype=bool),
                  beta.to_numpy(dtype=float))
    n_coins = S.shape[1]
    prev_w, prev_h = np.zeros(n_coins), 0.0
    out = []
    for i, t in zip(rows_idx[keep], times[keep], strict=True):
        ok = E[i] & np.isfinite(S[i]) & np.isfinite(Y[i]) & np.isfinite(B[i])
        idx = np.flatnonzero(ok)
        if len(idx) < min_coins:
            prev_w, prev_h = np.zeros(n_coins), 0.0
            continue
        k = max(1, int(round(len(idx) * quantile)))
        order = idx[np.argsort(S[i, idx], kind="stable")]
        bottom, top = order[:k], order[-k:]
        w = np.zeros(n_coins)
        w[top], w[bottom] = 1.0 / k, -1.0 / k
        hedge = -np.nansum(w * B[i])  # BTC needed to keep the book beta-neutral
        traded = np.abs(w - prev_w).sum() + abs(hedge - prev_h)
        rs, ry = _ranks(S[i, idx]), _ranks(Y[i, idx])
        ic = float(np.corrcoef(rs, ry)[0, 1]) if rs.std() > 0 and ry.std() > 0 else np.nan
        out.append({
            "time": t,
            "spread_bps": (Y[i, top].mean() - Y[i, bottom].mean()) * 1e4,
            "cost_bps": traded * cost_one_way_bps,
            "turnover": traded,
            "coins": len(idx),
            "ic": ic,
        })
        prev_w, prev_h = w, hedge
    return pd.DataFrame(out)


def _ranks(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank().to_numpy()


@dataclass(frozen=True)
class XSRules:
    min_net_bps: float = 5.0         # per holding period, after costs
    min_t: float = 3.0
    min_periods: int = 30
    min_periods_per_half: int = 10
    holdout_min_t: float = 2.0       # confirmation bar on the untouched holdout


def period_stats(net: pd.Series) -> dict[str, float]:
    n = len(net)
    sd = net.std()
    return {"periods": n, "net_bps": net.mean() if n else np.nan,
            "t": net.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan,
            "win_rate": float((net > 0).mean()) if n else np.nan}


def evaluate(periods: pd.DataFrame, direction: int, split: pd.Timestamp) -> dict[str, float]:
    """Net returns for a direction (+1 momentum, -1 reversal), whole sample and two halves."""
    net = direction * periods["spread_bps"] - periods["cost_bps"]
    first, second = net[periods["time"] < split], net[periods["time"] >= split]
    stats = period_stats(net)
    stats.update(gross_bps=(direction * periods["spread_bps"]).mean(), cost_bps=periods["cost_bps"].mean(),
                 net_first_half=first.mean() if len(first) else np.nan,
                 net_second_half=second.mean() if len(second) else np.nan,
                 periods_first=len(first), periods_second=len(second))
    return stats
