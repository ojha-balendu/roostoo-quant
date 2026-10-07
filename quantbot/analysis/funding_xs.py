"""H6: cross-sectional funding tilt (long low funding, short high funding).

Hypothesis
    Coins whose perps carry the highest funding are where leveraged long
    demand is most crowded; they should underperform the lowest-funding coins
    over the following days as the crowding unwinds.

Signal (at each rebalance time t, 00:00 UTC)
    funding_t = sum of settled funding rates in [t - window, t) / window_days
                (a daily carry rate, comparable across 8h- and 4h-interval perps)
    High-funding coins are usually recent winners, and cross-sectional
    momentum/reversal was already rejected, so by default the funding rank is
    residualised against 7- and 30-day return ranks.  score = -residual.

Portfolio
    Long the top ``quantile`` by score, short the bottom ``quantile``, equal
    weight within legs.  Leg sizes are set so that gross = 1 and the net beta
    to BTC is zero (1x: gross exposure <= equity).  Short losses are capped at
    the collateral, matching Roostoo's 1x shorts.  Turnover is measured against
    drifted weights and charged ``one_way_cost_bps``.

Reported per rebalance period: rank IC of score vs forward return, gross and
net long/short return.  Periods do not overlap, so the t-statistic is the
plain one over periods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from quantbot.data.futures_vision import to_epoch_ms

_MS_PER_DAY = 86_400_000


@dataclass(frozen=True)
class FundingXSParams:
    window_days: int = 7
    rebalance: str = "W-MON"
    quantile: float = 0.2
    min_coins: int = 10
    mom_days: Tuple[int, ...] = (7, 30)
    beta_days: int = 60
    residualize: bool = True
    beta_neutral: bool = True
    one_way_cost_bps: float = 12.5
    market_symbol: str = "BTCUSDT"
    beta_clip: Tuple[float, float] = (0.2, 3.0)


@dataclass(frozen=True)
class ProtocolRules:
    min_net_bps: float = 5.0
    min_t: float = 3.0
    min_periods: int = 30


# ----------------------------------------------------------------------- inputs


def daily_closes(close: pd.DataFrame) -> pd.DataFrame:
    """Closes at exactly 00:00 UTC on a gap-free daily grid."""
    if close.empty:
        return close
    idx = pd.DatetimeIndex(close.index)
    daily = close.loc[idx == idx.normalize()]
    if daily.empty:
        return daily
    full = pd.date_range(daily.index.min(), daily.index.max(), freq=pd.Timedelta(days=1))
    return daily.reindex(full)


def trailing_funding(funding: Dict[str, pd.DataFrame], dates: pd.DatetimeIndex, window_days: int) -> pd.DataFrame:
    """Average daily funding over settlements in [t - window, t).

    NaN when the perp did not exist for the whole window, had fewer than one
    settlement per day, or has not settled for more than a day (delisted).
    """
    tgt = to_epoch_ms(dates)
    win = window_days * _MS_PER_DAY
    out = {}
    for sym, df in funding.items():
        s = df["rate"].dropna()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        if s.empty:
            out[sym] = np.full(len(tgt), np.nan)
            continue
        t = to_epoch_ms(s.index)
        cum = np.concatenate([[0.0], np.cumsum(s.to_numpy(dtype=float))])
        hi = np.searchsorted(t, tgt, side="left")        # settlements strictly before target
        lo = np.searchsorted(t, tgt - win, side="left")
        total = cum[hi] - cum[lo]
        count = hi - lo
        last_age = tgt - t[np.clip(hi - 1, 0, None)]
        ok = (hi > 0) & (t[0] <= tgt - win) & (count >= window_days) & (last_age <= _MS_PER_DAY)
        out[sym] = np.where(ok, total / window_days, np.nan)
    return pd.DataFrame(out, index=dates)


def rolling_beta(rets: pd.DataFrame, market: pd.Series, window: int) -> pd.DataFrame:
    mp = max(2, window // 2)
    m = market.reindex(rets.index)
    mean_x = rets.rolling(window, min_periods=mp).mean()
    mean_m = m.rolling(window, min_periods=mp).mean()
    mean_xm = rets.mul(m, axis=0).rolling(window, min_periods=mp).mean()
    var_m = (m * m).rolling(window, min_periods=mp).mean() - mean_m ** 2
    cov = mean_xm - mean_x.mul(mean_m, axis=0)
    return cov.div(var_m.where(var_m > 0), axis=0)


# ---------------------------------------------------------------------- helpers


def _zrank(x) -> np.ndarray:
    r = pd.Series(np.asarray(x, dtype=float)).rank().to_numpy()
    sd = r.std()
    return (r - r.mean()) / sd if sd > 0 else np.zeros_like(r)


def _spearman(a, b) -> float:
    a = pd.Series(np.asarray(a, dtype=float)).rank()
    b = pd.Series(np.asarray(b, dtype=float)).rank()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def leg_weights(beta_long: float, beta_short: float, beta_neutral: bool = True) -> Tuple[float, float]:
    """(long gross, short gross) with sum 1 and, if requested, zero net beta."""
    if not beta_neutral:
        return 0.5, 0.5
    a = beta_short / (beta_long + beta_short)
    return a, 1.0 - a


def rebalance_dates(index: pd.DatetimeIndex, rule: str) -> pd.DatetimeIndex:
    if len(index) == 0:
        return index
    cand = pd.date_range(index[0], index[-1], freq=rule)
    return index[index.isin(cand)]


# --------------------------------------------------------------------- backtest


def run_backtest(close_daily: pd.DataFrame, fund_daily: pd.DataFrame, p: FundingXSParams,
                 start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """One row per rebalance period [t, t_next) fully inside [start, end]."""
    rets = close_daily / close_daily.shift(1) - 1.0
    market = rets[p.market_symbol] if p.market_symbol in rets.columns else rets.mean(axis=1)
    beta = rolling_beta(rets, market, p.beta_days)
    moms = {d: close_daily / close_daily.shift(d) - 1.0 for d in p.mom_days}
    fund_daily = fund_daily.reindex(index=close_daily.index, columns=close_daily.columns)

    rb = rebalance_dates(close_daily.index, p.rebalance)
    if start is not None:
        rb = rb[rb >= start]
    if end is not None:
        rb = rb[rb <= end]

    prev_w = pd.Series(dtype=float)
    rows = []
    for t, tn in zip(rb[:-1], rb[1:]):
        fwd = close_daily.loc[tn] / close_daily.loc[t] - 1.0
        frame = pd.DataFrame({"f": fund_daily.loc[t], "fwd": fwd, "beta": beta.loc[t],
                              **{f"m{d}": moms[d].loc[t] for d in p.mom_days}}).dropna()
        row = {"date": t, "next_date": tn, "n_coins": len(frame)}
        s = None
        if len(frame) >= p.min_coins:
            y = _zrank(frame["f"])
            if p.residualize:
                X = np.column_stack([np.ones(len(frame))] + [_zrank(frame[f"m{d}"]) for d in p.mom_days])
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                s = y - X @ coef
            else:
                s = y
            if np.std(s) < 1e-6:      # signal fully explained by momentum: nothing to trade on
                s = None
        if s is None:
            w = pd.Series(dtype=float)
            gross = 0.0
            r_eff = pd.Series(dtype=float)
            row.update(ic=np.nan, ic_raw=np.nan, k=0, beta_long=np.nan, beta_short=np.nan, corr_f_mom=np.nan)
        else:
            score = pd.Series(-s, index=frame.index)
            k = max(1, int(np.floor(len(frame) * p.quantile)))
            ranked = score.sort_values(kind="mergesort")
            shorts, longs = ranked.index[:k], ranked.index[-k:]
            lo, hi = p.beta_clip
            b_l = float(np.clip(frame.loc[longs, "beta"].mean(), lo, hi))
            b_s = float(np.clip(frame.loc[shorts, "beta"].mean(), lo, hi))
            a, b = leg_weights(b_l, b_s, p.beta_neutral)
            w = pd.Series(0.0, index=frame.index)
            w[longs] = a / k
            w[shorts] = -b / k
            w = w[w != 0]
            r = frame.loc[w.index, "fwd"]
            r_eff = r.where(w > 0, np.minimum(r, 1.0))     # 1x short: loss capped at collateral
            gross = float((w * r_eff).sum())
            row.update(ic=_spearman(score, frame["fwd"]), ic_raw=_spearman(-frame["f"], frame["fwd"]), k=k,
                       beta_long=b_l, beta_short=b_s,
                       corr_f_mom=_spearman(frame["f"], frame[f"m{p.mom_days[0]}"]))
        union = w.index.union(prev_w.index)
        turnover = float((w.reindex(union, fill_value=0.0) - prev_w.reindex(union, fill_value=0.0)).abs().sum())
        cost = turnover * p.one_way_cost_bps / 1e4
        row.update(gross=gross, turnover=turnover, cost=cost, net=gross - cost)
        rows.append(row)
        if len(w) and 1.0 + gross > 0:
            prev_w = w * (1.0 + r_eff) / (1.0 + gross)
        else:
            prev_w = pd.Series(dtype=float)
    return pd.DataFrame(rows)


def summarize_periods(periods: pd.DataFrame, rules: ProtocolRules = ProtocolRules()) -> Dict[str, float]:
    active = periods[periods["k"] > 0] if not periods.empty else periods
    n = len(active)
    if n == 0:
        return {"periods": 0, "passes_protocol": False}

    def _t(x: pd.Series) -> float:
        x = x.dropna()
        return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 1 and x.std(ddof=1) > 0 else np.nan

    split = active["date"].iloc[0] + (active["date"].iloc[-1] - active["date"].iloc[0]) / 2
    first = active[active["date"] < split]["net"].mean()
    second = active[active["date"] >= split]["net"].mean()
    days = float((active["next_date"] - active["date"]).dt.days.median())
    per_year = 365.25 / days if days > 0 else np.nan
    net = active["net"]
    sd = net.std(ddof=1)
    t_net = _t(net)
    out = {
        "periods": n,
        "period_days": days,
        "coins_median": float(active["n_coins"].median()),
        "per_leg": float(active["k"].median()),
        "mean_ic": float(active["ic"].mean()),
        "t_ic": _t(active["ic"]),
        "mean_ic_raw_funding": float(active["ic_raw"].mean()),
        "corr_funding_mom": float(active["corr_f_mom"].mean()),
        "gross_bps": float(active["gross"].mean() * 1e4),
        "cost_bps": float(active["cost"].mean() * 1e4),
        "net_bps": float(net.mean() * 1e4),
        "t_net": t_net,
        "net_bps_1st_half": float(first * 1e4),
        "net_bps_2nd_half": float(second * 1e4),
        "hit_rate": float((net > 0).mean()),
        "ann_sharpe": float(net.mean() / sd * np.sqrt(per_year)) if sd > 0 else np.nan,
        "avg_turnover": float(active["turnover"].mean()),
    }
    out["passes_protocol"] = bool(
        out["net_bps"] >= rules.min_net_bps and np.isfinite(t_net) and t_net >= rules.min_t
        and n >= rules.min_periods and first > 0 and second > 0)
    return out
