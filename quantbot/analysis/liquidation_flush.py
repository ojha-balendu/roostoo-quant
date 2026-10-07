"""H5: liquidation-flush rebound event study.

Hypothesis
    When a coin falls hard within an hour *and* futures open interest drops at
    the same time, the move was driven by forced liquidations rather than new
    information.  Whoever absorbs that flow should be paid: price should
    partly rebound over the next hours.  Prior positive funding (longs were
    crowded and paying) should strengthen the effect.

Event (on hourly bars, indexed by bar close time t)
    z_t      = log return of bar t / trailing std of hourly log returns
               (``vol_lookback`` bars, ending at t-1: no look-ahead)
    oi_chg_t = OI(t) / OI(t-1h) - 1, OI snapshots shifted by ``oi_lag_minutes``
               to the moment they become available
    funding  = mean of the last ``funding_n`` settled funding rates before t

    price     : z_t <= -z_threshold
    price_oi  : ... and oi_chg_t <= -oi_drop
    full      : ... and funding > 0

    All three conditions are evaluated on the same coin-hours (where OI and
    funding data exist) so the incremental value of each filter is visible.

Trade
    Buy at the close of bar t + ``entry_lag``, sell ``hold`` bars later, pay
    ``cost_bps`` per round trip.  One open event per coin at a time
    (``cooldown`` bars, default = longest hold).

Statistics
    Events cluster on crash days, so the t-statistic is computed on daily
    means of event returns (one observation per UTC day with events), not on
    individual events.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from quantbot.data.futures_vision import to_epoch_ms

CONDITIONS = ("price", "price_oi", "full")


@dataclass(frozen=True)
class FlushParams:
    z_threshold: float = 3.0
    vol_lookback: int = 720          # hours (30 days)
    min_vol_obs: int = 240
    oi_drop: float = 0.05
    funding_n: int = 3               # settlements averaged for the funding filter
    holds: Tuple[int, ...] = (4, 12, 24)
    entry_lag: int = 0               # extra bars between signal and entry
    cost_bps: float = 25.0           # round trip, incl. spread
    cooldown: Optional[int] = None   # bars; default max(holds)
    oi_lag_minutes: int = 5
    oi_tolerance_minutes: int = 30
    funding_tolerance_hours: int = 24

    @property
    def effective_cooldown(self) -> int:
        return int(self.cooldown) if self.cooldown is not None else int(max(self.holds))


@dataclass(frozen=True)
class ProtocolRules:
    min_net_bps: float = 5.0
    min_t: float = 3.0
    min_days: int = 30


# ----------------------------------------------------------------------- inputs


def regular_grid(close: pd.DataFrame, step: pd.Timedelta = pd.Timedelta(hours=1)) -> pd.DataFrame:
    """Reindex to a gap-free grid so that shift(1) always means one bar."""
    if close.empty:
        return close
    full = pd.date_range(close.index.min(), close.index.max(), freq=step)
    return close.reindex(full)


def align_asof(series_by_symbol: Dict[str, pd.Series], index: pd.DatetimeIndex,
               lag: pd.Timedelta, tolerance: pd.Timedelta) -> pd.DataFrame:
    """Latest value available at each index time.

    A value stamped at ``s`` becomes usable at ``s + lag``; values older than
    ``tolerance`` at the target time are treated as missing.
    """
    tgt = to_epoch_ms(index)
    lag_ms = int(lag / pd.Timedelta(milliseconds=1))
    tol_ms = int(tolerance / pd.Timedelta(milliseconds=1))
    out = {}
    for sym, s in series_by_symbol.items():
        s = s.dropna()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        if s.empty:
            out[sym] = np.full(len(tgt), np.nan)
            continue
        src = to_epoch_ms(s.index) + lag_ms
        pos = np.searchsorted(src, tgt, side="right") - 1
        safe = np.clip(pos, 0, None)
        ok = (pos >= 0) & (tgt - src[safe] <= tol_ms)
        out[sym] = np.where(ok, s.to_numpy(dtype=float)[safe], np.nan)
    return pd.DataFrame(out, index=index)


def prepare_inputs(close: pd.DataFrame, metrics: Dict[str, pd.DataFrame],
                   funding: Dict[str, pd.DataFrame], p: FlushParams
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hourly close grid plus OI and prior-funding panels aligned to it."""
    close = regular_grid(close)
    oi = align_asof({s: m["oi"] for s, m in metrics.items() if "oi" in m.columns}, close.index,
                    lag=pd.Timedelta(minutes=p.oi_lag_minutes),
                    tolerance=pd.Timedelta(minutes=p.oi_tolerance_minutes))
    fund = align_asof({s: f["rate"].rolling(p.funding_n, min_periods=p.funding_n).mean()
                       for s, f in funding.items()}, close.index,
                      lag=pd.Timedelta(0), tolerance=pd.Timedelta(hours=p.funding_tolerance_hours))
    return close, oi.reindex(columns=close.columns), fund.reindex(columns=close.columns)


def equal_weight_index(close: pd.DataFrame) -> pd.Series:
    rets = close / close.shift(1) - 1.0
    return (1.0 + rets.mean(axis=1, skipna=True).fillna(0.0)).cumprod()


# ----------------------------------------------------------------------- events


def compute_features(close: pd.DataFrame, oi: pd.DataFrame, p: FlushParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logret = np.log(close).diff()
    sigma = logret.rolling(p.vol_lookback, min_periods=p.min_vol_obs).std().shift(1)
    z = logret / sigma
    oi_chg = oi / oi.shift(1) - 1.0
    return z, oi_chg


def detect_events(close: pd.DataFrame, oi: pd.DataFrame, funding: pd.DataFrame,
                  p: FlushParams, condition: str = "full", require_data: bool = True) -> pd.DataFrame:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    oi = oi.reindex(index=close.index, columns=close.columns)
    funding = funding.reindex(index=close.index, columns=close.columns)
    z, oi_chg = compute_features(close, oi, p)
    mask = np.array((z <= -p.z_threshold).to_numpy(), dtype=bool)   # writable copy (pandas CoW)
    if require_data:
        mask = mask & oi_chg.notna().to_numpy() & funding.notna().to_numpy()
    if condition in ("price_oi", "full"):
        mask = mask & (oi_chg <= -p.oi_drop).to_numpy()
    if condition == "full":
        mask = mask & (funding > 0).to_numpy()
    zz, oo, ff = z.to_numpy(), oi_chg.to_numpy(), funding.to_numpy()
    cd = p.effective_cooldown
    rows = []
    for j, sym in enumerate(close.columns):
        last = -(10 ** 9)
        for i in np.flatnonzero(mask[:, j]):
            if i - last < cd:
                continue
            last = i
            rows.append((close.index[i], sym, int(i), zz[i, j], oo[i, j], ff[i, j]))
    return pd.DataFrame(rows, columns=["time", "symbol", "pos", "z", "oi_chg", "funding"])


def event_returns(events: pd.DataFrame, close: pd.DataFrame, p: FlushParams,
                  benchmark: Optional[pd.Series] = None) -> pd.DataFrame:
    cols = ["time", "symbol", "hold", "entry_time", "exit_time", "gross", "net", "excess"]
    if events.empty:
        return pd.DataFrame(columns=cols)
    px = close.to_numpy(dtype=float)
    col = {c: i for i, c in enumerate(close.columns)}
    b = benchmark.reindex(close.index).to_numpy(dtype=float) if benchmark is not None else None
    n = len(close)
    cost = p.cost_bps / 1e4
    rows = []
    for ev in events.itertuples(index=False):
        e = ev.pos + p.entry_lag
        j = col[ev.symbol]
        for h in p.holds:
            x = e + h
            if x >= n:
                continue
            p0, p1 = px[e, j], px[x, j]
            if not (np.isfinite(p0) and np.isfinite(p1)) or p0 <= 0:
                continue
            gross = p1 / p0 - 1.0
            bench = np.nan
            if b is not None and np.isfinite(b[e]) and np.isfinite(b[x]) and b[e] > 0:
                bench = b[x] / b[e] - 1.0
            rows.append((ev.time, ev.symbol, h, close.index[e], close.index[x], gross, gross - cost, gross - bench))
    return pd.DataFrame(rows, columns=cols)


# ------------------------------------------------------------------- statistics


def clustered_stats(values: pd.Series, times: pd.Series) -> Tuple[int, float, float]:
    """(n_days, mean of daily means, t-stat over days)."""
    if len(values) == 0:
        return 0, np.nan, np.nan
    day = pd.DatetimeIndex(times).floor("D")
    daily = pd.Series(np.asarray(values, dtype=float)).groupby(np.asarray(day)).mean()
    n = len(daily)
    mean = float(daily.mean())
    sd = float(daily.std(ddof=1)) if n > 1 else np.nan
    t = mean / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    return n, mean, t


def baseline_forward(close: pd.DataFrame, oi: pd.DataFrame, funding: pd.DataFrame, p: FlushParams,
                     start: pd.Timestamp, end: pd.Timestamp) -> Dict[int, float]:
    """Unconditional mean forward return (bps) on the same data-available coin-hours."""
    avail = oi.reindex_like(close).notna() & funding.reindex_like(close).notna()
    in_window = (close.index >= start) & (close.index <= end)
    out = {}
    for h in p.holds:
        fwd = close.shift(-(h + p.entry_lag)) / close.shift(-p.entry_lag) - 1.0
        vals = fwd[avail].to_numpy()[in_window]
        vals = vals[np.isfinite(vals)]
        out[h] = float(vals.mean() * 1e4) if vals.size else np.nan
    return out


def summarize(rets: pd.DataFrame, split_time: pd.Timestamp, rules: ProtocolRules = ProtocolRules(),
              baseline: Optional[Dict[int, float]] = None) -> pd.DataFrame:
    rows = []
    if rets.empty:
        return pd.DataFrame(rows)
    for (cond, hold), g in rets.groupby(["condition", "hold"], sort=False):
        n_days, net, t = clustered_stats(g["net"], g["time"])
        _, gross, _ = clustered_stats(g["gross"], g["time"])
        _, excess, t_exc = clustered_stats(g["excess"].dropna(), g.loc[g["excess"].notna(), "time"])
        first = g[g["time"] < split_time]
        second = g[g["time"] >= split_time]
        _, net1, _ = clustered_stats(first["net"], first["time"])
        _, net2, _ = clustered_stats(second["net"], second["time"])
        passed = (np.isfinite(net) and net * 1e4 >= rules.min_net_bps and np.isfinite(t) and t >= rules.min_t
                  and n_days >= rules.min_days and np.nan_to_num(net1) > 0 and np.nan_to_num(net2) > 0)
        rows.append({
            "condition": cond, "hold_h": int(hold), "events": len(g), "event_days": n_days,
            "gross_bps": gross * 1e4, "net_bps": net * 1e4, "t_days": t,
            "excess_gross_bps": excess * 1e4, "t_excess": t_exc,
            "net_bps_1st_half": net1 * 1e4, "net_bps_2nd_half": net2 * 1e4,
            "hit_rate": float((g["net"] > 0).mean()),
            "baseline_gross_bps": (baseline or {}).get(int(hold), np.nan),
            "passes_protocol": bool(passed),
        })
    return pd.DataFrame(rows)


def run_study(close: pd.DataFrame, oi: pd.DataFrame, funding: pd.DataFrame, p: FlushParams,
              start: pd.Timestamp, end: pd.Timestamp, conditions: Sequence[str] = CONDITIONS,
              benchmark: Optional[pd.Series] = None, rules: ProtocolRules = ProtocolRules()
              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect events on full history (features are backward-looking), then keep
    events whose signal *and* exit fall inside [start, end]."""
    if benchmark is None:
        benchmark = equal_weight_index(close)
    all_rets = []
    for cond in conditions:
        ev = detect_events(close, oi, funding, p, cond)
        r = event_returns(ev, close, p, benchmark)
        if not r.empty:
            r = r[(r["time"] >= start) & (r["exit_time"] <= end)]
            r.insert(0, "condition", cond)
            all_rets.append(r)
    rets = pd.concat(all_rets, ignore_index=True) if all_rets else pd.DataFrame(
        columns=["condition", "time", "symbol", "hold", "entry_time", "exit_time", "gross", "net", "excess"])
    split = start + (end - start) / 2
    base = baseline_forward(close, oi, funding, p, start, end)
    return summarize(rets, split, rules, base), rets
