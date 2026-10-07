"""Minute-bar oscillator strategies: RSI mean reversion and MACD crossovers, optionally beta-hedged.

Positions are states per coin (+1 long, -1 short, 0 flat) decided at the close of minute t from
closes up to t. A coin's dollar position changes only when its state changes (no resizing of open
positions as prices drift), so every trade corresponds to a signal event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantbot.analysis.indicators import ema, rsi


def rsi_states(r: pd.DataFrame, low: float = 30.0, high: float = 70.0, exit_mid: float | None = 50.0) -> np.ndarray:
    """Enter long below `low`, short above `high`. Exit when RSI returns to `exit_mid`
    (None = hold until the opposite extreme, which flips the position)."""
    R = r.to_numpy(dtype=float)
    out = np.zeros(R.shape, dtype=np.int8)
    for j in range(R.shape[1]):
        pos = 0
        col = R[:, j]
        for t in range(len(col)):
            v = col[t]
            if not np.isfinite(v):
                pos = 0
            elif v < low:
                pos = 1
            elif v > high:
                pos = -1
            elif exit_mid is not None and ((pos == 1 and v >= exit_mid) or (pos == -1 and v <= exit_mid)):
                pos = 0
            out[t, j] = pos
    return out


def sliding_bar_rsi(close: pd.DataFrame, n: int = 14, bar: int = 60) -> pd.DataFrame:
    """RSI on `bar`-minute bars, recomputed every minute: at minute t it uses the closes at t, t-bar,
    t-2*bar, ... so an extreme is seen as soon as it happens rather than at the bar's close. At minutes
    that end a clock bar it equals the ordinary bar RSI. Needs a gap-free minute index."""
    vals = np.full(close.shape, np.nan)
    for k in range(bar):
        vals[k::bar] = rsi(close.iloc[k::bar], n).to_numpy()
    return pd.DataFrame(vals, index=close.index, columns=close.columns)


def at_bar_closes(x: pd.DataFrame, bar: int = 60) -> pd.DataFrame:
    """Keep values only at minutes that end a clock bar (e.g. xx:59 for hourly), carried forward:
    the signal as seen by someone who only checks at each bar's close."""
    ends = ((x.index.hour * 60 + x.index.minute + 1) % bar) == 0
    return x.where(pd.Series(ends, index=x.index), axis=0).ffill()


def holding_minutes(states: np.ndarray) -> float:
    """Average minutes per position (time in a position / number of entries)."""
    prev = np.vstack([np.zeros((1, states.shape[1]), dtype=states.dtype), states[:-1]])
    entries = int(((states != 0) & (states != prev)).sum())
    return float((states != 0).sum() / entries) if entries else float("nan")


def macd_states(close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """+1 while the MACD line is above its signal line, -1 while below (always in the market)."""
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    diff = (line - sig).to_numpy(dtype=float)
    return np.where(np.isfinite(diff), np.sign(diff), 0).astype(np.int8)


def relative_rsi(close: pd.DataFrame, btc: pd.Series, n: int = 14) -> pd.DataFrame:
    """RSI of each coin's price relative to BTC: fires only when a coin moves against the market."""
    return rsi(close.div(btc, axis=0), n)


def state_decider(states: np.ndarray, unit: float, beta: np.ndarray | None, gross: float, first: int,
                  hedge_band: float = 0.05, order_log: dict[int, int] | None = None):
    """decide(t, cur) for the simulator. Columns: traded coins, then (if beta given) the BTC hedge.
    Only coins whose state changed are traded. The hedge is re-set only when the required hedge differs
    from the current one by more than `hedge_band` (share of equity): adjusting it on every coin trade
    costs more than it saves. `order_log`, if given, records the number of orders sent at each minute."""
    n = states.shape[1]
    changed = np.zeros(len(states), dtype=bool)
    changed[1:] = (states[1:] != states[:-1]).any(axis=1)

    def decide(t: int, cur: np.ndarray) -> np.ndarray:
        if not (changed[t] or t == first):
            return cur
        target = cur.copy()
        moved = np.ones(n, dtype=bool) if t == first else states[t] != states[t - 1]
        target[:n][moved] = states[t][moved] * unit
        if beta is not None:
            b = np.nan_to_num(beta[t], nan=1.0)
            need = -float(np.dot(target[:n], b))
            if abs(need - cur[n]) > hedge_band or t == first:
                target[n] = need
        total = np.abs(target).sum()
        if total > gross:
            target = target * (gross / total)
        if order_log is not None:
            order_log[t] = int(np.count_nonzero(np.abs(target - cur) > 1e-12))
        return target

    return decide


def orders_per_minute(states: np.ndarray, hedged: bool) -> np.ndarray:
    """Orders needed each minute: one per coin whose state changed, plus one hedge adjustment."""
    n_changes = np.zeros(len(states), dtype=int)
    n_changes[1:] = (states[1:] != states[:-1]).sum(axis=1)
    return n_changes + (hedged & (n_changes > 0)).astype(int)


def signal_check(states: np.ndarray, close: pd.DataFrame, btc: pd.Series, beta: np.ndarray | None,
                 horizons=(1, 5, 15, 60)) -> pd.DataFrame:
    """Average move in the trade's direction after each entry (flat or opposite -> +1/-1), in bps:
    raw, and relative to beta x BTC over the same horizon."""
    C = close.to_numpy(dtype=float)
    b_px = btc.to_numpy(dtype=float)
    prev = np.vstack([np.zeros((1, states.shape[1]), dtype=np.int8), states[:-1]])
    entry = (states != 0) & (states != prev)
    t_idx, j_idx = np.nonzero(entry)
    rows = []
    for h in horizons:
        ok = t_idx + h < len(C)
        t, j = t_idx[ok], j_idx[ok]
        d = states[t, j].astype(float)
        raw = C[t + h, j] / C[t, j] - 1
        mkt = b_px[t + h] / b_px[t] - 1
        bb = np.nan_to_num(beta[t, j], nan=1.0) if beta is not None else np.ones(len(t))
        fin = np.isfinite(raw) & np.isfinite(mkt)
        rows.append({"horizon_min": h, "entries": int(fin.sum()),
                     "raw_bps": float(np.mean(d[fin] * raw[fin]) * 1e4) if fin.any() else np.nan,
                     "vs_btc_bps": float(np.mean(d[fin] * (raw[fin] - bb[fin] * mkt[fin])) * 1e4) if fin.any() else np.nan,
                     "hit": float(np.mean(d[fin] * raw[fin] > 0)) if fin.any() else np.nan})
    return pd.DataFrame(rows)
