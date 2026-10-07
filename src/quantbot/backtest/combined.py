"""Several strategy sleeves sharing one account, simulated on 1-minute bars.

Each sleeve keeps its own (virtual) dollar positions, sized as a fixed share (`alloc`) of total
equity whenever it rebalances, and drifting with prices in between. The account holds the SUM of the
sleeves' positions and trades only the net difference, so opposite trades in different sleeves net
out (fewer orders, less fee). Fees are paid by the account on what it actually trades.

Sleeves implement  decide(t, equity, h, first) -> new dollar positions for this sleeve, or None (no change).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantbot.backtest.momentum import MomentumConfig, sleeve_weights
from quantbot.rl.sim import PortfolioSim, SimConfig, simple_returns


def last_closed_row(row_close_times: pd.DatetimeIndex, minute_index: pd.DatetimeIndex) -> np.ndarray:
    """For each minute bar, the row of the latest coarser bar that has CLOSED by the end of that minute."""
    return row_close_times.searchsorted(minute_index + pd.Timedelta(minutes=1), side="right") - 1


class HoldSleeve:
    """Buy once at the start and hold (benchmark)."""

    def __init__(self, alloc: float, weights: np.ndarray):
        self.alloc, self.weights = alloc, weights

    def decide(self, t, equity, h, first):
        return self.weights * self.alloc * equity if first else None


class MomentumSleeve:
    """Momentum sleeve (backtest/momentum.py) rebalanced on hourly rows at chosen minutes (00:00 UTC)."""

    def __init__(self, alloc: float, mc: MomentumConfig, x: dict[str, np.ndarray], beta_h: np.ndarray,
                 elig_h: np.ndarray, rets_h: np.ndarray, hour_row: np.ndarray, rebalance: np.ndarray,
                 coin_cols: np.ndarray, btc_col: int):
        self.alloc, self.mc, self.x, self.beta, self.elig, self.rets = alloc, mc, x, beta_h, elig_h, rets_h
        self.hour_row, self.rebalance, self.coin_cols, self.btc_col = hour_row, rebalance, coin_cols, btc_col

    def decide(self, t, equity, h, first):
        if not (self.rebalance[t] or first):
            return None
        hr = int(self.hour_row[t])
        if hr < 0:
            return None
        cap = self.alloc * equity
        cur = np.r_[h[self.coin_cols], h[self.btc_col]] / cap if cap > 0 else np.zeros(len(self.coin_cols) + 1)
        w = sleeve_weights(hr, cur, self.x, self.beta, self.elig, self.rets, self.mc)
        new = np.zeros_like(h)
        new[self.coin_cols] = w[:-1] * cap
        new[self.btc_col] += w[-1] * cap
        return new


class BtcTrendSleeve:
    """Long BTC (or flat) by an hourly target weight; re-sized only when it moves by more than `band`."""

    def __init__(self, alloc: float, target_h: np.ndarray, hour_row: np.ndarray, hour_end: np.ndarray,
                 btc_col: int, band: float = 0.10):
        self.alloc, self.target, self.hour_row, self.hour_end = alloc, target_h, hour_row, hour_end
        self.btc_col, self.band = btc_col, band

    def decide(self, t, equity, h, first):
        if not (self.hour_end[t] or first):
            return None
        hr = int(self.hour_row[t])
        tgt = float(np.nan_to_num(self.target[hr])) if hr >= 0 else 0.0
        cap = self.alloc * equity
        cur = h[self.btc_col] / cap if cap > 0 else 0.0
        if not (first or (tgt == 0.0 and cur != 0.0) or abs(tgt - cur) > self.band):
            return None
        new = np.zeros_like(h)
        new[self.btc_col] = tgt * cap
        return new


class RsiSleeve:
    """Per-coin RSI states (+1/0/-1) on minute bars; only coins whose state changes are traded."""

    def __init__(self, alloc: float, states: np.ndarray, cols: np.ndarray, unit: float):
        self.alloc, self.states, self.cols, self.unit = alloc, states, cols, unit
        self.changed = np.zeros(len(states), dtype=bool)
        self.changed[1:] = (states[1:] != states[:-1]).any(axis=1)

    def decide(self, t, equity, h, first):
        if not (first or self.changed[t]):
            return None
        moved = np.ones(self.states.shape[1], dtype=bool) if first else self.states[t] != self.states[t - 1]
        new = h.copy()
        new[self.cols[moved]] = self.states[t, moved] * self.unit * self.alloc * equity
        return new


def simulate_combined(prices: np.ndarray, index: pd.DatetimeIndex, sleeves: list, start: int, end: int,
                      cfg: SimConfig, rets: np.ndarray | None = None, tradable: np.ndarray | None = None) -> dict:
    """Pass precomputed `rets` (simple_returns(prices)) and `tradable` when running many windows."""
    rets = simple_returns(prices) if rets is None else rets
    tradable = np.isfinite(prices) if tradable is None else tradable
    end = min(end, len(prices) - 1)
    sim = PortfolioSim(prices.shape[1], cfg)
    H = [np.zeros(prices.shape[1]) for _ in sleeves]
    curve, pnl, orders, gross = [sim.equity], 0.0, 0, []
    for t in range(start, end):
        changed = False
        for i, s in enumerate(sleeves):
            new = s.decide(t, sim.equity, H[i], t == start)
            if new is not None:
                H[i] = new
                changed = True
        if changed:
            before = sim.trades
            sim.rebalance(np.sum(H, axis=0) / sim.equity, tradable[t])
            orders += sim.trades - before
        gross.append(float(np.abs(sim.weights).sum()))
        r = np.nan_to_num(rets[t])
        pnl += sim.advance(r)
        for i in range(len(H)):
            H[i] = H[i] * (1.0 + r)
        curve.append(sim.equity)
    scale = max(cfg.initial_capital, abs(sim.equity), sim.traded)
    if abs(sim.equity - (cfg.initial_capital + pnl - sim.fees)) > 1e-9 * scale:
        raise AssertionError("accounting identity violated: equity != start + P&L - fees")
    return {"equity": pd.Series(curve, index=index[start:end + 1]), "fees": sim.fees, "traded": sim.traded,
            "trades": orders, "pnl": pnl, "avg_gross": float(np.mean(gross)) if gross else 0.0, "avg_positions": 0.0}
