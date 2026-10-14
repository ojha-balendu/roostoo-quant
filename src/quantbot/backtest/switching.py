"""Stop-and-switch: run book A until the account falls `stop` below its peak, then switch to book B for good.

The drawdown is measured on the account's equity (after fees) at every bar close, from the peak since
the start of the run. At the first bar where equity <= peak x (1 - stop), book B is built immediately
(a full rebalance into B at that bar's close) and kept until the end.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from quantbot.backtest.momentum import MomentumConfig, sleeve_weights
from quantbot.rl.sim import PortfolioSim, SimConfig, simple_returns

Decide = Callable[[int, np.ndarray], np.ndarray]


def momentum_decider(mc: MomentumConfig, x: dict[str, np.ndarray], beta: np.ndarray, eligible: np.ndarray,
                     rets: np.ndarray, rebalance: np.ndarray, first: int, cap: float = 0.25) -> Decide:
    """Momentum sleeve decisions: rebalance on `rebalance` bars (and at `first`), hold in between;
    the BTC hedge (last slot) capped at `cap` like the live risk layer."""
    def decide(t: int, cur: np.ndarray) -> np.ndarray:
        if not (rebalance[t] or t == first):
            return cur
        w = sleeve_weights(t, cur, x, beta, eligible, rets, mc)
        w[-1] = float(np.clip(w[-1], -cap, cap))
        return w
    return decide


def simulate_with_stop(prices: np.ndarray, index: pd.DatetimeIndex, decide_a: Decide,
                       make_b: Callable[[int], Decide], start: int, end: int, cfg: SimConfig,
                       stop: float | None, rets: np.ndarray | None = None,
                       tradable: np.ndarray | None = None) -> dict:
    """Run from flat; switch from A to B (built by make_b(t)) at the first drawdown of `stop` (None = never)."""
    rets = simple_returns(prices) if rets is None else rets
    tradable = np.isfinite(prices) if tradable is None else tradable
    end = min(end, len(prices) - 1)
    sim = PortfolioSim(prices.shape[1], cfg)
    decide, peak, switched, dd_at_switch = decide_a, sim.equity, None, None
    curve, pnl, gross = [sim.equity], 0.0, []
    for t in range(start, end):
        if switched is None and stop is not None and sim.equity <= peak * (1 - stop):
            switched, dd_at_switch = t, sim.equity / peak - 1
            decide = make_b(t)
        sim.rebalance(np.nan_to_num(decide(t, sim.weights)), tradable[t])
        gross.append(float(np.abs(sim.weights).sum()))
        pnl += sim.advance(rets[t])
        peak = max(peak, sim.equity)
        curve.append(sim.equity)
    scale = max(cfg.initial_capital, abs(sim.equity), sim.traded)
    if abs(sim.equity - (cfg.initial_capital + pnl - sim.fees)) > 1e-9 * scale:
        raise AssertionError("accounting identity violated: equity != start + P&L - fees")
    return {"equity": pd.Series(curve, index=index[start:end + 1]), "fees": sim.fees, "traded": sim.traded,
            "trades": sim.trades, "pnl": pnl, "avg_gross": float(np.mean(gross)) if gross else 0.0,
            "avg_positions": 0.0, "switch_index": switched,
            "switched_at": index[switched] if switched is not None else None, "drawdown_at_switch": dd_at_switch}
