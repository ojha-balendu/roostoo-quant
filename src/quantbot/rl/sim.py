"""Portfolio accounting shared by the RL environment and every benchmark.

Positions are signed dollar notionals h (long > 0, short < 0). Over one bar, every position
earns h * return, and its notional drifts with price (a short of fixed quantity changes value
too). Rebalancing to target weights costs |traded $| * (fee + half spread). Longs and short
collateral both use cash, so gross exposure is capped (default 95% of equity).
Not modelled: Roostoo's cap on short losses at the posted collateral (irrelevant at these sizes).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class SimConfig:
    initial_capital: float = 100_000.0
    fee: float = 0.001             # per side; Roostoo taker and short open/close are all 0.1%
    spread_bps: float = 2.0        # full bid/ask spread; half is paid on each trade
    max_gross: float = 0.95
    max_weight: float = 0.25
    trade_band: float = 0.05       # per-coin weight changes smaller than this are not traded

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def cost_rate(self) -> float:
        return self.fee + self.spread_bps / 2e4


def weights_from_action(action: np.ndarray, valid: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Map an action in [-1, 1]^N to target weights: x max_weight, zero for unavailable coins, gross cap."""
    w = np.clip(np.nan_to_num(np.asarray(action, dtype=float)), -1.0, 1.0) * cfg.max_weight
    w[~valid] = 0.0
    return cap_gross(w, cfg.max_gross)


def cap_gross(w: np.ndarray, max_gross: float) -> np.ndarray:
    gross = np.abs(w).sum()
    return w * (max_gross / gross) if gross > max_gross else w


class PortfolioSim:
    def __init__(self, n_assets: int, cfg: SimConfig):
        self.n = n_assets
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.equity = float(self.cfg.initial_capital)
        self.h = np.zeros(self.n)
        self.fees = 0.0
        self.traded = 0.0
        self.trades = 0

    @property
    def weights(self) -> np.ndarray:
        return self.h / self.equity if self.equity > 0 else np.zeros(self.n)

    def rebalance(self, target: np.ndarray, tradable: np.ndarray) -> float:
        """Move towards target weights; coins that can't trade this bar keep their position."""
        delta = np.asarray(target, dtype=float) - self.weights
        delta[np.abs(delta) < self.cfg.trade_band] = 0.0
        delta[~tradable] = 0.0
        traded = float(np.abs(delta).sum() * self.equity)
        self.trades += int(np.count_nonzero(delta))
        cost = traded * self.cfg.cost_rate
        self.h = self.h + delta * self.equity
        self.equity -= cost
        self.fees += cost
        self.traded += traded
        return cost

    def advance(self, returns: np.ndarray) -> float:
        r = np.nan_to_num(np.asarray(returns, dtype=float))
        pnl = float(self.h @ r)
        self.h = self.h * (1.0 + r)
        self.equity += pnl
        return pnl


def simple_returns(prices: np.ndarray) -> np.ndarray:
    """r[t] = return from bar t to bar t+1 (length T-1); missing prices give 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = prices[1:] / prices[:-1] - 1.0
    return np.where(np.isfinite(r), r, 0.0)


def run_weights(prices: np.ndarray, weights: np.ndarray, start: int, n_steps: int, cfg: SimConfig,
                rebalance: np.ndarray | None = None) -> dict:
    """Run a fixed weight schedule through the same accounting as the agent.
    weights[t] is the target decided at the close of bar t; rebalance[t] says whether to trade then."""
    rets = simple_returns(prices)
    tradable = np.isfinite(prices)
    sim = PortfolioSim(prices.shape[1], cfg)
    curve, gross, net = [sim.equity], [], []
    for t in range(start, start + n_steps):
        if rebalance is None or rebalance[t] or t == start:
            sim.rebalance(cap_gross(np.nan_to_num(weights[t]), cfg.max_gross), tradable[t])
        gross.append(np.abs(sim.weights).sum())
        net.append(sim.weights.sum())
        sim.advance(rets[t])
        curve.append(sim.equity)
    return {"equity": np.array(curve), "fees": sim.fees, "traded": sim.traded,
            "gross": np.array(gross), "net": np.array(net)}
