"""Live version of the momentum sleeve chosen for the competition (research/18, variant C).

Long the k coins with the strongest residual (BTC-removed), risk-adjusted 16-day momentum, short the k
weakest, equal weight, BTC hedge to zero beta, gross 0.95. Rank buffer: a held coin stays until it
falls outside the best (worst) buffer_k. Rebalanced once a day on the hourly bar closing at 00:00 UTC.

It uses the same functions as the backtest (backtest/momentum.py), so live decisions match the tested ones.
Inputs beyond closes come through set_context():
  quote_volume  hourly USD volume per pair (liquidity filter: median >= min_usd_per_hour over 30 days)
  holdings      current weights per pair from the exchange (drives the rank buffer, survives restarts)
  tradable      pairs that can be traded right now (others are never picked)
Without holdings, target_weights() replays the daily decisions over the history (as in a backtest).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from quantbot.analysis.xsection import eligibility, residual_returns
from quantbot.backtest.momentum import MomentumConfig, momentum_inputs, sleeve_weights
from quantbot.strategy.base import Strategy


class MomentumSleeve(Strategy):
    name = "momentum_sleeve"
    needs_quote_volume = True

    def __init__(self, k: int = 5, buffer_k: int | None = 10, lookbacks_days: list[int] | None = None,
                 skip_days: int = 0, residual_riskadj: bool = True, weighting: str = "equal",
                 vol_target: float | None = None, max_weight: float = 1.0, gross: float = 0.95,
                 min_usd_per_hour: float = 200_000, min_history_hours: int = 720, beta_hours: int = 720,
                 leader: str = "BTC/USD", rebalance_utc_hour: int = 0, **params: Any):
        super().__init__(**params)
        self.mc = MomentumConfig(k=k, buffer_k=buffer_k, lookbacks_days=list(lookbacks_days or [16]),
                                 skip_days=skip_days, residual_riskadj=residual_riskadj, weighting=weighting,
                                 vol_target=vol_target, max_weight=max_weight, gross=gross)
        self.min_usd_per_hour, self.min_history_hours, self.beta_hours = min_usd_per_hour, min_history_hours, beta_hours
        self.leader, self.rebalance_utc_hour = leader, rebalance_utc_hour

    @property
    def warmup_bars(self) -> int:
        return self.min_history_hours + max(self.mc.lookbacks_days) * 24 + 48

    def _inputs(self, closes: pd.DataFrame):
        if self.leader not in closes:
            raise ValueError(f"{self.leader} is needed for the hedge and the residual returns")
        resid, beta = residual_returns(closes, self.leader, self.beta_hours)
        coins = list(resid.columns)
        qv = self.context.get("quote_volume")
        qv = qv if isinstance(qv, pd.DataFrame) and not qv.empty else None
        elig = eligibility(closes, qv, min_history_hours=self.min_history_hours,
                           min_usd_per_hour=self.min_usd_per_hour if qv is not None else 0.0,
                           leader=self.leader)[coins]
        tradable = self.context.get("tradable")
        if tradable:
            elig = elig & pd.Series([c in tradable for c in coins], index=coins)
        x = {k: v.reindex(columns=coins).to_numpy(dtype=float)
             for k, v in momentum_inputs(closes, resid, elig, self.mc).items()}
        P = closes[coins + [self.leader]].to_numpy(dtype=float)
        rets = np.vstack([np.zeros((1, P.shape[1])), P[1:] / P[:-1] - 1])
        return coins, x, beta[coins].to_numpy(dtype=float), elig.to_numpy(dtype=bool), rets

    def generate_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Replay: decisions on each bar closing at the rebalance hour, held in between (hourly bars)."""
        coins, x, B, E, R = self._inputs(closes)
        rebal = np.asarray(((closes.index + pd.Timedelta(hours=1)).hour == self.rebalance_utc_hour)
                           & ((closes.index + pd.Timedelta(hours=1)).minute == 0))
        out = np.zeros((len(closes), len(coins) + 1))
        cur = np.zeros(len(coins) + 1)
        for t in np.flatnonzero(rebal):
            cur = sleeve_weights(int(t), cur, x, B, E, R, self.mc)
            out[t:] = cur
        frame = pd.DataFrame(out, index=closes.index, columns=coins + [self.leader])
        return frame.reindex(columns=closes.columns, fill_value=0.0)

    def target_weights(self, closes: pd.DataFrame) -> dict[str, float]:
        holdings = self.context.get("holdings")
        if closes.empty:
            return {}
        if holdings is None:
            return super().target_weights(closes)
        coins, x, B, E, R = self._inputs(closes)
        cur = np.array([holdings.get(c, 0.0) for c in coins] + [holdings.get(self.leader, 0.0)], dtype=float)
        w = sleeve_weights(len(closes) - 1, cur, x, B, E, R, self.mc)
        return {p: float(v) for p, v in zip(coins + [self.leader], w, strict=True) if math.isfinite(v) and abs(v) > 1e-12}
