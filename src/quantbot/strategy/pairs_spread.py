"""Pairs trade on the spread between two coins (default ETH vs BTC).

On bars of `bar_minutes`, over a rolling window of `lookback_hours`:
  beta   = cov(r_A, r_B) / var(r_B)          hedge ratio from a straight-line fit of A's returns on B's
  spread = log(A) - beta * log(B)
  z      = (spread - window mean) / window std   (window spread rebuilt with the current beta)
  edge   = |spread - window mean|                 profit per $1 of A-leg if the spread fully reverts
  cost   = (round_trip_fee + spread_cost) * (1 + |beta|)   fees for both legs, per $1 of A-leg

Enter when |z| >= z_entry AND edge >= fee_multiple * cost:
  z > 0 (A rich): short A, long beta x B     z < 0 (A cheap): long A, short beta x B
Exit on the first of: z back to z_exit ("revert"), |z| >= z_stop against us ("stop"),
or max_hold_hours elapsed ("timeout"). Legs are not rebalanced while a trade is open.

The fee rule guarantees a trade is worth taking IF the spread reverts. It cannot guarantee
the spread reverts; the stop and timeout exist for when it does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantbot.strategy.base import Strategy


def known_time_bars(closes: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample to `minutes` bars labelled by the time each close is KNOWN (end of bar)."""
    bar = closes.index.to_series().diff().median()
    shifted = closes.copy()
    shifted.index = closes.index + bar
    return shifted.resample(f"{minutes}min", label="right", closed="right").last()


class PairsSpread(Strategy):
    name = "pairs_spread"

    def __init__(
        self,
        leg_a: str = "ETH/USD",
        leg_b: str = "BTC/USD",
        bar_minutes: int = 5,
        lookback_hours: float = 72,
        z_entry: float = 2.0,
        z_exit: float = 0.0,
        z_stop: float = 4.0,
        max_hold_hours: float = 48,
        fee_multiple: float = 2.0,
        round_trip_fee: float = 0.002,
        spread_cost: float = 0.0002,
        gross: float = 0.95,
        **params,
    ):
        super().__init__(leg_a=leg_a, leg_b=leg_b, bar_minutes=bar_minutes, lookback_hours=lookback_hours,
                         z_entry=z_entry, z_exit=z_exit, z_stop=z_stop, max_hold_hours=max_hold_hours,
                         fee_multiple=fee_multiple, round_trip_fee=round_trip_fee, spread_cost=spread_cost,
                         gross=gross, **params)
        self.a, self.b = leg_a, leg_b
        self.bar_minutes = int(bar_minutes)
        self.n = max(int(lookback_hours * 60 / self.bar_minutes), 10)
        self.z_entry, self.z_exit, self.z_stop = float(z_entry), float(z_exit), float(z_stop)
        self.max_hold = pd.Timedelta(hours=max_hold_hours)
        self.fee_multiple = float(fee_multiple)
        self.unit_cost = float(round_trip_fee) + float(spread_cost)
        self.gross = float(gross)

    @property
    def warmup_bars(self) -> int:
        return self.n  # in strategy bars

    def spread_stats(self, closes: pd.DataFrame) -> pd.DataFrame:
        px = known_time_bars(closes[[self.a, self.b]], self.bar_minutes).dropna()
        la, lb = np.log(px[self.a]), np.log(px[self.b])
        ra, rb = la.diff(), lb.diff()
        roll = dict(window=self.n, min_periods=self.n)
        beta = ra.rolling(**roll).cov(rb) / rb.rolling(**roll).var()
        mean = la.rolling(**roll).mean() - beta * lb.rolling(**roll).mean()
        var = (la.rolling(**roll).var() + beta**2 * lb.rolling(**roll).var()
               - 2 * beta * la.rolling(**roll).cov(lb))
        spread = la - beta * lb
        sd = np.sqrt(var.clip(lower=0))
        return pd.DataFrame({
            "beta": beta,
            "spread": spread,
            "z": (spread - mean) / sd.replace(0, np.nan),
            "edge": (spread - mean).abs(),
            "cost": self.unit_cost * (1 + beta.abs()),
        })

    def signals(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Per strategy bar: stats plus position (+1 long spread, -1 short), hedge beta and events."""
        st = self.spread_stats(closes)
        pos = np.zeros(len(st))
        hedge = np.full(len(st), np.nan)
        event = np.array([""] * len(st), dtype=object)
        cur, entry_beta, entry_t = 0, np.nan, None
        z, edge, cost, beta = st["z"].to_numpy(), st["edge"].to_numpy(), st["cost"].to_numpy(), st["beta"].to_numpy()
        for i, t in enumerate(st.index):
            zi = z[i]
            if cur == 0:
                if (np.isfinite(zi) and beta[i] > 0 and abs(zi) >= self.z_entry
                        and edge[i] >= self.fee_multiple * cost[i]):
                    cur, entry_beta, entry_t = (-1 if zi > 0 else 1), beta[i], t
                    event[i] = "enter"
            elif np.isfinite(zi):
                reason = ""
                if (cur == -1 and zi <= self.z_exit) or (cur == 1 and zi >= -self.z_exit):
                    reason = "revert"
                elif abs(zi) >= self.z_stop and np.sign(zi) == -cur:
                    reason = "stop"
                elif t - entry_t >= self.max_hold:
                    reason = "timeout"
                if reason:
                    cur, event[i] = 0, reason
            pos[i] = cur
            hedge[i] = entry_beta if cur else np.nan
        st["position"], st["hedge_beta"], st["event"] = pos, hedge, event
        return st

    def generate_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        sig = self.signals(closes)
        hb = sig["hedge_beta"].fillna(0.0)
        scale = self.gross / (1.0 + hb.abs())
        bars = pd.DataFrame(0.0, index=sig.index, columns=closes.columns)
        bars[self.a] = sig["position"] * scale
        bars[self.b] = -sig["position"] * scale * hb
        bar = closes.index.to_series().diff().median()
        out = bars.reindex(closes.index + bar, method="ffill")
        out.index = closes.index
        return out.fillna(0.0)
