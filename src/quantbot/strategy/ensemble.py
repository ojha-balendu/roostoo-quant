"""Ensemble: a weighted average of other strategies' target weights.

Averaging a few reasonable configurations instead of picking the single best-looking one
reduces the risk of fitting noise. Configure in YAML:

  strategy:
    name: ensemble
    params:
      members:
        - {name: trend_vol, weight: 1, params: {...}}
        - {name: trend_vol, weight: 1, params: {...}}
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from quantbot.strategy.base import Strategy


class Ensemble(Strategy):
    name = "ensemble"

    def __init__(self, members: list[dict[str, Any]] | None = None, **params):
        super().__init__(members=members, **params)
        from quantbot.strategy.registry import get_strategy  # local import: registry imports this module

        if not members:
            raise ValueError("ensemble needs at least one member")
        built = [(get_strategy(m["name"], m.get("params") or {}), float(m.get("weight", 1.0))) for m in members]
        total = sum(w for _, w in built)
        if total <= 0:
            raise ValueError("ensemble member weights must sum to a positive number")
        self.members = [(s, w / total) for s, w in built]

    @property
    def warmup_bars(self) -> int:
        return max(s.warmup_bars for s, _ in self.members)

    def generate_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        total = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        for strategy, weight in self.members:
            w = strategy.generate_weights(closes).reindex(index=closes.index, columns=closes.columns)
            total = total + weight * w.fillna(0.0)
        return total
