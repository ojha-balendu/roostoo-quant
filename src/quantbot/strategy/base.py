"""Strategy interface.

A strategy maps a panel of bar closes (index = bar open time UTC, columns = Roostoo
pairs like "BTC/USD") to target portfolio weights:

    weight > 0  -> long  that fraction of equity
    weight < 0  -> short that fraction of equity
    missing / 0 -> flat

Row t of `generate_weights` may only use data up to and including row t. The
backtester trades it at bar t + execution_delay_bars; live trading uses the last row.
The same method drives both, so research results carry over to production.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Strategy(ABC):
    name: str = "base"
    needs_quote_volume: bool = False  # live runner fetches hourly USD volumes when True

    def __init__(self, **params: Any):
        self.params = params
        self.context: dict[str, Any] = {}

    def set_context(self, **context: Any) -> None:
        """Extra live inputs (e.g. quote_volume, holdings, tradable), refreshed before each rebalance."""
        self.context = dict(context)

    @property
    def warmup_bars(self) -> int:
        """Minimum history needed before the strategy should trade."""
        return 1

    @abstractmethod
    def generate_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Return a weights frame with the same index/columns as `closes`."""

    def target_weights(self, closes: pd.DataFrame) -> dict[str, float]:
        if closes.empty:
            return {}
        last = self.generate_weights(closes).iloc[-1]
        return {p: float(w) for p, w in last.items() if w is not None and math.isfinite(w) and abs(w) > 1e-12}
