from __future__ import annotations

import pandas as pd

from quantbot.strategy.base import Strategy


class CrossSectionalMomentum(Strategy):
    """PLACEHOLDER (not research-validated): long the top-N trailing performers,
    short the bottom-N, equal weight. Exists to exercise all four actions end to end."""

    name = "xs_momentum"

    def __init__(self, lookback_bars: int = 96, top_n: int = 3, bottom_n: int = 3, gross: float = 0.6, **params):
        super().__init__(lookback_bars=lookback_bars, top_n=top_n, bottom_n=bottom_n, gross=gross, **params)
        self.lookback = int(lookback_bars)
        self.top_n = int(top_n)
        self.bottom_n = int(bottom_n)
        self.gross = float(gross)

    @property
    def warmup_bars(self) -> int:
        return self.lookback + 1

    def generate_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        momentum = closes / closes.shift(self.lookback) - 1.0
        ranks = momentum.rank(axis=1, ascending=False, method="first")  # 1 = strongest
        n_valid = momentum.notna().sum(axis=1)
        enough = n_valid >= (self.top_n + self.bottom_n)
        longs = ranks.le(self.top_n) & momentum.notna()
        shorts = ranks.gt(n_valid - self.bottom_n, axis=0) & momentum.notna()
        per_leg = self.gross / max(self.top_n + self.bottom_n, 1)
        weights = longs.astype(float) * per_leg - shorts.astype(float) * per_leg
        return weights.where(enough, 0.0, axis=0).fillna(0.0)
