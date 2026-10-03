from __future__ import annotations

import pandas as pd

from quantbot.strategy.base import Strategy


class Flat(Strategy):
    """Holds nothing. Useful for plumbing tests and as a safe default."""

    name = "flat"

    def generate_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
