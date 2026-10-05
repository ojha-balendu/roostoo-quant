"""Volatility-managed exposure: hold more when predicted volatility is low, less when high.

No return forecast at all. A base portfolio (fixed assets, equal or inverse-volatility
weights) is scaled every hour so its predicted volatility equals `target_vol`:

  r_p[t]     = hourly return of the base portfolio
  var_hat[t] = exponentially weighted mean of r_p^2 (half-life `halflife_hours`)   (RiskMetrics)
  sigma_hat  = sqrt(var_hat * hours_per_year)
  exposure   = min(target_vol / sigma_hat, max_gross)

The bet is that volatility is predictable (it clusters) while returns are not, so cutting
exposure in turbulent periods improves Sharpe, Sortino and Calmar (Moreira & Muir, 2017).
Row t only uses prices known at the close of row t.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from quantbot.strategy.base import Strategy
from quantbot.strategy.trend_vol import HOURS_PER_YEAR, known_hourly


class VolManaged(Strategy):
    name = "vol_managed"

    def __init__(
        self,
        assets: Sequence[str] = ("BTC/USD",),
        base: str = "inverse_vol",
        target_vol: float = 0.40,
        halflife_hours: float = 72,
        max_gross: float = 0.95,
        base_vol_hours: int = 720,
        min_hours: int = 168,
        **params,
    ):
        if base not in ("equal", "inverse_vol"):
            raise ValueError("base must be 'equal' or 'inverse_vol'")
        super().__init__(assets=list(assets), base=base, target_vol=target_vol, halflife_hours=halflife_hours,
                         max_gross=max_gross, base_vol_hours=base_vol_hours, min_hours=min_hours, **params)
        self.assets = list(assets)
        self.base = base
        self.target_vol = float(target_vol)
        self.halflife = float(halflife_hours)
        self.max_gross = float(max_gross)
        self.base_vol_hours = int(base_vol_hours)
        self.min_hours = int(min_hours)

    @property
    def warmup_bars(self) -> int:
        return max(self.min_hours, self.base_vol_hours // 4) + 2  # hourly bars

    def base_weights(self, r: pd.DataFrame) -> pd.DataFrame:
        avail = r.notna()
        if self.base == "equal" or r.shape[1] == 1:
            raw = avail.astype(float)
        else:
            vol = r.rolling(self.base_vol_hours, min_periods=self.base_vol_hours // 4).std()
            raw = (1.0 / vol).where(avail & (vol > 0))
        return raw.div(raw.sum(axis=1), axis=0).fillna(0.0)

    def predicted_vol(self, closes: pd.DataFrame) -> pd.Series:
        """Annualised predicted volatility of the base portfolio, per hour (known time)."""
        r = np.log(known_hourly(closes)[[a for a in self.assets if a in closes.columns]]).diff()
        base = self.base_weights(r)
        port = (base.shift(1) * r.fillna(0.0)).sum(axis=1)
        var = port.pow(2).ewm(halflife=self.halflife, min_periods=self.min_hours).mean()
        return np.sqrt(var * HOURS_PER_YEAR)

    def hourly_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        r = np.log(known_hourly(closes)[[a for a in self.assets if a in closes.columns]]).diff()
        base = self.base_weights(r)
        sigma = self.predicted_vol(closes)
        exposure = (self.target_vol / sigma).clip(upper=self.max_gross).fillna(0.0)
        return base.mul(exposure, axis=0).reindex(columns=closes.columns, fill_value=0.0)

    def generate_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        hourly = self.hourly_weights(closes)
        bar = closes.index.to_series().diff().median()
        out = hourly.reindex(closes.index + bar, method="ffill")
        out.index = closes.index
        return out.fillna(0.0)
