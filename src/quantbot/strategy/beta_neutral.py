"""Beta-neutral alt basket: long every alt, short BTC by the basket's beta.

Every hour:
  1. Estimate each coin's beta to BTC from hourly returns over the last `beta_lookback_hours`.
     * Dimson correction: alts react to BTC with a small lag, so beta is the sum of the
       coefficients on BTC's return in the same hour and the previous `dimson_lags` hours.
     * Returns are clipped at +/- `clip_sigma` rolling standard deviations (outlier control).
     * Betas are shrunk toward the cross-sectional mean beta:
           beta_adj = w * beta_raw + (1 - w) * mean(beta_raw over all alts)
       (your 0.2 + 0.8 * cov/var formula, with the crypto average instead of 1 as the target).
  2. Long book: equal weight (or inverse volatility) across alts with a beta estimate.
  3. BTC short = long book x its average beta, so total beta to BTC is ~0.
  4. Scale so long + short gross = `gross` (Roostoo shorts need full collateral).

Works on any bar size (1m, 15m, 1h): row t only uses prices known at the close of row t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantbot.strategy.base import Strategy


class BetaNeutralBasket(Strategy):
    name = "beta_neutral"

    def __init__(
        self,
        benchmark: str = "BTC/USD",
        beta_lookback_hours: int = 720,
        min_hours: int = 240,
        dimson_lags: int = 2,
        shrink_weight: float = 0.8,
        clip_sigma: float = 5.0,
        gross: float = 0.95,
        sizing: str = "equal",
        vol_lookback_hours: int = 168,
        **params,
    ):
        super().__init__(benchmark=benchmark, beta_lookback_hours=beta_lookback_hours, min_hours=min_hours,
                         dimson_lags=dimson_lags, shrink_weight=shrink_weight, clip_sigma=clip_sigma,
                         gross=gross, sizing=sizing, vol_lookback_hours=vol_lookback_hours, **params)
        if sizing not in ("equal", "inverse_vol"):
            raise ValueError("sizing must be 'equal' or 'inverse_vol'")
        self.benchmark = benchmark
        self.lookback = int(beta_lookback_hours)
        self.min_hours = int(min_hours)
        self.lags = int(dimson_lags)
        self.w = float(shrink_weight)
        self.clip_sigma = float(clip_sigma)
        self.gross = float(gross)
        self.sizing = sizing
        self.vol_lookback = int(vol_lookback_hours)

    @property
    def warmup_bars(self) -> int:
        return self.min_hours + self.lags + 2  # in hourly bars; scale up for finer live bars

    # ------------------------------------------------------------------ pieces
    @staticmethod
    def _hourly_closes(closes: pd.DataFrame) -> pd.DataFrame:
        """Hourly closes labelled by the time they are KNOWN (end of the hour)."""
        bar = closes.index.to_series().diff().median()
        known = closes.copy()
        known.index = closes.index + bar  # a bar labelled 10:59 closes (is known) at 11:00
        return known.resample("1h", label="right", closed="right").last()

    def hourly_returns(self, closes: pd.DataFrame) -> pd.DataFrame:
        r = np.log(self._hourly_closes(closes)).diff()
        std = r.rolling(self.lookback, min_periods=self.min_hours).std()
        return r.clip(lower=-self.clip_sigma * std, upper=self.clip_sigma * std)

    def hourly_betas(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Shrunk Dimson betas, one row per hour (index = time the estimate is known)."""
        r = self.hourly_returns(closes)
        b = r[self.benchmark]
        var = b.rolling(self.lookback, min_periods=self.min_hours).var()
        raw = sum(r.rolling(self.lookback, min_periods=self.min_hours).cov(b.shift(k)) for k in range(self.lags + 1))
        raw = raw.div(var, axis=0)
        alts = [c for c in raw.columns if c != self.benchmark]
        prior = raw[alts].mean(axis=1)
        adj = raw.mul(self.w).add(prior * (1 - self.w), axis=0)
        adj[self.benchmark] = 1.0
        return adj

    def hourly_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        betas = self.hourly_betas(closes)
        alts = [c for c in betas.columns if c != self.benchmark]
        B = betas[alts]
        valid = B.notna()
        if self.sizing == "inverse_vol":
            vol = self.hourly_returns(closes)[alts].rolling(self.vol_lookback, min_periods=24).std()
            raw = (1.0 / vol).where(valid & (vol > 0))
        else:
            raw = valid.astype(float).where(valid)
        long_w = raw.div(raw.sum(axis=1), axis=0)          # long book, sums to 1 per row
        basket_beta = (long_w * B).sum(axis=1, min_count=1)  # beta of $1 of long book
        scale = self.gross / (1.0 + basket_beta)
        weights = long_w.mul(scale, axis=0)
        weights[self.benchmark] = -scale * basket_beta
        return weights.fillna(0.0)

    # --------------------------------------------------------------- interface
    def generate_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        hourly = self.hourly_weights(closes)
        bar = closes.index.to_series().diff().median()
        known = closes.index + bar
        out = hourly.reindex(known, method="ffill")
        out.index = closes.index
        return out.reindex(columns=closes.columns).fillna(0.0)
