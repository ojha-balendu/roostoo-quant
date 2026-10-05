"""Trend following with volatility targeting and a market regime filter (#1 + #2).

Every hour, for each coin in the universe:
  z_L        = log return over the last L hours / (hourly vol * sqrt(L))   trend strength in std devs
  signal_i   = average over lookbacks L of tanh(z_L)        -> in [-1, 1]   ("tanh", default)
               or of sign(z_L)                                              ("sign": all-or-nothing)
  raw_i      = signal_i / vol_i         (inverse-volatility: each coin contributes similar risk)
  The continuous tanh signal scales positions with trend strength, so a price wobbling around
  zero trend gives small positions instead of flipping fully long/short (that churn costs fees).

Regime filter (BTC vs its moving average over `regime_ma_hours`, with a +/- `regime_band` buffer:
the regime only switches once BTC is that far beyond the average, which avoids flip-flopping):
  bull:  shorts scaled by `short_in_bull`  (default 0 -> no shorts)
  bear:  longs  scaled by `long_in_bear`   (default 0 -> no longs)

Portfolio volatility targeting:
  normalise raw weights to gross 1, estimate the portfolio's volatility by applying those
  weights to the last `cov_hours` of hourly returns (this accounts for correlations), then
  scale so expected volatility = target_vol (annualised), capped at max_gross, and each
  coin capped at max_weight.

Works on any bar size; row t only uses prices known at the close of row t.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from quantbot.strategy.base import Strategy

HOURS_PER_YEAR = 24 * 365


def known_hourly(closes: pd.DataFrame) -> pd.DataFrame:
    bar = closes.index.to_series().diff().median()
    shifted = closes.copy()
    shifted.index = closes.index + bar
    return shifted.resample("1h", label="right", closed="right").last()


class TrendVolTarget(Strategy):
    name = "trend_vol"

    def __init__(
        self,
        lookbacks_hours: Sequence[int] = (24, 72, 168),
        asset_vol_hours: int = 72,
        cov_hours: int = 168,
        target_vol: float = 0.30,
        max_gross: float = 0.95,
        max_weight: float = 0.25,
        regime_asset: str = "BTC/USD",
        regime_ma_hours: int = 168,
        long_in_bear: float = 0.0,
        short_in_bull: float = 0.0,
        regime_band: float = 0.01,
        signal: str = "tanh",
        assets: Sequence[str] | None = None,
        **params,
    ):
        if signal not in ("tanh", "sign"):
            raise ValueError("signal must be 'tanh' or 'sign'")
        super().__init__(lookbacks_hours=list(lookbacks_hours), asset_vol_hours=asset_vol_hours,
                         cov_hours=cov_hours, target_vol=target_vol, max_gross=max_gross,
                         max_weight=max_weight, regime_asset=regime_asset, regime_ma_hours=regime_ma_hours,
                         long_in_bear=long_in_bear, short_in_bull=short_in_bull, regime_band=regime_band,
                         signal=signal, assets=list(assets) if assets else None, **params)
        self.lookbacks = [int(x) for x in lookbacks_hours]
        self.asset_vol_hours = int(asset_vol_hours)
        self.cov_hours = int(cov_hours)
        self.target_vol = float(target_vol)
        self.max_gross = float(max_gross)
        self.max_weight = float(max_weight)
        self.regime_asset = regime_asset
        self.regime_ma_hours = int(regime_ma_hours)
        self.long_in_bear = float(long_in_bear)
        self.short_in_bull = float(short_in_bull)
        self.regime_band = float(regime_band)
        self.signal = signal
        self.assets = list(assets) if assets else None

    @property
    def warmup_bars(self) -> int:
        return max(self.lookbacks + [self.cov_hours, self.regime_ma_hours]) + 2  # hourly bars

    def hourly_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        px = known_hourly(closes)
        regime_px = px[self.regime_asset] if self.regime_asset in px else None
        if self.assets:
            px = px[[a for a in self.assets if a in px.columns]]
        lp = np.log(px)
        r = lp.diff()

        vol = r.rolling(self.asset_vol_hours, min_periods=self.asset_vol_hours // 2).std()
        squash = np.tanh if self.signal == "tanh" else np.sign
        signal = sum(squash((lp - lp.shift(L)) / (vol * np.sqrt(L))).fillna(0.0)
                     for L in self.lookbacks) / len(self.lookbacks)
        raw = (signal / vol).where(vol > 0).fillna(0.0)

        if regime_px is not None:
            ma = regime_px.rolling(self.regime_ma_hours, min_periods=self.regime_ma_hours).mean()
            known = ma.notna()
            state = pd.Series(np.where(regime_px > ma * (1 + self.regime_band), 1.0,
                                       np.where(regime_px < ma * (1 - self.regime_band), 0.0, np.nan)),
                              index=regime_px.index).ffill()
            bull = state.fillna((regime_px > ma).astype(float)).astype(bool)
            long_scale = np.where(bull, 1.0, self.long_in_bear)
            short_scale = np.where(bull, self.short_in_bull, 1.0)
            raw = raw.clip(lower=0).mul(long_scale, axis=0) + raw.clip(upper=0).mul(short_scale, axis=0)
            raw = raw.mul(known.astype(float), axis=0)  # flat until the regime is known

        gross = raw.abs().sum(axis=1)
        unit = raw.div(gross.where(gross > 0), axis=0).fillna(0.0)

        # Portfolio vol targeting with the realised covariance of the last cov_hours.
        R = r.fillna(0.0).to_numpy()
        U = unit.to_numpy()
        scale = np.zeros(len(U))
        for t in range(self.cov_hours, len(U)):
            if not U[t].any():
                continue
            port = R[t - self.cov_hours + 1: t + 1] @ U[t]
            sigma = port.std() * np.sqrt(HOURS_PER_YEAR)
            scale[t] = min(self.target_vol / sigma, self.max_gross) if sigma > 0 else 0.0
        w = unit.mul(scale, axis=0).clip(-self.max_weight, self.max_weight)
        return w.reindex(columns=closes.columns, fill_value=0.0)

    def generate_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        hourly = self.hourly_weights(closes)
        bar = closes.index.to_series().diff().median()
        out = hourly.reindex(closes.index + bar, method="ffill")
        out.index = closes.index
        return out.fillna(0.0)
