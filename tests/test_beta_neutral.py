import numpy as np
import pandas as pd

from quantbot.backtest.engine import run_backtest
from quantbot.strategy.beta_neutral import BetaNeutralBasket
from quantbot.strategy.registry import get_strategy

BETAS = {"BTC/USD": 1.0, "A/USD": 0.5, "B/USD": 1.5, "C/USD": 2.0}


def synthetic(freq="1h", periods=24 * 40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=periods, freq=freq, tz="UTC")
    scale = 0.01 if freq == "1h" else 0.01 / np.sqrt(60)
    btc = rng.normal(0, scale, periods)
    cols = {p: b * btc + (0 if p == "BTC/USD" else rng.normal(0, scale * 0.5, periods)) for p, b in BETAS.items()}
    return pd.DataFrame({p: 100 * np.exp(np.cumsum(r)) for p, r in cols.items()}, index=idx)


def test_betas_recovered_and_book_is_neutral():
    closes = synthetic()
    s = BetaNeutralBasket(beta_lookback_hours=480, min_hours=200, shrink_weight=1.0, dimson_lags=0)
    betas = s.hourly_betas(closes).iloc[-1]
    for p, b in BETAS.items():
        assert abs(betas[p] - b) < 0.15, (p, betas[p])
    w = s.generate_weights(closes).iloc[-1]
    assert abs(sum(w[p] * betas[p] for p in BETAS)) < 1e-9  # sum(w_i * beta_i) == 0
    assert abs(w.abs().sum() - 0.95) < 1e-9 and w["BTC/USD"] < 0


def test_shrinkage_pulls_toward_cross_sectional_mean():
    closes = synthetic()
    raw = BetaNeutralBasket(shrink_weight=1.0, dimson_lags=0, min_hours=200).hourly_betas(closes).iloc[-1]
    shr = BetaNeutralBasket(shrink_weight=0.8, dimson_lags=0, min_hours=200).hourly_betas(closes).iloc[-1]
    mean = raw[["A/USD", "B/USD", "C/USD"]].mean()
    for p in ["A/USD", "B/USD", "C/USD"]:
        assert abs(shr[p] - (0.8 * raw[p] + 0.2 * mean)) < 1e-9


def test_no_lookahead_on_minute_data():
    closes = synthetic(freq="1min", periods=60 * 24 * 12)
    s = BetaNeutralBasket(beta_lookback_hours=200, min_hours=100)
    full = s.generate_weights(closes)
    cut = closes.index[-500]
    partial = s.generate_weights(closes.loc[:cut])
    pd.testing.assert_frame_equal(full.loc[:cut], partial, check_freq=False)
    changes = full.diff().abs().sum(axis=1) > 0
    assert set(full.index[changes][1:].minute) <= {59}  # weights only change when an hour is complete


def test_registered_and_hourly_trading_in_backtest():
    assert isinstance(get_strategy("beta_neutral"), BetaNeutralBasket)
    closes = synthetic(freq="1min", periods=60 * 24 * 12)
    w = BetaNeutralBasket(beta_lookback_hours=200, min_hours=100).generate_weights(closes)
    res = run_backtest(closes, w, threshold=0.0, min_trade_usd=1, trade_on=closes.index.minute == 0)
    assert len(res.trades) and set(pd.DatetimeIndex(res.trades["ts"]).minute) == {0}
