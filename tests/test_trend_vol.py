import numpy as np
import pandas as pd

from quantbot.backtest.windows import evaluate_windows, summarize, window_bounds
from quantbot.strategy.registry import get_strategy
from quantbot.strategy.trend_vol import TrendVolTarget


def market(hours=24 * 60, drift=0.0, freq="1h", seed=2):
    rng = np.random.default_rng(seed)
    per_bar = 1 if freq == "1h" else 60
    n = hours * per_bar
    idx = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    s = 0.01 / np.sqrt(per_bar)
    f = rng.normal(drift / per_bar, s, n)
    cols = {"BTC/USD": f, "ETH/USD": 1.2 * f + rng.normal(0, s * 0.5, n), "SOL/USD": 1.4 * f + rng.normal(0, s, n)}
    return pd.DataFrame({k: 100 * np.exp(np.cumsum(v)) for k, v in cols.items()}, index=idx)


def test_regime_filter_blocks_wrong_side():
    for drift in (0.002, -0.002, 0.0):
        closes = market(drift=drift)
        w = TrendVolTarget(regime_ma_hours=168, regime_band=0.0).generate_weights(closes)
        btc = closes["BTC/USD"]  # hourly rows: row t's weights use closes up to and including row t
        ma = btc.rolling(168, min_periods=168).mean()
        bull, bear = btc > ma, btc <= ma
        assert (w[bull] >= 0).all().all(), "short while BTC above its average"
        assert (w[bear] <= 0).all().all(), "long while BTC below its average"
        assert (w[ma.isna()] == 0).all().all(), "traded before the regime was known"
    up = TrendVolTarget().generate_weights(market(drift=0.002))
    assert (up > 0).to_numpy().mean() > (up < 0).to_numpy().mean()  # mostly long in an uptrend


def test_vol_target_and_caps():
    closes = market(hours=24 * 120, drift=0.001)
    s = TrendVolTarget(target_vol=0.20, long_in_bear=1, short_in_bull=1)
    w = s.generate_weights(closes)
    assert (w.abs().sum(axis=1) <= 0.95 + 1e-9).all() and (w.abs() <= 0.25 + 1e-9).all().all()
    r = np.log(closes).diff()
    port = (w.shift(1) * r).sum(axis=1).iloc[24 * 30:]
    realised = port.std() * np.sqrt(24 * 365)
    assert 0.10 < realised < 0.30, realised  # near the 20% target (caps can bind)


def test_no_lookahead_hourly_and_minute():
    for closes in (market(hours=24 * 30), market(hours=24 * 12, freq="1min")):
        s = get_strategy("trend_vol", {"cov_hours": 48, "regime_ma_hours": 48, "lookbacks_hours": [6, 24]})
        full = s.generate_weights(closes)
        cut = closes.index[-300]
        pd.testing.assert_frame_equal(full.loc[:cut], s.generate_weights(closes.loc[:cut]), check_freq=False)


def test_windows_start_flat_and_cover_history():
    closes = market(hours=24 * 70)
    bounds = window_bounds(closes.index, 14)
    assert len(bounds) == 5 and bounds[-1][1] > closes.index[-1]
    flat = evaluate_windows(closes, closes * 0.0, window_days=14, spread_bps=0)
    assert np.allclose(flat["return"], 0.0) and (flat["fees"] == 0).all()
    hold = closes * 0.0
    hold["BTC/USD"] = 1.0
    res = evaluate_windows(closes, hold, window_days=14, spread_bps=0, delay_bars=0, threshold=0.0,
                           cash_buffer=0.0)
    for _, row in res.iterrows():  # each window: BTC return minus one entry fee
        c = closes.loc[(closes.index >= row["start"]) & (closes.index < row["end"]), "BTC/USD"]
        assert abs((1 + row["return"]) - (c.iloc[-1] / c.iloc[0]) / 1.001) < 1e-6
    s = summarize(res, versus=res)
    assert s["windows"] == 5 and s["pct_beats_benchmark"] == 0.0


def test_regime_band_reduces_regime_switches():
    closes = market(hours=24 * 90)
    btc = closes["BTC/USD"]
    flips = {}
    for band in (0.0, 0.02):
        w = TrendVolTarget(regime_band=band, long_in_bear=0.0, short_in_bull=0.0).generate_weights(closes)
        side = np.sign(w.sum(axis=1)).replace(0, np.nan).ffill()
        flips[band] = int((side.diff().abs() > 0).sum())
    assert flips[0.02] <= flips[0.0]
    assert len(btc) > 0


def test_tanh_signal_trades_less_than_sign():
    closes = market(hours=24 * 90)
    churn = {}
    for sig in ("sign", "tanh"):
        w = TrendVolTarget(signal=sig, long_in_bear=1, short_in_bull=1).generate_weights(closes)
        churn[sig] = w.diff().abs().sum(axis=1).mean()
    assert churn["tanh"] < churn["sign"]
