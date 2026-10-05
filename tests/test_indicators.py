import numpy as np
import pandas as pd

from quantbot.analysis import indicators as ind
from quantbot.analysis.xsection import signal_periods


def ohlcv(n=400, coins=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    cols = [f"C{i}/USD" for i in range(coins)]
    c = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, coins)), axis=0)), index=idx, columns=cols)
    o = c.shift(1).fillna(c)
    spread = c * rng.uniform(0.002, 0.02, (n, coins))
    h, l = np.fmax(o, c) + spread, np.fmin(o, c) - spread
    v = pd.DataFrame(rng.lognormal(10, 0.5, (n, coins)), index=idx, columns=cols)
    return o, h, l, c, v, v * c, v * c * rng.uniform(0.3, 0.7, (n, coins))


def test_rsi_extremes_and_midpoint():
    idx = pd.date_range("2026", periods=60, freq="4h", tz="UTC")
    up = pd.DataFrame({"A": np.arange(60, dtype=float) + 100}, index=idx)
    zigzag = pd.DataFrame({"A": 100 + np.tile([0.0, 1.0], 30)}, index=idx)
    assert ind.rsi(up).iloc[-1, 0] == 100.0
    assert abs(ind.rsi(zigzag).iloc[-1, 0] - 50) < 5


def test_oscillator_identities_and_ranges():
    o, h, l, c, v, qv, tbq = ohlcv()
    st, wr = ind.stochastic(h, l, c), ind.williams_r(h, l, c)
    assert np.allclose((st - 100).dropna(), wr.dropna())               # Williams %R = stochastic - 100
    assert st.stack().dropna().between(0, 100).all() and ind.rsi(c).stack().dropna().between(0, 100).all()
    assert ind.chaikin_money_flow(h, l, c, v).stack().dropna().between(-1, 1).all()
    assert ind.obv_flow(c, v).stack().dropna().between(-1, 1).all()
    assert ind.aroon(h, l).stack().dropna().between(-100, 100).all() and ind.cmo(c).stack().dropna().between(-100, 100).all()


def test_cci_matches_textbook_formula():
    o, h, l, c, v, qv, tbq = ohlcv(n=60, coins=1)
    tp = ((h + l + c) / 3).iloc[-20:, 0]
    expected = (tp.iloc[-1] - tp.mean()) / (0.015 * np.mean(np.abs(tp - tp.mean())))
    assert abs(ind.cci(h, l, c).iloc[-1, 0] - expected) < 1e-9


def test_aroon_on_rising_prices():
    idx = pd.date_range("2026", periods=40, freq="4h", tz="UTC")
    rising = pd.DataFrame({"A": np.arange(40, dtype=float)}, index=idx)
    assert ind.aroon(rising + 1, rising).iloc[-1, 0] == 100.0           # newest bar is the high, oldest the low


def test_no_indicator_uses_future_data():
    o, h, l, c, v, qv, tbq = ohlcv()
    full = ind.compute_all(o, h, l, c, v, qv, tbq)
    cut = c.index[250]
    part = ind.compute_all(*(x.loc[:cut] for x in (o, h, l, c, v, qv, tbq)))
    assert len(full) == 23
    for name in full:
        pd.testing.assert_frame_equal(full[name].loc[:cut], part[name], check_freq=False, obj=name)
        assert full[name].iloc[100:].notna().mean().mean() > 0.9, name   # mostly defined after warm-up


def test_signal_periods_ic_detects_a_perfect_signal():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2026", periods=300, freq="4h", tz="UTC")
    cols = [f"C{i}/USD" for i in range(12)]
    resid = pd.DataFrame(rng.normal(0, 0.01, (300, 12)), index=idx, columns=cols)
    future = resid.rolling(3).sum().shift(-3)                          # the next 3 bars' outcome
    beta = pd.DataFrame(1.0, index=idx, columns=cols)
    elig = pd.DataFrame(True, index=idx, columns=cols)
    p = signal_periods(future, resid, beta, elig, horizon=3, min_coins=5)
    assert p["ic"].mean() > 0.99 and p["spread_bps"].mean() > 0
    noise = pd.DataFrame(rng.normal(size=(300, 12)), index=idx, columns=cols)
    assert abs(signal_periods(noise, resid, beta, elig, horizon=3, min_coins=5)["ic"].mean()) < 0.1
