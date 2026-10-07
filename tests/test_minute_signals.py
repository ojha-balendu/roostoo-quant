import numpy as np
import pandas as pd

from quantbot.backtest.minute_signals import (
    macd_states,
    orders_per_minute,
    relative_rsi,
    rsi_states,
    signal_check,
    state_decider,
)


def frame(values, cols=("A",)):
    idx = pd.date_range("2026-01-01", periods=len(values), freq="1min", tz="UTC")
    return pd.DataFrame(np.asarray(values, dtype=float).reshape(len(values), -1), index=idx, columns=list(cols))


def test_rsi_entries_exits_and_flips():
    r = frame([50, 29, 35, 49, 50, 60, 71, 65, 51, 50, 25, 75, np.nan, 50])
    assert list(rsi_states(r)[:, 0]) == [0, 1, 1, 1, 0, 0, -1, -1, -1, 0, 1, -1, 0, 0]
    hold = rsi_states(frame([29, 50, 60, 71, 50, 31]), exit_mid=None)
    assert list(hold[:, 0]) == [1, 1, 1, -1, -1, -1]                    # hold until the opposite extreme


def test_macd_turns_with_the_price():
    # MACD above its signal line means the trend is strengthening, so check around a clear turning point
    logp = np.r_[np.linspace(0, 0.05, 150), np.linspace(0.05, 0.0, 150)]
    s = macd_states(frame(100 * np.exp(logp)))[:, 0]
    assert (s[40:140] == 1).all()                                       # rising: long
    assert (s[160:200] == -1).all()                                     # after the top: short


def test_relative_rsi_ignores_market_wide_moves():
    btc = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.002, 300)))
    coins = frame(np.column_stack([btc * 3.0, btc * 0.5]), cols=("A", "B"))
    rel = relative_rsi(coins, pd.Series(btc, index=coins.index))
    assert (rsi_states(rel) == 0).all()                                 # coins only followed BTC: no signal


def test_decider_trades_only_changed_coins_and_hedges():
    states = np.array([[0, 0], [1, 0], [1, -1], [1, -1], [0, -1]], dtype=np.int8)
    beta = np.array([[1.5, 0.5]] * 5)
    log = {}
    decide = state_decider(states, 0.1, beta, 0.95, first=0, hedge_band=0.0, order_log=log)
    cur = np.zeros(3)
    assert np.allclose(decide(0, cur), 0)
    w1 = decide(1, cur)
    assert np.allclose(w1, [0.1, 0, -0.15])                             # long A, BTC hedge = -1.5 x 0.1
    drifted = np.array([0.107, 0.0, -0.15])
    w2 = decide(2, drifted)
    assert w2[0] == 0.107 and w2[1] == -0.1 and abs(w2[2] - -(1.5 * 0.107 - 0.5 * 0.1)) < 1e-12
    assert decide(3, w2) is w2                                           # nothing changed: no trade at all
    assert log == {0: 0, 1: 2, 2: 2}                                     # orders actually sent per minute
    banded = state_decider(states, 0.1, beta, 0.95, first=0, hedge_band=0.2)
    w = banded(1, np.zeros(3))
    w = banded(2, w)
    assert w[2] == 0 and w[1] == -0.1                                    # hedge gap 0.1 < band: hedge untouched
    assert np.allclose(orders_per_minute(states, hedged=True), [0, 2, 2, 0, 2])
    assert np.allclose(orders_per_minute(states, hedged=False), [0, 1, 1, 0, 1])
    big = state_decider(np.ones((2, 2), dtype=np.int8), 1.0, None, 0.95, first=0)(0, np.zeros(2))
    assert abs(np.abs(big).sum() - 0.95) < 1e-12                        # gross cap


def test_signal_check_measures_moves_in_trade_direction():
    px = frame(np.r_[np.full(10, 100.0), np.full(10, 101.0)])          # +1% after minute 9
    states = np.zeros((20, 1), dtype=np.int8)
    states[9:] = 1                                                      # long entry at minute 9
    btc = pd.Series(100.0, index=px.index)
    sc = signal_check(states, px, btc, None, horizons=(1, 5))
    assert sc["entries"].tolist() == [1, 1] and np.allclose(sc["raw_bps"], 100.0)


def test_sliding_hourly_rsi_matches_clock_hours_and_updates_every_minute():
    from quantbot.analysis.indicators import rsi
    from quantbot.backtest.minute_signals import at_bar_closes, holding_minutes, sliding_bar_rsi

    rng = np.random.default_rng(2)
    idx = pd.date_range("2026-01-01", periods=60 * 24 * 5, freq="1min", tz="UTC")
    close = pd.DataFrame({"A": 100 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx))))}, index=idx)
    slide = sliding_bar_rsi(close, 14, 60)
    hourly = rsi(close.resample("1h").last(), 14)                      # bar labelled by its open time
    ends = slide[idx.minute == 59]
    ends.index = ends.index.floor("h")
    pd.testing.assert_frame_equal(ends.loc[hourly.index[20]:], hourly.loc[hourly.index[20]:], check_freq=False)
    assert slide["A"].iloc[-120:].nunique() > 100                       # a new value every minute
    checked = at_bar_closes(slide, 60)
    assert checked["A"].iloc[-60:-1].nunique() == 1                     # constant until the hour closes (xx:59)
    states = np.array([[0], [1], [1], [0], [-1], [-1], [-1], [0]], dtype=np.int8)
    assert holding_minutes(states) == 2.5                                # 5 minutes in position / 2 entries
