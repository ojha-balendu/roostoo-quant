import numpy as np
import pandas as pd

from quantbot.backtest.composite_bt import simulate
from quantbot.backtest.switching import simulate_with_stop
from quantbot.rl.sim import SimConfig

CFG = SimConfig(initial_capital=100_000, fee=0.001, spread_bps=2, max_gross=1, max_weight=1, trade_band=0.0)


def path():
    idx = pd.date_range("2026-01-01", periods=60, freq="1h", tz="UTC")
    a = np.r_[np.linspace(100, 110, 20), np.linspace(110, 95, 40)]           # up, then a steady fall
    btc = np.full(60, 50_000.0)
    return idx, np.column_stack([a, btc])


def hold_a(t, cur):
    return np.array([0.9, 0.0]) if t == 0 else cur                          # buy once, then hold


def make_flat(first):
    return lambda t, cur: np.zeros(2)


def test_no_stop_equals_running_book_a():
    idx, P = path()
    ref = simulate(P, idx, hold_a, 0, 59, CFG)
    res = simulate_with_stop(P, idx, hold_a, make_flat, 0, 59, CFG, stop=None)
    assert np.allclose(res["equity"].to_numpy(), ref["equity"].to_numpy()) and res["switched_at"] is None


def test_switches_at_the_first_bar_below_the_stop_and_book_b_takes_over():
    idx, P = path()
    res = simulate_with_stop(P, idx, hold_a, make_flat, 0, 59, CFG, stop=0.05)
    eq = res["equity"].to_numpy()
    peak = np.maximum.accumulate(eq)
    first_breach = int(np.argmax(eq <= peak * 0.95))
    assert res["switch_index"] == first_breach                              # bar t is judged on equity at t
    assert res["drawdown_at_switch"] <= -0.05 and res["drawdown_at_switch"] > -0.07
    after = eq[first_breach + 1:]
    assert np.allclose(after, after[0])                                     # book B (flat) holds nothing
    assert res["trades"] == 2                                               # buy A, then sell it at the switch
