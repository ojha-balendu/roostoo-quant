import numpy as np
import pandas as pd

from quantbot.analysis.xsection import residual_returns
from quantbot.backtest.combined import (
    BtcTrendSleeve,
    HoldSleeve,
    MomentumSleeve,
    RsiSleeve,
    last_closed_row,
    simulate_combined,
)
from quantbot.backtest.composite_bt import simulate
from quantbot.backtest.momentum import MomentumConfig, momentum_inputs, sleeve_weights
from quantbot.rl.sim import SimConfig, simple_returns

CFG = SimConfig(initial_capital=1e6, fee=0.001, spread_bps=2.0, max_gross=1.0, max_weight=1.0, trade_band=0.0)


def prices(T=500, n=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=T, freq="1min", tz="UTC")
    return idx, 100 * np.exp(np.cumsum(rng.normal(0, 0.002, (T, n)), axis=0))


def test_hold_sleeve_equals_buy_and_hold():
    idx, P = prices()
    w = np.array([0.0, 0.0, 0.95])
    res = simulate_combined(P, idx, [HoldSleeve(1.0, w)], 0, len(P) - 1, CFG)
    e0, cost = 1e6, 0.001 + 1e-4
    assert abs(res["equity"].iloc[-1] - (e0 * (1 - 0.95 * cost) + 0.95 * e0 * (P[-1, 2] / P[0, 2] - 1))) < 1e-6
    assert res["trades"] == 1


def test_opposite_sleeves_net_out_with_no_trades_or_fees():
    idx, P = prices()
    long_btc, short_btc = np.array([0, 0, 0.5]), np.array([0, 0, -0.5])
    res = simulate_combined(P, idx, [HoldSleeve(0.5, long_btc), HoldSleeve(0.5, short_btc)], 0, len(P) - 1, CFG)
    assert res["trades"] == 0 and res["fees"] == 0 and abs(res["equity"].iloc[-1] - 1e6) < 1e-6


def test_rsi_sleeve_trades_only_changed_coins_and_leaves_other_sleeves_alone():
    idx, P = prices(T=50)
    states = np.zeros((50, 2), dtype=np.int8)
    states[10:30, 0] = 1                                                  # coin 0 long from minute 10 to 29
    rsi = RsiSleeve(0.5, states, np.array([0, 1]), unit=0.2)
    hold = HoldSleeve(0.5, np.array([0, 0, 0.9]))
    res = simulate_combined(P, idx, [hold, rsi], 0, 49, CFG)
    assert res["trades"] == 3                                             # BTC buy, coin 0 open, coin 0 close


def test_btc_trend_band_and_exit():
    idx, P = prices(T=10, n=1)
    target = np.array([0.5, 0.55, 0.8, 0.0, 0.0, 0.0, 0, 0, 0, 0])
    sleeve = BtcTrendSleeve(1.0, target, np.arange(10), np.ones(10, bool), btc_col=0, band=0.10)
    h = np.zeros(1)
    h = sleeve.decide(0, 1e6, h, True)
    assert h[0] == 0.5e6
    assert sleeve.decide(1, 1e6, h, False) is None                        # 0.55 vs 0.5: inside the band
    h = sleeve.decide(2, 1e6, h, False)
    assert h[0] == 0.8e6
    assert sleeve.decide(3, 1e6, h, False)[0] == 0.0                     # trend gone: exit fully


def test_momentum_sleeve_alone_matches_the_standalone_simulation():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2025-01-01", periods=24 * 70, freq="1h", tz="UTC")
    btc = rng.normal(0, 0.006, len(idx))
    cols = {"BTC/USD": btc} | {f"C{i:02d}/USD": (0.8 + 0.05 * i) * btc + rng.normal(0, 0.008, len(idx))
                               for i in range(14)}
    close = pd.DataFrame({k: 100 * np.exp(np.cumsum(v)) for k, v in cols.items()}, index=idx)
    resid, beta = residual_returns(close, "BTC/USD", beta_hours=24 * 20)
    coins = list(resid.columns)
    elig = pd.DataFrame(True, index=idx, columns=coins)
    mc = MomentumConfig(buffer_k=10)
    x = {k: v.reindex(columns=coins).to_numpy(dtype=float) for k, v in momentum_inputs(close, resid, elig, mc).items()}
    P = close[coins + ["BTC/USD"]].to_numpy()
    R, B, E = simple_returns(P), beta[coins].to_numpy(), elig.to_numpy()
    daily = np.asarray((idx + pd.Timedelta(hours=1)).hour == 0)
    a = 24 * 40
    ref = simulate(P, idx, lambda t, cur: sleeve_weights(t, cur, x, B, E, R, mc) if (daily[t] or t == a) else cur,
                   a, len(idx) - 1, CFG)
    sleeve = MomentumSleeve(1.0, mc, x, B, E, R, np.arange(len(idx)), daily, np.arange(len(coins)), len(coins))
    comb = simulate_combined(P, idx, [sleeve], a, len(idx) - 1, CFG)
    assert np.allclose(comb["equity"].to_numpy(), ref["equity"].to_numpy(), rtol=1e-9)


def test_last_closed_row_never_uses_an_unfinished_bar():
    hours = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    minutes = pd.DatetimeIndex(pd.to_datetime(["2026-01-01 00:58", "2026-01-01 00:59", "2026-01-01 01:30"], utc=True))
    rows = last_closed_row(hours + pd.Timedelta(hours=1), minutes)
    assert list(rows) == [-1, 0, 0]                                       # hour 00:00-01:00 known only at 00:59's close
