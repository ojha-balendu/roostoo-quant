import numpy as np
import pandas as pd
import pytest

from quantbot.analysis.composite import COEFFS, affine_unit, composite, normalise, term_shares
from quantbot.backtest.composite_bt import (
    CompositeConfig,
    calibrate,
    run_metrics,
    simulate,
    v1_weights,
    v2_weights,
)


def panel(T=600, N=20, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=T, freq="1h", tz="UTC")
    cols = [f"C{i}/USD" for i in range(N)]
    return idx, cols, rng


# ------------------------------------------------------------------------------------ the signal
def test_affine_maps_each_row_to_minus_one_plus_one():
    idx, cols, rng = panel(T=50)
    s = pd.DataFrame(rng.normal(size=(50, 20)) * 7 + 3, index=idx, columns=cols)
    a = affine_unit(s)
    assert np.allclose(a.min(axis=1), -1) and np.allclose(a.max(axis=1), 1)
    assert (a.rank(axis=1) == s.rank(axis=1)).all().all()          # order preserved


def test_zscore_normalisation_and_raw_mode_dominance():
    idx, cols, rng = panel(T=40)
    elig = pd.DataFrame(True, index=idx, columns=cols)
    big = pd.DataFrame(rng.lognormal(20, 2, (40, 20)), index=idx, columns=cols)    # like raw volume
    small = pd.DataFrame(rng.normal(0, 1, (40, 20)), index=idx, columns=cols)
    z = normalise(big, elig)
    assert np.allclose(z.mean(axis=1), 0, atol=0.2) and (z.abs() <= 3).all().all()
    named = {k: small for k in COEFFS}
    named["volume"] = big
    raw = term_shares(named, elig, method="raw", every=1)
    zs = term_shares(named, elig, method="zscore", every=1)
    assert raw["volume"] > 0.99 and zs["volume"] < 0.5               # raw units let volume swamp everything


def test_composite_requires_every_indicator():
    idx, cols, rng = panel(T=10)
    elig = pd.DataFrame(True, index=idx, columns=cols)
    with pytest.raises(ValueError):
        composite({"rsi": pd.DataFrame(0.0, index=idx, columns=cols)}, elig)


# ----------------------------------------------------------------------------------- strategies
def test_v1_deciles_equal_weights_beta_neutral():
    cfg = CompositeConfig()
    rng = np.random.default_rng(1)
    sp, beta = rng.uniform(-1, 1, 50), rng.uniform(0.5, 2, 50)
    w = v1_weights(sp, np.ones(50, bool), beta, cfg)
    order = np.argsort(sp)
    assert (w[order[-5:]] > 0).all() and (w[order[:5]] < 0).all() and np.count_nonzero(w[:50]) == 10
    assert abs(np.dot(w[:50], beta) + w[50]) < 1e-12                 # coins' beta + BTC hedge = 0
    assert abs(np.abs(w).sum() - cfg.gross) < 1e-12


def test_v2_trades_only_when_expected_profit_beats_fees():
    cfg = CompositeConfig(min_coins=1)
    c, beta, elig = cfg.cost, np.ones(10), np.ones(10, bool)
    unit = cfg.gross * 0.8 / (2 * 1)                                   # decile 10% of 10 coins -> 1 per side
    flat = np.zeros(11)

    def units(A, cur):
        return v2_weights(np.array(A, float), cur, elig, beta, cfg)[:10] / unit

    A = np.zeros(10)
    A[0], A[1], A[2] = 2.5 * c, 1.5 * c, -2.5 * c
    u = units(A, flat)
    assert u[0] == 1 and u[1] == 0 and u[2] == -1 and (u[3:] == 0).all()   # open only if |A| > entry + exit
    held = np.zeros(11)
    held[:4] = unit
    A2 = np.array([0.5 * c, -0.5 * c, -1.5 * c, -2.5 * c] + [0.0] * 6)
    u2 = units(A2, held)
    assert u2[0] == 1                                                    # still positive: keep (exit fee is sunk either way)
    assert u2[1] == 0 and u2[2] == 0                                     # turned negative: close
    assert u2[3] == -1                                                   # negative beyond 2c: flip
    drifted = held.copy()
    drifted[0] = unit * 1.07                                             # price moved the position 7%
    w = v2_weights(A2, drifted, elig, beta, cfg)
    assert w[0] == drifted[0]                                            # kept exactly: no resizing trade


def test_v2_capacity_takes_best_expected_profit_and_favours_held():
    cfg = CompositeConfig(min_coins=1)                                   # 10 coins -> k = 1 -> at most 2 positions
    c, beta, elig = cfg.cost, np.ones(10), np.ones(10, bool)
    A = np.array([3.5, 4.0, 5.0, -6.0, 0, 0, 0, 0, 0, 0]) * c          # balanced book: one long, one short
    w = v2_weights(A, np.zeros(11), elig, beta, cfg)
    assert np.count_nonzero(w[:10]) == 2 and w[3] < 0 and w[2] > 0 and w[0] == w[1] == 0   # best two by |A|
    held = np.zeros(11)
    held[0] = cfg.gross * 0.8 / 2
    w2 = v2_weights(A, held, elig, beta, cfg)
    assert w2[0] == held[0] and w2[3] < 0 and w2[2] == 0                 # held 3.5c (+2c bonus) beats a new 5c


def test_calibration_uses_only_past_outcomes_and_recovers_slope():
    idx, cols, rng = panel(T=800)
    sp = pd.DataFrame(rng.uniform(-1, 1, (800, 20)), index=idx, columns=cols)
    resid = sp.shift(1) * 0.002 + rng.normal(0, 0.004, (800, 20))      # next-bar return = 0.002 * s' + noise
    cfg = CompositeConfig(calib_bars=200, min_coins=5)
    cal = calibrate(sp, resid, cfg)
    assert abs(cal["b"].iloc[-1] - 0.002) < 0.0005
    cut = idx[500]
    part = calibrate(sp.loc[:cut], resid.loc[:cut], cfg)
    pd.testing.assert_series_equal(cal["b"].loc[:idx[499]], part["b"].loc[:idx[499]], check_freq=False)
    flipped = calibrate(sp, -resid, cfg)
    assert (flipped["A"].iloc[-100:].abs().max().max() == 0)            # inverted signal -> stay out (no flip)


# ----------------------------------------------------------------------------------- simulation
def prices_and_resid(T=600, N=20, seed=2):
    idx, cols, rng = panel(T, N, seed)
    r = rng.normal(0, 0.01, (T, N + 1))
    P = 100 * np.exp(np.cumsum(r, axis=0))
    P[0] = 100.0
    return idx, cols, P


def test_hold_btc_and_accounting():
    idx, cols, P = prices_and_resid()
    cfg = CompositeConfig(capital=1e6)
    target = np.zeros(P.shape[1])
    target[-1] = 0.95
    res = simulate(P, idx, lambda t, cur: target if t == 0 else cur, 0, len(P) - 1, cfg.sim_config())  # buy & hold
    e0 = 1e6
    btc = P[-1, -1] / P[0, -1]
    assert abs(res["equity"].iloc[-1] - (e0 * (1 - 0.95 * cfg.cost) + 0.95 * e0 * (btc - 1))) < 1e-6
    assert res["trades"] == 1
    m = run_metrics(res, e0)
    assert abs(m["fees_pct"] - 0.95 * cfg.cost) < 1e-12


def test_perfect_foresight_canary_proves_timing():
    """If the signal is next bar's return, v1 must win big; with the signal lagged one bar it must not."""
    idx, cols, P = prices_and_resid(T=400)
    nxt = P[1:, :-1] / P[:-1, :-1] - 1
    sp_future = np.vstack([nxt, np.zeros((1, nxt.shape[1]))])          # row t = return from t to t+1
    beta, elig = np.zeros(len(cols)), np.ones(len(cols), bool)
    cfg = CompositeConfig(capital=1e6, band=0.0, gross=0.9)
    zero = cfg.sim_config(zero_cost=True)
    good = simulate(P, idx, lambda t, cur: v1_weights(sp_future[t], elig, beta, cfg), 0, 399, zero)
    stale = simulate(P, idx, lambda t, cur: v1_weights(sp_future[t - 1], elig, beta, cfg), 1, 399, zero)
    assert good["equity"].iloc[-1] > 3e6                                  # knowing the future pays enormously
    assert abs(stale["equity"].iloc[-1] / 1e6 - 1) < 0.25                 # one bar late: nothing special


def test_topk_weights_long_strongest_short_weakest_beta_neutral():
    from quantbot.backtest.composite_bt import topk_weights

    rng = np.random.default_rng(4)
    sig, beta = rng.normal(size=30), rng.uniform(0.6, 1.8, 30)
    w = topk_weights(sig, np.ones(30, bool), beta, 5, 0.95)
    order = np.argsort(sig)
    assert (w[order[-5:]] > 0).all() and (w[order[:5]] < 0).all() and np.count_nonzero(w[:30]) == 10
    assert abs(np.dot(w[:30], beta) + w[30]) < 1e-12 and abs(np.abs(w).sum() - 0.95) < 1e-12
    assert not topk_weights(sig, np.arange(30) < 8, beta, 5, 0.95).any()      # fewer than 2k eligible: flat


def test_signal_periods_fixed_k():
    from quantbot.analysis.xsection import signal_periods

    rng = np.random.default_rng(5)
    idx = pd.date_range("2026", periods=200, freq="1h", tz="UTC")
    cols = [f"C{i}/USD" for i in range(30)]
    resid = pd.DataFrame(rng.normal(0, 0.01, (200, 30)), index=idx, columns=cols)
    sig = pd.DataFrame(np.tile(np.arange(30.0), (200, 1)), index=idx, columns=cols)   # fixed ranking
    beta = pd.DataFrame(1.0, index=idx, columns=cols)
    elig = pd.DataFrame(True, index=idx, columns=cols)
    p = signal_periods(sig, resid, beta, elig, horizon=24, k=5)
    expected = (resid.iloc[1:25, 25:].sum().mean() - resid.iloc[1:25, :5].sum().mean()) * 1e4
    assert abs(p["spread_bps"].iloc[0] - expected) < 1e-9                     # top 5 minus bottom 5, next 24 bars
    assert abs(p["turnover"].iloc[0] - 2.0) < 1e-12 and np.allclose(p["turnover"].iloc[1:], 0)
