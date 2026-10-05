import numpy as np
import pandas as pd

from quantbot.rl.features import N_FEATURES, build_features
from quantbot.rl.scoring import composite_score, episode_metrics
from quantbot.rl.sim import PortfolioSim, SimConfig, run_weights, weights_from_action

NO_COST = SimConfig(fee=0.0, spread_bps=0.0, trade_band=0.0, max_gross=1.0, max_weight=1.0)


def prices_15m(days=40, n=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=days * 96, freq="15min", tz="UTC")
    r = rng.normal(0, 0.002, (len(idx), n))
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx,
                        columns=["BTC/USD", "ETH/USD", "SOL/USD"][:n])


# ----------------------------------------------------------------------------- scoring
def test_flat_scores_zero_and_ratios_are_capped():
    m = episode_metrics(np.full(100, 1e5))
    assert composite_score(m) == 0.0 and m["max_dd"] == 0.0
    up = episode_metrics(1e5 * np.cumprod(np.full(100, 1.001)))      # never a losing bar
    assert up["sortino"] == 10.0 and up["calmar"] == 10.0 and up["sharpe"] in (0.0, 10.0)
    assert abs(up["return_pct"] - (1.001 ** 99 - 1) * 100) < 1e-9


def test_composite_weights():
    m = {"return_pct": 2.0, "sortino": 3.0, "sharpe": 1.0, "calmar": 4.0}
    assert abs(composite_score(m) - (0.5 * 2 + 0.2 * 3 + 0.15 * 1 + 0.15 * 4)) < 1e-12


# ------------------------------------------------------------------------- accounting
def test_long_and_short_accounting_is_exact_without_costs():
    p = prices_15m(days=5)
    P = p.to_numpy()
    w = np.zeros_like(P)
    w[:, 0], w[:, 1] = 0.6, -0.4
    res = run_weights(P, w, 0, len(P) - 1, NO_COST, rebalance=np.zeros(len(P), dtype=bool))
    e0 = NO_COST.initial_capital
    expected = e0 + 0.6 * e0 * (P[-1, 0] / P[0, 0] - 1) - 0.4 * e0 * (P[-1, 1] / P[0, 1] - 1)
    assert abs(res["equity"][-1] - expected) < 1e-6


def test_costs_band_and_caps():
    cfg = SimConfig()
    sim = PortfolioSim(3, cfg)
    cost = sim.rebalance(np.array([0.5, 0.0, 0.0]), np.ones(3, bool))
    assert abs(cost - 0.5 * 100_000 * (0.001 + 0.0001)) < 1e-9
    before = sim.h.copy()
    sim.rebalance(sim.weights + np.array([0.005, 0, 0]), np.ones(3, bool))  # below the 1% band
    assert np.allclose(sim.h, before)
    w = weights_from_action(np.array([1.0, -1.0, 1.0, 1.0, 1.0]), np.array([1, 1, 1, 1, 0], bool), cfg)
    assert np.abs(w).max() <= 0.25 + 1e-12 and abs(np.abs(w).sum() - 0.95) < 1e-12 and w[4] == 0


def test_features_use_only_the_past():
    p = prices_15m()
    f_full, v_full = build_features(p)
    cut = 2000
    f_part, v_part = build_features(p.iloc[:cut])
    assert f_full.shape == (len(p), 3, N_FEATURES) and np.isfinite(f_full).all()
    assert np.array_equal(f_full[:cut], f_part) and np.array_equal(v_full[:cut], v_part)


def test_feature_lookbacks_are_configurable():
    p = prices_15m()
    f, v = build_features(p, (16, 96, 384))
    assert f.shape == (len(p), 3, 4) and np.isfinite(f).all()
