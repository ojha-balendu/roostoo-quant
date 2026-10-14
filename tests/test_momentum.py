import numpy as np
import pandas as pd

from quantbot.analysis.xsection import residual_returns
from quantbot.backtest.composite_bt import topk_weights
from quantbot.backtest.momentum import MomentumConfig, _pick, momentum_inputs, sleeve_weights


def market(T=24 * 80, N=16, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=T, freq="1h", tz="UTC")
    btc = rng.normal(0, 0.005, T)
    cols = {"BTC/USD": btc}
    for i in range(N):
        cols[f"C{i:02d}/USD"] = (0.8 + 0.05 * i) * btc + rng.normal(0.00005 * (i - N / 2), 0.006 + 0.0005 * i, T)
    close = pd.DataFrame({k: 100 * np.exp(np.cumsum(v)) for k, v in cols.items()}, index=idx)
    resid, beta = residual_returns(close, "BTC/USD", beta_hours=24 * 20)
    elig = pd.DataFrame(True, index=idx, columns=resid.columns)
    return close, resid, beta, elig


def arrays(close, resid, beta, elig, mc):
    x = {k: v.reindex(columns=resid.columns).to_numpy(dtype=float) for k, v in momentum_inputs(close, resid, elig, mc).items()}
    P = close[list(resid.columns) + ["BTC/USD"]].to_numpy()
    return x, beta.to_numpy(), elig.to_numpy(), np.vstack([P[1:] / P[:-1] - 1, np.zeros((1, P.shape[1]))])


def test_variant_a_equals_the_original_h3_sleeve():
    close, resid, beta, elig = market()
    mc = MomentumConfig()
    x, B, E, R = arrays(close, resid, beta, elig, mc)
    sig = (np.log(close[resid.columns]) - np.log(close[resid.columns]).shift(16 * 24)).to_numpy()
    for t in (24 * 40, 24 * 60, 24 * 79):
        a = sleeve_weights(t, np.zeros(B.shape[1] + 1), x, B, E, R, mc)
        h3 = topk_weights(sig[t], E[t], B[t], 5, 0.95)
        assert np.allclose(a, h3)


def test_inputs_use_only_past_data():
    close, resid, beta, elig = market()
    mc = MomentumConfig(residual_riskadj=True, lookbacks_days=[7, 16, 30], skip_days=1, proximity=True)
    full = momentum_inputs(close, resid, elig, mc)
    cut = close.index[24 * 60]
    part = momentum_inputs(close.loc[:cut], resid.loc[:cut], elig.loc[:cut], mc)
    for k in full:
        pd.testing.assert_frame_equal(full[k].loc[:cut], part[k], check_freq=False, obj=k)


def test_consistency_counts_days_moving_with_the_trend():
    idx = pd.date_range("2025-01-01", periods=24 * 40, freq="1h", tz="UTC")
    steady = np.full(len(idx), 0.001)                                   # up every hour
    jumpy = np.zeros(len(idx))
    jumpy[-24 * 2] = 0.3                                                # one big jump, flat otherwise
    resid = pd.DataFrame({"S/USD": steady, "J/USD": jumpy}, index=idx)
    close = pd.DataFrame(100 * np.exp(resid.cumsum()), index=idx)
    elig = pd.DataFrame(True, index=idx, columns=resid.columns)
    x = momentum_inputs(close, resid, elig, MomentumConfig(residual_riskadj=False))
    assert x["consistency"]["S/USD"].iloc[-1] == 1.0
    assert x["consistency"]["J/USD"].iloc[-1] < 0.2                     # 1 of 16 days carried the move


def test_buffer_keeps_held_coins_until_they_drop_out():
    score = np.arange(20, 0, -1).astype(float)                          # coin 0 best ... coin 19 worst
    pool = np.ones(20, bool)
    held = np.zeros(20, bool)
    held[[7, 12]] = True                                                # ranked 8th and 13th
    picked = _pick(pool, score, held, 5, 10, True)
    assert 7 in picked and 12 not in picked and len(picked) == 5        # inside top 10 kept, outside dropped
    assert set(picked) == {7, 0, 1, 2, 3}


def test_inverse_vol_cap_and_vol_target():
    close, resid, beta, elig = market()
    base = MomentumConfig(residual_riskadj=True, weighting="inverse_vol")
    x, B, E, R = arrays(close, resid, beta, elig, base)
    t = 24 * 70
    w = sleeve_weights(t, np.zeros(B.shape[1] + 1), x, B, E, R, base)
    longs = np.flatnonzero(w[:-1] > 0)
    risk = x["risk"][t][longs]
    assert np.argmax(w[longs]) == np.argmin(risk)                       # calmest coin gets the most
    assert abs(np.dot(w[:-1], B[t]) + w[-1]) < 1e-12                    # beta-neutral
    capped = sleeve_weights(t, np.zeros(B.shape[1] + 1), x, B, E, R, MomentumConfig(residual_riskadj=True,
                                                                                    weighting="inverse_vol", max_weight=0.05))
    assert np.abs(capped[:-1]).max() <= 0.05 + 1e-12
    tgt = sleeve_weights(t, np.zeros(B.shape[1] + 1), x, B, E, R, MomentumConfig(residual_riskadj=True, vol_target=0.01))
    sigma = np.std(R[t - 720:t] @ tgt) * np.sqrt(24 * 365)
    assert abs(sigma - 0.01) < 1e-6 and np.abs(tgt).sum() < 0.95


def test_competition_score_formula_and_summaries():
    from quantbot.backtest.metrics import competition_score, compute_metrics
    from quantbot.backtest.windows import summarize

    assert abs(competition_score({"sortino": 2.0, "sharpe": 1.0, "calmar": 3.0}) - (0.8 + 0.3 + 0.9)) < 1e-12
    assert np.isnan(competition_score({"sortino": 2.0, "sharpe": 1.0, "calmar": float("nan")}))
    idx = pd.date_range("2025-01-01", periods=24 * 20, freq="1h", tz="UTC")
    eq = pd.Series(1e6 * np.exp(np.cumsum(np.random.default_rng(1).normal(2e-4, 3e-3, len(idx)))), index=idx)
    m = compute_metrics(eq)
    assert abs(m["score"] - (0.4 * m["sortino"] + 0.3 * m["sharpe"] + 0.3 * m["calmar"])) < 1e-12
    rows = pd.DataFrame({"return": [0.01, 0.02, -0.01], "max_dd": [-0.02] * 3, "sharpe": [1, 2, 3], "sortino": [1, 2, 3],
                         "calmar": [1, 2, 3], "score": [1.0, 2.0, 4.0], "fees": [0] * 3, "avg_gross": [1] * 3,
                         "beta_btc": [np.nan] * 3})
    s = summarize(rows)
    assert s["median_score"] == 2.0 and abs(s["mean_score"] - 7 / 3) < 1e-12


def test_leg_weights_three_schemes():
    from quantbot.backtest.momentum import leg_weights

    sig = np.array([0.01, 0.02, 0.04, np.nan])
    assert np.allclose(leg_weights(sig, "equal"), 1.0)
    inv = leg_weights(sig, "inverse_vol")
    filled = np.array([0.01, 0.02, 0.04, np.mean([0.01, 0.02, 0.04])])     # missing sigma -> leg average
    assert np.allclose(inv, 1 / filled)
    assert np.allclose(leg_weights(sig, "mul_vol"), filled)                # proportional to volatility
    ex = leg_weights(sig, "exp_vol")
    assert np.allclose(ex, np.exp(-filled / filled.sum()))                 # the requested formula
    rng = np.random.default_rng(0)
    for _ in range(200):                                                    # never more than e apart
        w = leg_weights(rng.lognormal(-4, 1.0, rng.integers(1, 9)), "exp_vol")
        assert w.max() / w.min() <= np.e + 1e-12
    assert leg_weights(np.array([0.03]), "exp_vol").shape == (1,)


def test_uneven_long_and_short_counts_keep_equal_dollar_legs():
    close, resid, beta, elig = market(N=20)
    for scheme in ("equal", "inverse_vol", "exp_vol"):
        mc = MomentumConfig(residual_riskadj=True, k=2, k_short=7, buffer_k=4, buffer_short=14, weighting=scheme)
        x, B, E, R = arrays(close, resid, beta, elig, mc)
        t = 24 * 70
        w = sleeve_weights(t, np.zeros(B.shape[1] + 1), x, B, E, R, mc)
        coins = w[:-1]
        assert (coins > 0).sum() == 2 and (coins < 0).sum() == 7
        assert abs(coins[coins > 0].sum() + coins[coins < 0].sum()) < 1e-12   # equal dollars long and short
        assert abs(np.dot(coins, B[t]) + w[-1]) < 1e-12 and abs(np.abs(w).sum() - 0.95) < 1e-12
        if scheme != "equal":
            shorts = np.flatnonzero(coins < 0)
            calm = shorts[np.argmin(x["risk"][t][shorts])]
            assert coins[calm] == coins[shorts].min()                     # calmest short gets the most


def test_short_buffer_is_separate():
    score = np.arange(20, 0, -1).astype(float)                            # coin 19 is the worst
    held = np.zeros(20, bool)
    held[11] = True                                                       # 9th worst
    from quantbot.backtest.momentum import _pick

    assert 11 in _pick(np.ones(20, bool), score, held, 3, 10, False)      # inside the worst 10: kept
    assert 11 not in _pick(np.ones(20, bool), score, held, 3, 6, False)   # outside the worst 6: dropped


def test_one_sided_books_and_empty_book():
    close, resid, beta, elig = market(N=20)
    t = 24 * 70
    for k, k_short in ((0, 4), (3, 0)):
        mc = MomentumConfig(residual_riskadj=True, k=k, k_short=k_short, weighting="mul_vol")
        x, B, E, R = arrays(close, resid, beta, elig, mc)
        w = sleeve_weights(t, np.zeros(B.shape[1] + 1), x, B, E, R, mc)
        coins = w[:-1]
        assert (coins > 0).sum() == k and (coins < 0).sum() == k_short
        assert abs(np.dot(coins, B[t]) + w[-1]) < 1e-12                     # hedge still cancels the beta
        assert np.sign(w[-1]) == (1 if k == 0 else -1)                        # shorts only -> BTC long
        assert abs(np.abs(w).sum() - 0.95) < 1e-12
        vols = x["risk"][t][np.flatnonzero(coins)]
        assert np.argmax(np.abs(coins[np.flatnonzero(coins)])) == np.argmax(vols)   # most volatile gets the most
    mc0 = MomentumConfig(residual_riskadj=True, k=0, k_short=0)
    x, B, E, R = arrays(close, resid, beta, elig, mc0)
    assert not sleeve_weights(t, np.zeros(B.shape[1] + 1), x, B, E, R, mc0).any()
