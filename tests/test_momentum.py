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
