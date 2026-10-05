import numpy as np
import pandas as pd

from quantbot.analysis.xsection import eligibility, evaluate, residual_returns, spread_periods


def market(n_coins=25, hours=24 * 200, reversion=0.0, seed=0):
    """Hourly prices: beta * BTC + idiosyncratic noise; with reversion > 0 each coin's
    idiosyncratic price also mean-reverts (over-reaction that partially undoes itself)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=hours, freq="1h", tz="UTC")
    btc = rng.normal(0, 0.006, hours)
    cols = {"BTC/USD": btc}
    for i in range(n_coins):
        shocks = rng.normal(0, 0.01, hours)
        x = np.zeros(hours)
        for t in range(1, hours):
            x[t] = (1 - reversion) * x[t - 1] + shocks[t]
        idio = np.diff(np.r_[0.0, x])
        cols[f"C{i}/USD"] = (1.0 + 0.04 * i) * btc + idio
    return pd.DataFrame({k: 100 * np.exp(np.cumsum(v)) for k, v in cols.items()}, index=idx)


def run(closes, F=24, H=24):
    resid, beta = residual_returns(closes, beta_hours=24 * 20)
    elig = eligibility(closes, None, min_history_hours=24 * 20)
    return spread_periods(resid, beta, elig, formation=F, horizon=H, cost_one_way_bps=12.5,
                          start=closes.index[24 * 30])


def test_detects_planted_reversal():
    closes = market(reversion=0.04)  # half-life ~17 hours
    p = run(closes)
    stats = evaluate(p, -1, p["time"].iloc[len(p) // 2])
    assert p["spread_bps"].mean() < 0            # momentum spread negative => reversal
    assert stats["net_bps"] > 20 and stats["t"] > 3


def test_random_market_shows_no_edge_after_costs():
    p = run(market(reversion=0.0, seed=1))
    for d in (1, -1):
        st = evaluate(p, d, p["time"].iloc[len(p) // 2])
        assert st["net_bps"] < 5 or st["t"] < 3


def test_residuals_and_signal_use_only_past_data():
    closes = market(hours=24 * 60)
    cut = closes.index[-200]
    full, _ = residual_returns(closes, beta_hours=24 * 20)
    part, _ = residual_returns(closes.loc[:cut], beta_hours=24 * 20)
    pd.testing.assert_frame_equal(full.loc[:cut], part, check_freq=False)


def test_turnover_counts_only_changes():
    closes = market(hours=24 * 80)
    p = run(closes, F=24, H=24)
    assert p["turnover"].iloc[0] >= 2.0                    # opening both legs (+ BTC hedge)
    assert (p["turnover"] <= 4.0 + 2.0).all()               # at most both legs fully flipped + hedge
    assert np.allclose(p["cost_bps"].to_numpy(), p["turnover"].to_numpy() * 12.5)


def test_unchanged_ranking_costs_nothing_after_entry():
    from quantbot.analysis.xsection import spread_periods

    idx = pd.date_range("2025-01-01", periods=24 * 10, freq="1h", tz="UTC")
    cols = [f"C{i}/USD" for i in range(10)]
    noise = np.random.default_rng(0).normal(0, 1e-6, (len(idx), 1))  # same for every coin: ranking fixed
    resid = pd.DataFrame(np.tile(np.linspace(-1e-3, 1e-3, 10), (len(idx), 1)) + noise, index=idx, columns=cols)
    beta = pd.DataFrame(1.0, index=idx, columns=cols)
    elig = pd.DataFrame(True, index=idx, columns=cols)
    p = spread_periods(resid, beta, elig, formation=4, horizon=4, vol_hours=24, min_coins=5)
    assert p["turnover"].iloc[0] > 0 and np.allclose(p["turnover"].iloc[1:], 0.0)
