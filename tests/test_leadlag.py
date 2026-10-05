import numpy as np
import pandas as pd

from quantbot.analysis.leadlag import catch_up, cross_correlations, lead_regression, stale_share, to_returns


def lagged_market(n=60 * 24 * 20, seed=4):
    """FOLLOW reacts to BTC 30% in the same minute and 70% one minute later; INDEP is unrelated."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    btc = rng.normal(0, 0.001, n)
    follow = 0.3 * btc + 0.7 * np.r_[0.0, btc[:-1]] + rng.normal(0, 0.0005, n)
    indep = rng.normal(0, 0.001, n)
    r = {"BTC/USD": btc, "FOLLOW/USD": follow, "INDEP/USD": indep}
    return pd.DataFrame({k: 100 * np.exp(np.cumsum(v)) for k, v in r.items()}, index=idx)


def test_detects_one_minute_lag_and_it_fades_at_lower_frequency():
    closes = lagged_market()
    r1 = to_returns(closes, "1m")
    cc = cross_correlations(r1, "BTC/USD", 3)
    assert cc.loc["FOLLOW/USD", 1] > 0.5 > cc.loc["FOLLOW/USD", 0] > 0.2
    assert abs(cc.loc["FOLLOW/USD", -1]) < 0.05 and abs(cc.loc["INDEP/USD", 1]) < 0.05
    reg = lead_regression(r1, "BTC/USD")
    assert reg.loc["FOLLOW/USD", "lead_t"] > 20 and abs(reg.loc["INDEP/USD", "lead_t"]) < 4
    cc_h = cross_correlations(to_returns(closes, "1h"), "BTC/USD", 1)
    assert cc_h.loc["FOLLOW/USD", 0] > 0.8 and abs(cc_h.loc["FOLLOW/USD", 1]) < 0.25  # by 1h it is same-bar


def test_catch_up_and_stale_share():
    closes = lagged_market()
    cu = catch_up(to_returns(closes, "1m"), "BTC/USD", 0.95, 1)
    assert cu.loc["FOLLOW/USD", "catchup_bps"] > 10 and cu.loc["FOLLOW/USD", "hit_rate"] > 0.9
    assert abs(cu.loc["INDEP/USD", "catchup_bps"]) < 3
    flat = closes.copy()
    flat.iloc[::2, 2] = np.nan
    flat = flat.ffill()
    assert 0.45 < stale_share(flat)["INDEP/USD"] < 0.55


def test_tradability_flags_a_real_catch_up_and_nothing_else():
    from quantbot.analysis.leadlag import TradeRules, event_returns, tradability

    rng = np.random.default_rng(9)
    n = 60 * 24 * 40
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    btc = rng.normal(0, 0.0007, n)
    jumps = rng.random(n) < 0.004
    btc[jumps] += rng.choice([-1, 1], jumps.sum()) * 0.01     # occasional 1% BTC jumps
    lag = np.r_[0.0, btc[:-1]]
    rets = {"BTC/USD": btc, "LAGGY/USD": 0.3 * btc + 0.9 * lag + rng.normal(0, 5e-4, n),
            "INDEP/USD": rng.normal(0, 7e-4, n)}
    r = pd.DataFrame(rets, index=idx)
    ev = event_returns(r, "BTC/USD", "LAGGY/USD", 0.99, 3, 60 * 24 * 7)
    assert (np.diff(np.flatnonzero(r.index.isin(ev.index))) >= 3).all()  # trades never overlap
    stale = pd.Series(0.0, index=r.columns)
    grid = tradability(r, "BTC/USD", freq_label="1m", horizons=(1, 3), quantiles=(0.99,),
                       lookback_bars=60 * 24 * 7, split=idx[n // 2], stale=stale, rules=TradeRules())
    winners = set(grid.loc[grid["tradeable"], "coin"])
    assert winners == {"LAGGY/USD"}
    stale["LAGGY/USD"] = 0.4  # same edge, but an illiquid price -> rejected
    grid2 = tradability(r, "BTC/USD", freq_label="1m", horizons=(1,), quantiles=(0.99,),
                        lookback_bars=60 * 24 * 7, split=idx[n // 2], stale=stale, rules=TradeRules())
    assert not grid2["tradeable"].any()
