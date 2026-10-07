import numpy as np
import pandas as pd
import pytest

from quantbot.analysis import funding_xs as fx
from quantbot.data import futures_vision as fv
from quantbot.data.synthetic_vision import LocalBucketFetcher, generate_mirror

D = pd.Timedelta(days=1)


def _funding(start, periods, every_h, rate):
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=periods, freq=pd.Timedelta(hours=every_h))
    return pd.DataFrame({"rate": np.full(periods, rate), "interval_h": every_h}, index=idx)


def test_trailing_funding_is_daily_carry_and_point_in_time():
    f8 = _funding("2024-01-01", 3 * 30, 8, 1e-4)        # 3 settlements/day
    f4 = _funding("2024-01-01", 6 * 30, 4, 0.5e-4)      # 6 settlements/day, same daily carry
    late = _funding("2024-01-10", 3 * 20, 8, 1e-4)      # listed later
    dates = pd.date_range(pd.Timestamp("2024-01-05", tz="UTC"), periods=20, freq=D)
    got = fx.trailing_funding({"A": f8, "B": f4, "C": late}, dates, window_days=7)
    day15 = pd.Timestamp("2024-01-15", tz="UTC")
    assert got.loc[day15, "A"] == pytest.approx(3e-4)
    assert got.loc[day15, "B"] == pytest.approx(3e-4)
    assert np.isnan(got.loc[pd.Timestamp("2024-01-07", tz="UTC"), "A"])       # window not yet full
    assert np.isnan(got.loc[day15, "C"]) and np.isfinite(got.loc[pd.Timestamp("2024-01-17", tz="UTC"), "C"])

    # a settlement stamped exactly at t is not used at t (strictly before)
    spike = f8.copy()
    spike.loc[day15, "rate"] = 1.0
    assert fx.trailing_funding({"A": spike}, pd.DatetimeIndex([day15]), 7).iloc[0, 0] == pytest.approx(3e-4)

    # delisted: no settlement in the last day -> NaN
    stale = f8[f8.index < pd.Timestamp("2024-01-13", tz="UTC")]
    assert np.isnan(fx.trailing_funding({"A": stale}, pd.DatetimeIndex([day15]), 7).iloc[0, 0])


def test_leg_weights_are_beta_neutral_with_unit_gross():
    a, b = fx.leg_weights(1.5, 1.0)
    assert a + b == pytest.approx(1.0)
    assert a * 1.5 - b * 1.0 == pytest.approx(0.0)
    assert fx.leg_weights(1.5, 1.0, beta_neutral=False) == (0.5, 0.5)


def test_rolling_beta_recovers_known_beta():
    rng = np.random.default_rng(0)
    m = pd.Series(rng.normal(0, 0.02, 300))
    rets = pd.DataFrame({"A": 1.7 * m + rng.normal(0, 0.002, 300), "B": 0.5 * m})
    beta = fx.rolling_beta(rets, m, 60)
    assert beta["A"].iloc[-1] == pytest.approx(1.7, abs=0.05)
    assert beta["B"].iloc[-1] == pytest.approx(0.5, abs=1e-9)


def _panel(n_coins=12, days=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp("2024-01-01", tz="UTC"), periods=days, freq=D)
    cols = ["BTCUSDT"] + [f"C{i:02d}USDT" for i in range(n_coins - 1)]
    rets = rng.normal(0, 0.03, (days, n_coins))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)
    fund = pd.DataFrame(rng.normal(1e-4, 2e-4, (days, n_coins)), index=idx, columns=cols)
    return close, fund


def test_backtest_weights_costs_and_short_cap():
    close, fund = _panel()
    p = fx.FundingXSParams(min_coins=10, beta_days=20, quantile=0.25)
    per = fx.run_backtest(close, fund, p)
    active = per[per.k > 0]
    assert len(active) > 5
    assert (active.k == 3).all()
    first = active.iloc[0]
    assert first.turnover == pytest.approx(1.0)          # opening a gross-1 book
    assert np.allclose(active.net, active.gross - active.turnover * 12.5e-4)
    assert (per.next_date > per.date).all()

    # a short whose price goes x11 must lose at most its collateral (1x short)
    close2 = close.copy()
    t0, t1 = active.iloc[0].date, active.iloc[0].next_date
    p_no_resid = fx.FundingXSParams(min_coins=10, beta_days=20, quantile=0.25, residualize=False)
    top = fund.loc[t0].sort_values().index[-1]           # highest funding -> shorted
    close2.loc[t1:, top] *= 11.0
    per2 = fx.run_backtest(close2, fund, p_no_resid).set_index("date")
    assert per2.loc[t0, "k"] == 3
    assert -0.45 < per2.loc[t0, "gross"] < -0.05         # uncapped this would be about -1.7


def test_residualisation_removes_pure_momentum_signal():
    close, _ = _panel(seed=3)
    fund = close / close.shift(7) - 1.0                   # "funding" that is just 7-day momentum
    raw = fx.run_backtest(close, fund, fx.FundingXSParams(min_coins=10, beta_days=20, residualize=False))
    res = fx.run_backtest(close, fund, fx.FundingXSParams(min_coins=10, beta_days=20, residualize=True))
    assert raw["corr_f_mom"].dropna().mean() > 0.99
    assert (res["k"] == 0).all()                          # nothing left to trade after residualising
    assert (res["turnover"] == 0).all()


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    out = {}
    for name, effect in (("planted", 2.5), ("null", 0.0)):
        base = tmp_path_factory.mktemp(name)
        gt = generate_mirror(base / "m", n_coins=12, days=260, seed=3, flush_rate_per_day=0.0,
                             funding_effect=effect)
        fv.sync(gt["spot_symbols"], base / "v", "2024-01-01", "2026-01-01", kinds=("funding", "klines"),
                fetch=LocalBucketFetcher(base / "m"))
        close, funding, _ = fv.load_research_inputs(base / "v")
        daily = fx.daily_closes(close)
        out[name] = (daily, fx.trailing_funding(funding, daily.index, 7))
    return out


def test_pipeline_recovers_planted_funding_effect(synthetic):
    daily, fd = synthetic["planted"]
    per = fx.run_backtest(daily, fd, fx.FundingXSParams(min_coins=8, beta_days=30))
    s = fx.summarize_periods(per, fx.ProtocolRules(min_periods=20))
    assert s["mean_ic"] > 0.1 and s["t_net"] > 3 and s["passes_protocol"]


def test_pipeline_finds_nothing_in_null(synthetic):
    daily, fd = synthetic["null"]
    per = fx.run_backtest(daily, fd, fx.FundingXSParams(min_coins=8, beta_days=30))
    s = fx.summarize_periods(per, fx.ProtocolRules(min_periods=20))
    assert not s["passes_protocol"] and abs(s["mean_ic"]) < 0.1
