import numpy as np
import pandas as pd
import pytest

from quantbot.analysis import liquidation_flush as lf
from quantbot.data import futures_vision as fv
from quantbot.data.synthetic_vision import LocalBucketFetcher, generate_mirror

H = pd.Timedelta(hours=1)


def _idx(n, start="2024-01-01 01:00"):
    return pd.date_range(pd.Timestamp(start, tz="UTC"), periods=n, freq=H)


def test_align_asof_respects_lag_and_tolerance():
    idx = _idx(4, "2024-01-01 00:00")
    s = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2024-01-01 00:55", "2024-01-01 01:00"], tz="UTC"))
    got = lf.align_asof({"X": s}, idx, lag=pd.Timedelta(minutes=5), tolerance=pd.Timedelta(minutes=90))["X"]
    # value stamped 00:55 is usable at 01:00; value stamped 01:00 only at 01:05
    assert np.isnan(got.iloc[0]) and got.iloc[1] == 1.0 and got.iloc[2] == 2.0
    got = lf.align_asof({"X": s}, idx, lag=pd.Timedelta(minutes=5), tolerance=pd.Timedelta(minutes=30))["X"]
    assert np.isnan(got.iloc[3])                       # too stale


def _toy(n=400, crash_at=300, oi_drop=0.10, funding=1e-4, rebound=0.03, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.005, n)
    r[crash_at] = -0.06
    r[crash_at + 1: crash_at + 13] += rebound / 12
    close = pd.DataFrame({"A": 100 * np.exp(np.cumsum(r))}, index=_idx(n))
    oi = pd.DataFrame({"A": 1e6 * np.ones(n)}, index=close.index)
    oi.iloc[crash_at:, 0] *= 1 - oi_drop
    fund = pd.DataFrame({"A": np.full(n, funding)}, index=close.index)
    return close, oi, fund


P = lf.FlushParams(z_threshold=5.0, vol_lookback=200, min_vol_obs=100, holds=(12,))


@pytest.mark.parametrize("cond,oi_drop,funding,expected", [
    ("price", 0.0, -1e-4, 1),
    ("price_oi", 0.0, 1e-4, 0),     # no OI drop
    ("price_oi", 0.10, -1e-4, 1),
    ("full", 0.10, -1e-4, 0),       # funding negative
    ("full", 0.10, 1e-4, 1),
])
def test_detect_event_conditions(cond, oi_drop, funding, expected):
    close, oi, fund = _toy(oi_drop=oi_drop, funding=funding)
    ev = lf.detect_events(close, oi, fund, P, cond)
    assert len(ev) == expected
    if expected:
        assert ev["pos"].iloc[0] == 300


def test_detection_has_no_lookahead():
    close, oi, fund = _toy()
    ev1 = lf.detect_events(close, oi, fund, P, "full")
    close2 = close.copy()
    close2.iloc[301:, 0] *= 5.0                          # change only the future
    ev2 = lf.detect_events(close2, oi, fund, P, "full")
    assert ev1[["pos", "z"]].equals(ev2[["pos", "z"]])


def test_cooldown_keeps_one_event_per_hold():
    close, oi, fund = _toy()
    close.iloc[305:, 0] *= 0.94                          # second crash 5 bars later
    oi.iloc[305:, 0] *= 0.9
    ev = lf.detect_events(close, oi, fund, lf.FlushParams(vol_lookback=200, min_vol_obs=100, holds=(12,)), "full")
    assert list(ev["pos"]) == [300]
    ev = lf.detect_events(close, oi, fund, lf.FlushParams(vol_lookback=200, min_vol_obs=100, holds=(12,),
                                                          cooldown=1), "full")
    assert list(ev["pos"]) == [300, 305]


def test_event_returns_math():
    close = pd.DataFrame({"A": [100.0, 90.0, 99.0, 108.0]}, index=_idx(4))
    ev = pd.DataFrame({"time": [close.index[1]], "symbol": ["A"], "pos": [1]})
    p = lf.FlushParams(holds=(1, 2, 5), cost_bps=20.0, entry_lag=0)
    r = lf.event_returns(ev, close, p, benchmark=pd.Series([1.0, 1.0, 1.05, 1.05], index=close.index))
    assert list(r["hold"]) == [1, 2]                     # hold 5 runs past the data
    assert r["gross"].iloc[0] == pytest.approx(0.10)
    assert r["net"].iloc[0] == pytest.approx(0.098)
    assert r["excess"].iloc[0] == pytest.approx(0.05)
    r_lag = lf.event_returns(ev, close, lf.FlushParams(holds=(1,), entry_lag=1))
    assert r_lag["gross"].iloc[0] == pytest.approx(108 / 99 - 1)


def test_clustered_stats_uses_days_not_events():
    times = pd.Series(pd.DatetimeIndex(["2024-01-01 01:00", "2024-01-01 02:00", "2024-01-02 01:00"], tz="UTC"))
    n, mean, _ = lf.clustered_stats(pd.Series([0.01, 0.03, -0.01]), times)
    assert n == 2 and mean == pytest.approx((0.02 - 0.01) / 2)


@pytest.fixture(scope="module")
def synthetic_inputs(tmp_path_factory):
    out = {}
    for name, rebound in (("planted", 0.045), ("null", 0.0)):
        base = tmp_path_factory.mktemp(name)
        gt = generate_mirror(base / "m", n_coins=6, days=150, seed=7, flush_rate_per_day=0.08,
                             flush_rebound=rebound)
        fv.sync(gt["spot_symbols"], base / "v", "2024-01-01", "2026-01-01", fetch=LocalBucketFetcher(base / "m"))
        close, funding, metrics = fv.load_research_inputs(base / "v")
        p = lf.FlushParams()
        out[name] = (gt, p) + lf.prepare_inputs(close, metrics, funding, p)
    return out


def test_pipeline_recovers_planted_flushes(synthetic_inputs):
    gt, p, close, oi, fund = synthetic_inputs["planted"]
    ev = lf.detect_events(close, oi, fund, p, "full")
    planted = set(zip(gt["flushes"]["symbol"], gt["flushes"]["bar_close"]))
    found = set(zip(ev["symbol"], ev["time"]))
    assert len(found & planted) >= 0.8 * len(planted)
    assert len(found - planted) <= 0.1 * len(found)
    summary, _ = lf.run_study(close, oi, fund, p, close.index[0], close.index[-1],
                              rules=lf.ProtocolRules(min_days=20))
    row = summary[(summary.condition == "full") & (summary.hold_h == 12)].iloc[0]
    assert row.net_bps > 200 and row.t_days > 3 and row.passes_protocol


def test_pipeline_finds_nothing_without_rebound(synthetic_inputs):
    _, p, close, oi, fund = synthetic_inputs["null"]
    summary, _ = lf.run_study(close, oi, fund, p, close.index[0], close.index[-1],
                              rules=lf.ProtocolRules(min_days=20))
    full = summary[summary.condition == "full"]
    assert not full["passes_protocol"].any()
    assert (full["gross_bps"].abs() < 150).all()
