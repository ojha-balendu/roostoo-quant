import numpy as np
import pandas as pd

from quantbot.analysis.latency import lag_estimate, opportunities, panels, update_stats


def recording(lag=3, n=400, pairs=("AAA/USD", "BBB/USD"), seed=0, jump_every=50):
    rng = np.random.default_rng(seed)
    rows = []
    paths = {}
    for k, p in enumerate(pairs):
        r = rng.normal(0, 0.0003, n)
        r[::jump_every] += 0.01 * (1 if k == 0 else -1)                     # occasional 1% moves
        paths[p] = 100 * np.exp(np.cumsum(r))
    for t in range(n):
        for p, px in paths.items():
            rows.append({"poll": t, "ts": float(t), "source": "binance", "pair": p, "bid": px[t] * 0.9999,
                         "ask": px[t] * 1.0001, "last": px[t]})
            stale = px[max(t - lag, 0)]
            rows.append({"poll": t, "ts": float(t), "source": "roostoo", "pair": p, "bid": stale * 0.9999,
                         "ask": stale * 1.0001, "last": stale})
    return pd.DataFrame(rows)


def test_lag_is_recovered():
    p = panels(recording(lag=3))
    lag = lag_estimate(p["r_last"], p["b_mid"], max_lag=10)
    assert (lag["lag_polls"] == 3).all() and (lag["error_at_lag_bps"] < 0.01).all()
    assert p["seconds_per_poll"] == 1.0
    assert (update_stats(p["r_last"], 1.0)["share_of_polls_changed"] > 0.95).all()


def test_lag_creates_profitable_trades_and_no_lag_creates_none():
    p = panels(recording(lag=3))
    ops = opportunities(p, fee=0.001, margin_bps=5, horizon=5)
    assert len(ops) >= 10 and ops["net_bps"].mean() > 30 and set(ops["side"]) == {1, -1}
    p0 = panels(recording(lag=0))
    assert opportunities(p0, fee=0.001, margin_bps=5, horizon=5).empty


def test_probe_records_and_analyses_end_to_end(tmp_path, capsys):
    import importlib.util
    from pathlib import Path

    from quantbot.exchange.models import Ticker

    spec = importlib.util.spec_from_file_location(
        "probe", Path(__file__).resolve().parents[1] / "research" / "29_latency_probe.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.0003, 200)
    r[::40] += 0.01
    px = 100 * np.exp(np.cumsum(r))
    state = {"t": 0}

    def binance():
        state["t"] += 1
        p_ = px[state["t"]]
        return {"AAAUSDT": (p_ * 0.9999, p_ * 1.0001), "OTHERUSDT": (1.0, 1.0)}

    def roostoo():
        p_ = px[max(state["t"] + 1 - 3, 0)]                                  # 3 polls behind Binance
        return {"AAA/USD": Ticker("AAA/USD", p_ * 0.9999, p_ * 1.0001, p_), "ZZZ/USD": Ticker("ZZZ/USD", 1, 1, 1)}

    path = tmp_path / "probe.csv"
    done = probe.record(roostoo, binance, {"AAA/USD": "AAAUSDT"}, n_polls=150, interval=0.0, path=path, progress_every=0)
    assert done == 150
    rec = pd.read_csv(path)
    assert set(rec["pair"]) == {"AAA/USD"} and set(rec["source"]) == {"roostoo", "binance"}
    probe.analyse(path, fee=0.001, margin_bps=5, max_lag_s=10)                # interval 0: counted in polls
    out = capsys.readouterr().out
    assert "TRADES THE LAG WOULD ALLOW" in out and "1 of 1 pairs lag Binance" in out
    p = panels(rec)
    assert int(lag_estimate(p["r_last"], p["b_mid"], max_lag=10)["lag_polls"].iloc[0]) == 3


def test_kelly_stats():
    from quantbot.analysis.latency import kelly_stats

    rng = np.random.default_rng(0)
    net = rng.normal(0.002, 0.004, 400)                                    # +20 bps mean, 40 bps spread
    k = kelly_stats(net, trades_per_day=100, cap=10.0, kelly_scale=1.0)
    mu, s = net.mean(), net.std(ddof=1)
    assert abs(k["kelly"] - mu / s ** 2) < 1e-9
    assert abs(k["kelly_cons"] - (mu - s / 20) / s ** 2) < 1e-9          # mu minus one standard error
    f = k["f"]
    assert abs(k["growth_day"] - 100 * (f * mu - f ** 2 * s ** 2 / 2)) < 1e-12
    capped = kelly_stats(net, 100, cap=0.25)
    assert capped["f"] == 0.25 and abs(capped["profit_day"] - 0.25 * mu * 100) < 1e-12
    losing = kelly_stats(rng.normal(-0.001, 0.004, 400), 100, cap=0.25)
    assert losing["f"] == 0 and losing["growth_day"] == 0                   # no edge: bet nothing
    assert kelly_stats(np.array([0.005]), 10, 0.25)["f"] == 0               # one trade: no variance, no bet


def test_margin_sweep_trades_fall_as_margin_rises():
    from quantbot.analysis.latency import margin_sweep

    p = panels(recording(lag=3))
    s = margin_sweep(p, fee=0.001, margins=[5, 20, 60], horizon=5, hours=400 / 3600, cap=0.25)
    assert list(s["trades"]) == sorted(s["trades"], reverse=True) and (s["entry_gap_bps"] == [25, 40, 80]).all()


def test_entry_delay_removes_profit_when_the_lag_is_shorter_than_the_delay():
    long_lag = panels(recording(lag=3))
    short_lag = panels(recording(lag=1))
    assert opportunities(long_lag, 0.001, 5, 5, delay=1)["net_bps"].mean() > 30       # 3 s lag survives 1 s delay
    assert opportunities(short_lag, 0.001, 5, 5, delay=0)["net_bps"].mean() > 30      # optimistic: looks great
    late = opportunities(short_lag, 0.001, 5, 5, delay=1)
    assert late.empty or late["net_bps"].mean() < 0                                  # but it was all look-ahead


def test_lag_profile_recovers_a_planted_millisecond_lag():
    from quantbot.analysis.latency import lag_profile, roostoo_changes

    # Binance mid steps up from 100 to 101 at t=10.00; Roostoo reflects it 0.4 s later at t=10.40.
    b = []
    for i in range(2000):                                                 # 20 s at 100 Hz
        t = i / 100
        mid = 100.0 if t < 10.0 else 101.0
        b.append({"ts": t, "pair": "AAA/USD", "mid": mid})
    binance = pd.DataFrame(b)
    reads = []
    for i in range(400):                                                  # 20 s at 20 Hz
        t = i / 20
        mid = 100.0 if t < 10.40 else 101.0
        reads.append({"ts": t, "pair": "AAA/USD", "mid": mid})
    reads = pd.DataFrame(reads)

    chg = roostoo_changes(reads)
    assert len(chg) == 1 and chg.iloc[0]["old_mid"] == 100.0 and chg.iloc[0]["new_mid"] == 101.0
    prof = lag_profile(reads, binance, window_s=5.0)
    assert len(prof) == 1
    assert abs(prof.iloc[0]["lag_s"] - 0.40) < 0.06                      # within one Roostoo read interval
    assert prof.attrs["n_changes"] == 1 and prof.attrs["n_unmatched"] == 0


def test_lag_profile_ignores_roostoo_moves_binance_did_not_make():
    from quantbot.analysis.latency import lag_profile

    binance = pd.DataFrame([{"ts": t / 100, "pair": "AAA/USD", "mid": 100.0} for t in range(1000)])
    reads = pd.DataFrame([{"ts": t / 20, "pair": "AAA/USD", "mid": 100.0 if t < 100 else 105.0}
                          for t in range(200)])                           # Roostoo jumps, Binance flat
    prof = lag_profile(reads, binance, window_s=5.0)
    assert prof.empty and prof.attrs["n_unmatched"] == 1


def test_ms_probe_analyse_end_to_end(tmp_path, capsys):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "ms_probe", Path(__file__).resolve().parents[1] / "research" / "30_ms_latency.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    d = tmp_path / "ms_test"
    d.mkdir()
    # Binance steps 100 -> 101 at t=10.0; Roostoo follows ~0.3 s later.
    pd.DataFrame([{"ts": i / 100, "pair": "AAA/USD", "mid": 100.0 if i / 100 < 10 else 101.0}
                  for i in range(2000)]).to_csv(d / "binance.csv", index=False)
    pd.DataFrame([{"ts": i / 20, "pair": "AAA/USD", "mid": 100.0 if i / 20 < 10.3 else 101.0}
                  for i in range(400)]).to_csv(d / "roostoo.csv", index=False)
    mod.analyse(d, window_s=5.0, fast_path_ms=200.0)
    out = capsys.readouterr().out
    assert "REACTION LAG" in out and "matched a prior Binance move" in out and "catchable" in out
