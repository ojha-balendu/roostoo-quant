import zipfile
from datetime import date

import numpy as np
import pandas as pd

from quantbot.analysis.positioning import (
    bar_oi_change,
    day_clustered,
    event_returns,
    flush_events,
    residualise,
    trailing_funding,
)
from quantbot.data.binance_vision import BinanceVisionDownloader
from quantbot.data.futures import (
    asof,
    load_futures_panel,
    perp_candidates,
    process_futures,
    read_funding_zip,
    read_metrics_zip,
    save_symbol_map,
)


def write_zip(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(path.stem + ".csv", text)


# ------------------------------------------------------------------------------------- data layer
def test_funding_reader_with_and_without_header(tmp_path):
    ms = 1704067200000  # 2024-01-01
    write_zip(tmp_path / "a.zip", f"calc_time,funding_interval_hours,last_funding_rate\n{ms},8,0.00010000\n"
                                  f"{ms + 8 * 3600_000},8,-0.00005\n")
    write_zip(tmp_path / "b.zip", f"{ms * 1000},8,0.0001\n")                  # no header, microseconds
    a, b = read_funding_zip(tmp_path / "a.zip"), read_funding_zip(tmp_path / "b.zip")
    assert a["time"].iloc[0] == b["time"].iloc[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert list(a["rate"]) == [0.0001, -0.00005]


def test_metrics_reader_text_times_and_named_columns(tmp_path):
    header = ("create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
              "sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n")
    write_zip(tmp_path / "m.zip", header + "2024-01-01 00:05:00,BTCUSDT,80000.5,3.4e9,1.6,1.5,1.7,0.9\n")
    m = read_metrics_zip(tmp_path / "m.zip")
    assert m["time"].iloc[0] == pd.Timestamp("2024-01-01 00:05", tz="UTC")
    assert m["oi"].iloc[0] == 80000.5 and m["oi_value"].iloc[0] == 3.4e9
    assert m["account_ls"].iloc[0] == 1.7 and m["top_trader_ls"].iloc[0] == 1.5 and m["taker_ls"].iloc[0] == 0.9


def test_process_and_load_panel(tmp_path):
    raw, proc = tmp_path / "raw", tmp_path / "proc"
    base = raw / "futures/um/monthly/fundingRate/1000PEPEUSDT"
    write_zip(base / "1000PEPEUSDT-fundingRate-2024-01.zip", "calc_time,funding_interval_hours,last_funding_rate\n"
                                                             "1704067200000,8,0.0003\n1704096000000,8,0.0001\n")
    write_zip(base / "1000PEPEUSDT-fundingRate-2024-02.zip", "1704096000000,8,0.0001\n1706745600000,8,0.0002\n")
    assert process_futures(raw, proc, "fundingRate", "1000PEPEUSDT")
    save_symbol_map(proc, {"PEPE/USD": "1000PEPEUSDT", "TON/USD": None})
    panel = load_futures_panel(proc, "fundingRate", "rate")
    assert list(panel.columns) == ["PEPE/USD"] and len(panel) == 3            # duplicate settlement removed


def test_perp_symbols_and_daily_only_download(monkeypatch, tmp_path):
    assert perp_candidates("PEPEUSDT") == ["PEPEUSDT", "1000PEPEUSDT"]
    assert perp_candidates("1000CHEEMSUSDT") == ["1000CHEEMSUSDT"]
    dl = BinanceVisionDownloader(tmp_path, market="um", data_type="metrics", interval="")
    urls = []
    monkeypatch.setattr(dl, "fetch", lambda url: urls.append(url) or "ok")
    counts = dl.download_symbol("BTCUSDT", date(2026, 9, 28), date(2026, 10, 5), today=date(2026, 10, 3), daily_only=True)
    assert counts["ok"] == 5 and urls[0].endswith("futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2026-09-28.zip")


def test_asof_never_uses_the_future():
    s = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 08:00"], utc=True))
    idx = pd.date_range("2024-01-01 00:00", periods=10, freq="1h", tz="UTC")
    a = asof(s, idx)
    assert a["A"].iloc[7] == 1.0 and a["A"].iloc[8] == 2.0                  # value appears only once settled
    stale = asof(s, idx, tolerance=pd.Timedelta(minutes=10))
    assert np.isnan(stale["A"].iloc[3]) and stale["A"].iloc[8] == 2.0


# -------------------------------------------------------------------------------------- analysis
def test_trailing_funding_is_causal():
    times = pd.date_range("2024-01-01", periods=30, freq="8h", tz="UTC")
    f = pd.DataFrame({"A": np.arange(30, dtype=float)}, index=times)
    idx = pd.date_range("2024-01-01", periods=240, freq="1h", tz="UTC")
    full = trailing_funding(f, idx)
    part = trailing_funding(f.iloc[:15], idx)
    cut = times[14]
    pd.testing.assert_frame_equal(full.loc[:cut], part.loc[:cut])


def test_residualised_signal_is_free_of_the_regressors():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024", periods=50, freq="1h", tz="UTC")
    cols = [f"C{i}" for i in range(30)]
    mom = pd.DataFrame(rng.normal(size=(50, 30)), index=idx, columns=cols)
    own = pd.DataFrame(rng.normal(size=(50, 30)), index=idx, columns=cols)
    funding = 2 * mom + own                                                   # funding = momentum + something else
    e = residualise(funding, [mom], pd.DataFrame(True, index=idx, columns=cols))
    corr_mom = np.mean([np.corrcoef(e.iloc[i], mom.iloc[i])[0, 1] for i in range(50)])
    corr_own = np.mean([np.corrcoef(e.iloc[i], own.iloc[i])[0, 1] for i in range(50)])
    assert abs(corr_mom) < 1e-8 and corr_own > 0.9


def make_crash():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2025-01-01", periods=24 * 60, freq="1h", tz="UTC")
    cols = ["A/USD", "B/USD"]
    r = rng.normal(0, 0.005, (len(idx), 2))
    i_crash = 24 * 50
    r[i_crash, :] = -0.08                                                     # both crash in the same hour
    r[i_crash + 1: i_crash + 13, 0] += 0.004                                  # A rebounds afterwards
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=cols)
    oi_times = pd.date_range(idx[0], idx[-1] + pd.Timedelta(hours=1), freq="5min")
    oi = pd.DataFrame(1000.0, index=oi_times, columns=cols)
    oi.loc[oi.index >= idx[i_crash] + pd.Timedelta(minutes=5), "A/USD"] = 900.0   # A: OI -10% during the crash
    return idx, cols, close, oi, i_crash


def test_flush_detection_uses_oi_drop_and_funding():
    idx, cols, close, oi, i = make_crash()
    oic = bar_oi_change(oi, idx, pd.Timedelta(hours=1))
    assert abs(oic["A/USD"].iloc[i] + 0.10) < 1e-12 and abs(oic["B/USD"].iloc[i]) < 1e-12
    elig = pd.DataFrame(True, index=idx, columns=cols)
    pos = pd.DataFrame(0.0001, index=idx, columns=cols)
    ev = flush_events(close, oic, pos, elig)
    assert ev["flush"]["A/USD"].iloc[i] and not ev["flush"]["B/USD"].iloc[i]   # B fell without an OI flush
    assert ev["no_flush"]["B/USD"].iloc[i] and ev["big_drop"].iloc[i].all()
    neg = flush_events(close, oic, -pos, elig)
    assert not neg["flush"].to_numpy().any()                                  # shorts were crowded: not a long flush


def test_event_returns_and_day_clustering():
    idx, cols, close, oi, i = make_crash()
    events = pd.DataFrame(False, index=idx, columns=cols)
    events.iloc[i, 0] = True
    events.iloc[i + 2, 0] = True                                              # overlaps the first: skipped
    beta = pd.DataFrame(1.0, index=idx, columns=cols)
    ev = event_returns(events, close, close["B/USD"], beta, hold=12, cost=0.0025)
    assert len(ev) == 1
    assert abs(ev["gross"].iloc[0] - (close["A/USD"].iloc[i + 12] / close["A/USD"].iloc[i] - 1)) < 1e-12
    assert abs(ev["net"].iloc[0] - (ev["gross"].iloc[0] - 0.0025)) < 1e-12
    many = pd.DataFrame({"time": pd.to_datetime(["2025-01-01 01:00", "2025-01-01 05:00", "2025-01-02 03:00",
                                                 "2025-01-03 03:00"], utc=True), "net": [0.01, 0.03, -0.01, 0.02]})
    s = day_clustered(many)
    assert s["events"] == 4 and s["days"] == 3 and abs(s["mean"] - 0.0125) < 1e-12
