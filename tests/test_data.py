import io
import zipfile
from datetime import date

import pandas as pd

from quantbot.data.binance_vision import build_url, plan_files, process_klines, read_kline_zip
from quantbot.data.loader import load_panel


def test_urls():
    assert build_url("spot", "monthly", "klines", "BTCUSDT", "1h", "2024-01") == (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip")
    assert build_url("um", "daily", "aggTrades", "ETHUSDT", "", "2024-01-02") == (
        "https://data.binance.vision/data/futures/um/daily/aggTrades/ETHUSDT/ETHUSDT-aggTrades-2024-01-02.zip")


def test_plan_monthly_then_daily():
    plan = plan_files(date(2025, 11, 15), date(2026, 12, 31), today=date(2026, 1, 4))
    assert plan[:2] == [("monthly", "2025-11"), ("monthly", "2025-12")]
    assert plan[2:] == [("daily", "2026-01-01"), ("daily", "2026-01-02"), ("daily", "2026-01-03")]


def _zip(path, rows, header=False):
    buf = io.StringIO()
    if header:
        buf.write("open_time,open,high,low,close,volume,close_time,quote_volume,count,tb,tq,ignore\n")
    for r in rows:
        buf.write(",".join(map(str, r)) + "\n")
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(path.stem + ".csv", buf.getvalue())


def test_read_handles_microseconds_and_headers(tmp_path):
    ms = 1735689600000  # 2025-01-01T00:00Z
    _zip(tmp_path / "a.zip", [[ms * 1000, 1, 2, 0.5, 1.5, 10, ms * 1000 + 3_599_999_999, 15, 3, 5, 7, 0]])
    _zip(tmp_path / "b.zip", [[ms, 1, 2, 0.5, 1.5, 10, ms + 3_599_999, 15, 3, 5, 7, 0]], header=True)
    a, b = read_kline_zip(tmp_path / "a.zip"), read_kline_zip(tmp_path / "b.zip")
    assert a["open_time"].iloc[0] == b["open_time"].iloc[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert a["close"].iloc[0] == 1.5


def test_process_and_load_panel(tmp_path):
    d = tmp_path / "raw/spot/monthly/klines/BTCUSDT/1h"
    d.mkdir(parents=True)
    base = 1704067200000  # 2024-01-01
    _zip(d / "BTCUSDT-1h-2024-01.zip", [[base + i * 3_600_000, 1, 1, 1, 100 + i, 1, base + i * 3_600_000 + 3_599_999,
                                          1, 1, 1, 1, 0] for i in range(5)])
    out = process_klines(tmp_path / "raw", tmp_path / "proc", "spot", "BTCUSDT", "1h")
    assert out and out.exists()
    panel = load_panel(tmp_path / "proc", "spot", "1h", {"BTC/USD": "BTCUSDT", "ETH/USD": "ETHUSDT"})
    assert list(panel.columns) == ["BTC/USD"] and panel["BTC/USD"].iloc[-1] == 104
