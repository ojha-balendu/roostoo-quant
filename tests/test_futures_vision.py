import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from quantbot.data import futures_vision as fv
from quantbot.data.synthetic_vision import LocalBucketFetcher, generate_mirror


def _zip(text: str, name: str = "x.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, text)
    return buf.getvalue()


def test_parse_s3_listing_namespace_and_pagination():
    xml = (b'<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
           b"<IsTruncated>true</IsTruncated><NextMarker>data/b/</NextMarker>"
           b"<Contents><Key>data/a.zip</Key></Contents>"
           b"<CommonPrefixes><Prefix>data/b/</Prefix></CommonPrefixes></ListBucketResult>")
    keys, prefixes, marker = fv.parse_s3_listing(xml)
    assert keys == ["data/a.zip"] and prefixes == ["data/b/"] and marker == "data/b/"
    xml2 = xml.replace(b"true", b"false")
    assert fv.parse_s3_listing(xml2)[2] is None


def test_to_utc_handles_ms_us_and_strings():
    ms = 1704067200000                      # 2024-01-01
    us = 1735689600000000                   # 2025-01-01 in microseconds
    idx = fv.to_utc([ms, us])
    assert list(idx) == [pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")]
    assert fv.to_utc(["2024-01-01 00:05:00"])[0] == pd.Timestamp("2024-01-01 00:05", tz="UTC")
    assert list(fv.to_epoch_ms(idx)) == [ms, us // 1000]


def test_read_vision_csv_with_and_without_header():
    no_header = _zip("1704067200000,1,2,0.5,1.5,10,1704070799999,15,3,5,7.5,0\n")
    df = fv.read_vision_csv(no_header, fv.KLINE_COLUMNS)
    assert list(df.columns[:5]) == ["open_time", "open", "high", "low", "close"]
    k = fv.parse_klines(df, "1h")
    assert k.index[0] == pd.Timestamp("2024-01-01 01:00", tz="UTC")     # indexed by bar close
    assert k["close"].iloc[0] == 1.5

    with_header = _zip("calc_time,funding_interval_hours,last_funding_rate\n1704067200003,8,0.0001\n")
    f = fv.parse_funding(fv.read_vision_csv(with_header, fv.FUNDING_COLUMNS))
    assert f.index[0] == pd.Timestamp("2024-01-01", tz="UTC") and f["rate"].iloc[0] == pytest.approx(1e-4)
    assert fv.read_vision_csv(_zip(""), fv.FUNDING_COLUMNS).empty


def test_resolve_futures_symbol_handles_multiplier_contracts():
    avail = {"BTCUSDT", "1000PEPEUSDT", "1000000MOGUSDT"}
    assert fv.resolve_futures_symbol("BTCUSDT", avail) == "BTCUSDT"
    assert fv.resolve_futures_symbol("PEPEUSDT", avail) == "1000PEPEUSDT"
    assert fv.resolve_futures_symbol("MOGUSDT", avail) == "1000000MOGUSDT"
    assert fv.resolve_futures_symbol("NVDABUSDT", avail) is None


def test_filter_keys_by_period():
    keys = ["p/X-metrics-2024-01-31.zip", "p/X-metrics-2024-02-01.zip", "p/X-metrics-2024-02-01.zip.CHECKSUM",
            "q/X-fundingRate-2024-01.zip", "q/X-fundingRate-2024-03.zip"]
    got = fv.filter_keys(keys, pd.Timestamp("2024-02-01", tz="UTC"), pd.Timestamp("2024-02-15", tz="UTC"))
    assert got == ["p/X-metrics-2024-02-01.zip"]
    got = fv.filter_keys(keys, pd.Timestamp("2024-01-20", tz="UTC"), pd.Timestamp("2024-02-15", tz="UTC"))
    assert "q/X-fundingRate-2024-01.zip" in got and "q/X-fundingRate-2024-03.zip" not in got


def test_download_marks_missing_and_skips_cached(tmp_path):
    body = _zip("a\n")
    store = {fv.DATA_HOST + "k/a.zip": body}

    def fetch(url):
        if url not in store:
            raise FileNotFoundError(url)
        return store[url]

    c1 = fv.download_keys(["k/a.zip", "k/b.zip"], tmp_path, fetch=fetch, workers=2)
    assert c1["downloaded"] == 1 and c1["missing"] == 1
    c2 = fv.download_keys(["k/a.zip"], tmp_path, fetch=fetch, workers=2)
    assert c2 == {"cached": 1, "downloaded": 0, "missing": 0, "error": 0}


def test_sync_end_to_end_against_local_mirror(tmp_path):
    mirror, root = tmp_path / "mirror", tmp_path / "vision"
    gt = generate_mirror(mirror, n_coins=4, start="2024-12-20", days=20, seed=1, flush_rate_per_day=0.0)
    fetch = LocalBucketFetcher(mirror, max_keys=7)          # small pages exercise pagination
    report, counts = fv.sync(gt["spot_symbols"], root, "2024-12-01", "2025-02-01", fetch=fetch, workers=4)
    assert counts["error"] == 0 and counts["downloaded"] > 0
    assert report.set_index("spot_symbol").loc["PEPEUSDT", "futures_symbol"] == "1000PEPEUSDT"
    assert (root / "processed" / "symbol_map.csv").exists()

    close, funding, metrics = fv.load_research_inputs(root)
    assert set(close.columns) == set(gt["spot_symbols"])
    # continuous hourly index across the 2025-01-01 ms -> us timestamp switch
    steps = close.index.to_series().diff().dropna().unique()
    assert len(steps) == 1 and steps[0] == pd.Timedelta(hours=1)
    assert close.index[0] == pd.Timestamp("2024-12-20 01:00", tz="UTC")
    assert set(funding) == set(metrics) == set(gt["spot_symbols"])
    assert len(metrics["BTCUSDT"]) == 20 * 288
    assert np.isfinite(funding["PEPEUSDT"]["rate"]).all()

    # second run downloads nothing new
    _, counts2 = fv.sync(gt["spot_symbols"], root, "2024-12-01", "2025-02-01", fetch=fetch, workers=4)
    assert counts2["downloaded"] == 0 and counts2["cached"] == counts["downloaded"]


def test_scan_kline_files_finds_any_layout(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "BTCUSDT-1h-2024-01.zip").write_bytes(_zip("1,1,1,1,1,1,1,1,1,1,1,0\n"))
    (tmp_path / "ETHUSDT-1h-2024-01-05.csv").write_text("1,1,1,1,1,1,1,1,1,1,1,0\n")
    (tmp_path / "ETHUSDT-1m-2024-01.zip").write_bytes(b"")
    found = fv.scan_kline_files(tmp_path, "1h")
    assert sorted(found) == ["BTCUSDT", "ETHUSDT"]
