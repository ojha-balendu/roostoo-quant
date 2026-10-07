"""Synthetic Binance Vision mirror for offline tests and smoke runs.

``generate_mirror`` writes files in the exact Vision formats (headerless spot
klines with ms timestamps before 2025 and us after; funding files with a
header; daily metrics files with string timestamps; .CHECKSUM siblings; a
1000-multiplier perp) and can plant the two effects under study so tests can
check that the pipeline recovers a known edge and finds none when there is
none.  ``LocalBucketFetcher`` serves such a directory through the same
listing/download interface as the real bucket.
"""
from __future__ import annotations

import io
import urllib.parse
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from quantbot.data.futures_vision import (DATA_HOST, FUNDING_DIR, LIST_HOST, METRICS_DIR,
                                          SPOT_KLINES_DIR)

_US_FROM = pd.Timestamp("2025-01-01", tz="UTC")


def _write_zip(path: Path, csv_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(path.name[:-4] + ".csv", csv_text)
    path.write_bytes(buf.getvalue())
    (path.parent / (path.name + ".CHECKSUM")).write_text("0" * 64 + "  " + path.name + "\n")


def _ms(ts: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray((ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1), dtype="int64")


def generate_mirror(root, n_coins: int = 10, start: str = "2024-10-01", days: int = 200, seed: int = 0,
                    flush_rate_per_day: float = 0.08, flush_rebound: float = 0.0,
                    funding_effect: float = 0.0) -> Dict[str, object]:
    """Write a fake bucket under ``root``; returns ground truth.

    flush_rebound  : total return over the 12 h after each planted flush
                     (0 = flushes happen but do not rebound).
    funding_effect : hourly drift = -funding_effect * (current funding - mean);
                     0 = funding carries no information.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    hours = days * 24
    t_open = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=hours, freq=pd.Timedelta(hours=1))
    spot = ["BTCUSDT", "ETHUSDT", "PEPEUSDT"] + [f"C{i:02d}USDT" for i in range(max(0, n_coins - 3))]
    spot = spot[:n_coins]
    fut = {s: ("1000" + s if s == "PEPEUSDT" else s) for s in spot}

    mkt = rng.normal(0, 0.005, hours)
    beta = {s: (1.0 if s == "BTCUSDT" else rng.uniform(0.8, 1.6)) for s in spot}
    idio = 0.007

    # funding: AR(1) per 8h settlement; settlement k at t_open[0] + 8h*k
    n_set = hours // 8
    mu, phi, sd = 1e-4, 0.97, 6e-5
    funding = {}
    for s in spot:
        f = np.empty(n_set)
        f[0] = mu + rng.normal(0, sd / np.sqrt(1 - phi ** 2))
        for k in range(1, n_set):
            f[k] = mu + phi * (f[k - 1] - mu) + rng.normal(0, sd)
        funding[s] = f

    flushes: List[Dict[str, object]] = []
    prices, ois = {}, {}
    for s in spot:
        r = beta[s] * mkt + rng.normal(0, idio, hours)
        oi_lr = rng.normal(0, 0.004, hours)
        set_idx = np.minimum(np.arange(hours) // 8, n_set - 1)   # settlement in force during hour h
        r += -funding_effect * (funding[s][set_idx] - mu)
        n_ev = rng.poisson(flush_rate_per_day * days)
        for h in np.sort(rng.choice(np.arange(800, hours - 30), size=n_ev, replace=False)):
            r[h] = -7.0 * np.sqrt(beta[s] ** 2 * 0.005 ** 2 + idio ** 2)
            oi_lr[h] = np.log(0.90)
            for k in range(max(0, set_idx[h] - 2), set_idx[h] + 1):   # settlements known at the bar close
                funding[s][k] = abs(funding[s][k]) + 1e-4
            r[h + 1: h + 13] += flush_rebound / 12.0
            flushes.append({"symbol": s, "bar_close": t_open[h] + pd.Timedelta(hours=1)})
        prices[s] = 100.0 * np.exp(np.cumsum(r)) * (1e-4 if s == "PEPEUSDT" else 1.0)
        ois[s] = 1e6 * np.exp(np.cumsum(oi_lr))

    # ---- spot klines (monthly, headerless, ms -> us switch at 2025-01-01)
    for s in spot:
        close = prices[s]
        open_ = np.concatenate([[close[0]], close[:-1]])
        df = pd.DataFrame({"t": t_open, "o": open_, "c": close})
        for month, g in df.groupby(df["t"].dt.strftime("%Y-%m")):
            ot = _ms(pd.DatetimeIndex(g["t"]))
            ct = ot + 3_599_999
            us = pd.DatetimeIndex(g["t"]) >= _US_FROM
            ot_out = np.where(us, ot * 1000, ot)
            ct_out = np.where(us, ct * 1000 + 999, ct)
            o, c = g["o"].to_numpy(), g["c"].to_numpy()
            lines = []
            for i in range(len(g)):
                hi, lo = max(o[i], c[i]) * 1.001, min(o[i], c[i]) * 0.999
                vol = 1000.0
                lines.append(f"{ot_out[i]},{o[i]:.10g},{hi:.10g},{lo:.10g},{c[i]:.10g},{vol},{ct_out[i]},"
                             f"{vol * c[i]:.6f},100,{vol / 2},{vol * c[i] / 2:.6f},0")
            key = f"{SPOT_KLINES_DIR}{s}/1h/{s}-1h-{month}.zip"
            _write_zip(root / key, "\n".join(lines) + "\n")

    # ---- funding (monthly, header)
    set_times = t_open[0] + pd.to_timedelta(np.arange(n_set) * 8, unit="h")
    for s in spot:
        f = fut[s]
        df = pd.DataFrame({"t": set_times, "rate": funding[s]})
        for month, g in df.groupby(df["t"].dt.strftime("%Y-%m")):
            ms = _ms(pd.DatetimeIndex(g["t"])) + 3          # Binance stamps a few ms late
            body = "calc_time,funding_interval_hours,last_funding_rate\n" + "".join(
                f"{m},8,{r:.8f}\n" for m, r in zip(ms, g["rate"]))
            _write_zip(root / f"{FUNDING_DIR}{f}/{f}-fundingRate-{month}.zip", body)

    # ---- metrics (daily, header, 5-minute snapshots, OI linear within the hour)
    for s in spot:
        f = fut[s]
        mult = 1000.0 if f.startswith("1000") else 1.0
        oi_close = ois[s]                                    # OI at each bar close (t_open + 1h)
        oi_open = np.concatenate([[oi_close[0]], oi_close[:-1]])
        snaps = pd.date_range(t_open[0], periods=hours * 12, freq=pd.Timedelta(minutes=5))
        frac = (np.arange(hours * 12) % 12) / 12.0
        bar = np.arange(hours * 12) // 12
        oi_snap = oi_open[bar] * (oi_close[bar] / oi_open[bar]) ** frac
        px = prices[s][bar] * (1000.0 if mult > 1 else 1.0)
        df = pd.DataFrame({"t": snaps, "oi": oi_snap / mult, "px": px})
        for day, g in df.groupby(df["t"].dt.strftime("%Y-%m-%d")):
            stamps = pd.DatetimeIndex(g["t"]).strftime("%Y-%m-%d %H:%M:%S")
            body = ("create_time,symbol,sum_open_interest,sum_open_interest_value,"
                    "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
                    "count_long_short_ratio,sum_taker_long_short_vol_ratio\n")
            body += "".join(f"{ts},{f},{o:.4f},{o * p:.4f},1.5,1.4,1.6,1.0\n"
                            for ts, o, p in zip(stamps, g["oi"], g["px"]))
            _write_zip(root / f"{METRICS_DIR}{f}/{f}-metrics-{day}.zip", body)

    return {"spot_symbols": spot, "futures_symbols": fut, "flushes": pd.DataFrame(flushes),
            "start": t_open[0], "end": t_open[-1] + pd.Timedelta(hours=1)}


class LocalBucketFetcher:
    """Serve a local mirror through the bucket listing + download URLs."""

    def __init__(self, root, max_keys: int = 1000):
        self.root = Path(root)
        self.max_keys = max_keys
        self.calls = 0
        self._keys: Optional[List[str]] = None

    def _all_keys(self) -> List[str]:
        if self._keys is None:
            self._keys = sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file())
        return self._keys

    def __call__(self, url: str) -> bytes:
        self.calls += 1
        if url.startswith(LIST_HOST):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            prefix = q.get("prefix", [""])[0]
            delim = q.get("delimiter", [""])[0]
            marker = q.get("marker", [""])[0]
            entries: Dict[str, str] = {}
            for k in self._all_keys():
                if not k.startswith(prefix):
                    continue
                rest = k[len(prefix):]
                if delim and delim in rest:
                    entries[prefix + rest.split(delim)[0] + delim] = "prefix"
                else:
                    entries[k] = "key"
            names = sorted(n for n in entries if n > marker)
            page, truncated = names[: self.max_keys], len(names) > self.max_keys
            parts = ['<?xml version="1.0" encoding="UTF-8"?>',
                     '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">',
                     f"<Name>data.binance.vision</Name><Prefix>{escape(prefix)}</Prefix>",
                     f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>"]
            if truncated:
                parts.append(f"<NextMarker>{escape(page[-1])}</NextMarker>")
            for n in page:
                if entries[n] == "key":
                    parts.append(f"<Contents><Key>{escape(n)}</Key><Size>1</Size></Contents>")
                else:
                    parts.append(f"<CommonPrefixes><Prefix>{escape(n)}</Prefix></CommonPrefixes>")
            parts.append("</ListBucketResult>")
            return "".join(parts).encode()
        if url.startswith(DATA_HOST):
            path = self.root / url[len(DATA_HOST):]
            if not path.is_file():
                raise FileNotFoundError(url)
            return path.read_bytes()
        raise FileNotFoundError(url)
