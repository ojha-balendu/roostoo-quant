"""Bulk historical data from Binance Vision (https://data.binance.vision), no API key needed.

URL layout:
  {BASE}/{spot|futures/um|futures/cm}/{monthly|daily}/{klines|aggTrades|trades}/{SYMBOL}/[{interval}/]
        {SYMBOL}-{interval|aggTrades|trades}-{YYYY-MM | YYYY-MM-DD}.zip
Every zip has a sibling .CHECKSUM file with its sha256.

Plan: monthly archives for complete past months, daily archives for the current month.
If last month's monthly archive is not published yet, fall back to its daily archives.

Gotchas handled in `read_kline_zip`:
  * spot klines switched from millisecond to microsecond timestamps from 2025-01-01;
  * some archives (mostly futures) include a header row, older spot ones do not.
"""

from __future__ import annotations

import calendar
import hashlib
import logging
import threading
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data"
MARKET_PATHS = {"spot": "spot", "um": "futures/um", "cm": "futures/cm"}
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def _months(start: date, end: date) -> Iterator[tuple[int, int]]:
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _days(first: date, last: date) -> Iterator[date]:
    d = first
    while d <= last:
        yield d
        d += timedelta(days=1)


def plan_files(start: date, end: date, today: date) -> list[tuple[str, str]]:
    """(period, date_str) archives covering [start, end], given that today's data isn't published."""
    end = min(end, today - timedelta(days=1))
    if end < start:
        return []
    this_month = today.replace(day=1)
    out: list[tuple[str, str]] = []
    for y, m in _months(start, end):
        first = date(y, m, 1)
        if first < this_month:
            out.append(("monthly", f"{y:04d}-{m:02d}"))
        else:
            out.extend(("daily", d.isoformat()) for d in _days(max(start, first), end))
    return out


def build_url(market: str, period: str, data_type: str, symbol: str, interval: str, date_str: str) -> str:
    root = f"{BASE_URL}/{MARKET_PATHS[market]}/{period}/{data_type}/{symbol}"
    if data_type == "klines":
        return f"{root}/{interval}/{symbol}-{interval}-{date_str}.zip"
    return f"{root}/{symbol}-{data_type}-{date_str}.zip"


class BinanceVisionDownloader:
    def __init__(
        self,
        raw_dir: str | Path,
        market: str = "spot",
        data_type: str = "klines",
        interval: str = "1h",
        max_workers: int = 8,
        verify_checksum: bool = True,
    ):
        if market not in MARKET_PATHS:
            raise ValueError(f"market must be one of {sorted(MARKET_PATHS)}")
        self.raw_dir = Path(raw_dir)
        self.market = market
        self.data_type = data_type
        self.interval = interval
        self.max_workers = max_workers
        self.verify_checksum = verify_checksum
        self._local = threading.local()

    @property
    def session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        return self._local.session

    def url_for(self, symbol: str, period: str, date_str: str) -> str:
        return build_url(self.market, period, self.data_type, symbol, self.interval, date_str)

    def local_path(self, url: str) -> Path:
        return self.raw_dir / url.split("/data/", 1)[1]

    def _expected_checksum(self, url: str) -> str | None:
        try:
            r = self.session.get(url + ".CHECKSUM", timeout=30)
            return r.text.split()[0].strip() if r.status_code == 200 and r.text.strip() else None
        except requests.RequestException:
            return None

    def fetch(self, url: str) -> str:
        """Download one archive. Returns ok | exists | missing | failed."""
        path = self.local_path(url)
        if path.exists():
            return "exists"
        tmp = path.with_suffix(".part")
        try:
            with self.session.get(url, stream=True, timeout=120) as r:
                if r.status_code == 404:
                    return "missing"
                r.raise_for_status()
                path.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                        digest.update(chunk)
            if self.verify_checksum:
                expected = self._expected_checksum(url)
                if expected and expected != digest.hexdigest():
                    tmp.unlink(missing_ok=True)
                    log.error("checksum mismatch for %s", url)
                    return "failed"
            tmp.replace(path)
            return "ok"
        except requests.RequestException as exc:
            tmp.unlink(missing_ok=True)
            log.warning("download failed %s: %s", url, exc)
            return "failed"

    def download_symbol(self, symbol: str, start: date, end: date, today: date | None = None,
                        daily_only: bool = False) -> Counter:
        today = today or datetime.now(timezone.utc).date()
        if daily_only:  # some datasets (e.g. futures metrics) are published as daily files only
            counts: Counter = Counter()
            for d in _days(start, min(end, today - timedelta(days=1))):
                counts[self.fetch(self.url_for(symbol, "daily", d.isoformat()))] += 1
            return counts
        last_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
        counts: Counter = Counter()
        for period, ds in plan_files(start, end, today):
            status = self.fetch(self.url_for(symbol, period, ds))
            if status == "missing" and period == "monthly" and ds == last_month.strftime("%Y-%m"):
                # Last month's archive may not be published yet -> use daily files.
                y, m = last_month.year, last_month.month
                month_end = date(y, m, calendar.monthrange(y, m)[1])
                for d in _days(max(start, last_month), min(end, month_end)):
                    counts[self.fetch(self.url_for(symbol, "daily", d.isoformat()))] += 1
                continue
            counts[status] += 1
        return counts

    def download(self, symbols: Iterable[str], start: date, end: date, daily_only: bool = False) -> dict[str, Counter]:
        symbols = list(symbols)
        results: dict[str, Counter] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self.download_symbol, s, start, end, None, daily_only): s for s in symbols}
            for n, fut in enumerate(as_completed(futures), 1):
                sym = futures[fut]
                results[sym] = fut.result()
                log.info("[%d/%d] %s %s", n, len(symbols), sym, dict(results[sym]))
        return results


def _to_ms(values: pd.Series) -> np.ndarray:
    v = values.to_numpy(dtype="int64")
    return np.where(v > 10**14, v // 1000, v)  # microseconds -> milliseconds


def read_kline_zip(path: str | Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        with zf.open(zf.namelist()[0]) as fh:
            df = pd.read_csv(fh, header=None, dtype=str)
    if df.empty:
        return pd.DataFrame(columns=KLINE_COLUMNS[:-1])
    if not str(df.iloc[0, 0]).strip().isdigit():  # header row
        df = df.iloc[1:]
    df = df.iloc[:, : len(KLINE_COLUMNS)]
    df.columns = KLINE_COLUMNS[: df.shape[1]]
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["open_time"])
    df["open_time"] = pd.to_datetime(_to_ms(df["open_time"]), unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(_to_ms(df["close_time"]), unit="ms", utc=True)
    return df.drop(columns=["ignore"], errors="ignore")


def process_klines(
    raw_dir: str | Path,
    processed_dir: str | Path,
    market: str,
    symbol: str,
    interval: str,
) -> Path | None:
    """Merge all downloaded archives for one symbol into a single parquet file."""
    files = sorted((Path(raw_dir) / MARKET_PATHS[market]).glob(f"*/klines/{symbol}/{interval}/*.zip"))
    if not files:
        return None
    frames = []
    for f in files:
        try:
            frames.append(read_kline_zip(f))
        except (zipfile.BadZipFile, ValueError) as exc:
            log.error("cannot read %s (%s); delete it and re-download", f, exc)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    out = Path(processed_dir) / market / interval / f"{symbol}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.set_index("open_time").to_parquet(out)
    return out
