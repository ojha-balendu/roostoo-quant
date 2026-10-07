"""Binance Vision positioning data for research: download, cache, parse.

Covers exactly what the H5 (liquidation-flush rebound) and H6 (cross-sectional
funding) studies need:

* USD-M funding rates    data/futures/um/monthly/fundingRate/{SYM}/{SYM}-fundingRate-YYYY-MM.zip
* USD-M metrics (5 min)  data/futures/um/daily/metrics/{SYM}/{SYM}-metrics-YYYY-MM-DD.zip
  (open interest, long/short account ratios, taker long/short volume ratio)
* Spot klines            data/spot/monthly/klines/{SYM}/{interval}/{SYM}-{interval}-YYYY-MM.zip

Files are discovered through the public S3 listing endpoint behind the
data.binance.vision web UI, so only files that exist are requested.  Raw zips
are cached under ``root/raw/`` mirroring the bucket layout (re-runs only fetch
what is new); one table per symbol is written to ``root/processed/<kind>/``,
always keyed by the *spot* symbol so every research input lines up.

Format quirks handled here so callers never see them:

* Spot kline timestamps switched from milliseconds to microseconds on
  2025-01-01.  Both are normalised to UTC.
* Some files have a header row and some don't.
* Low-priced coins trade as 1000PEPEUSDT etc. on futures.  Funding rates and
  percentage changes in open interest are unaffected by the multiplier, which
  is all the research code uses.
"""
from __future__ import annotations

import io
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DATA_HOST = "https://data.binance.vision/"
LIST_HOST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

FUNDING_DIR = "data/futures/um/monthly/fundingRate/"
METRICS_DIR = "data/futures/um/daily/metrics/"
SPOT_KLINES_DIR = "data/spot/monthly/klines/"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
FUNDING_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
METRICS_COLUMNS = [
    "create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
    "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
    "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
]
# Order matters: plain symbol first, then the multiplier contracts.
MULTIPLIER_PREFIXES = ("", "1000", "1000000", "1M")
KINDS = ("funding", "metrics", "klines")

Fetch = Callable[[str], bytes]
_MS_PER_DAY = 86_400_000


class RemoteError(RuntimeError):
    """The remote refused or kept failing.  Missing files raise FileNotFoundError."""


# --------------------------------------------------------------------------- HTTP


def http_get(url: str, timeout: float = 30.0, retries: int = 4) -> bytes:
    """GET with retries.  404 -> FileNotFoundError; 401/403/451 -> RemoteError at once."""
    last: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "quantbot-research/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:  # must precede URLError (subclass)
            if exc.code == 404:
                raise FileNotFoundError(url) from exc
            if exc.code in (401, 403, 451):
                raise RemoteError(
                    f"HTTP {exc.code} for {url}: access refused (proxy, firewall or blocked region?)"
                ) from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = exc
        time.sleep(min(30.0, 1.5 ** attempt))
    raise RemoteError(f"giving up on {url} after {retries} attempts: {last}")


# ------------------------------------------------------------------ bucket listing


def parse_s3_listing(body: bytes) -> Tuple[List[str], List[str], Optional[str]]:
    """Parse one S3 ListObjects (v1) page -> (keys, common_prefixes, next_marker)."""
    root = ET.fromstring(body)
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    keys = [e.findtext(f"{ns}Key") or "" for e in root.findall(f"{ns}Contents")]
    prefixes = [e.findtext(f"{ns}Prefix") or "" for e in root.findall(f"{ns}CommonPrefixes")]
    truncated = (root.findtext(f"{ns}IsTruncated") or "false").strip().lower() == "true"
    marker: Optional[str] = None
    if truncated:
        marker = root.findtext(f"{ns}NextMarker") or (max(keys + prefixes) if keys or prefixes else None)
    return [k for k in keys if k], [p for p in prefixes if p], marker


def list_bucket(prefix: str, fetch: Fetch = http_get, delimiter: str = "/") -> Tuple[List[str], List[str]]:
    """All keys and common prefixes directly under ``prefix`` (follows pagination)."""
    keys: List[str] = []
    prefixes: List[str] = []
    marker: Optional[str] = None
    seen = set()
    while True:
        params = {"delimiter": delimiter, "prefix": prefix}
        if marker:
            params["marker"] = marker
        k, p, marker = parse_s3_listing(fetch(f"{LIST_HOST}?{urllib.parse.urlencode(params)}"))
        keys.extend(k)
        prefixes.extend(p)
        if marker is None or marker in seen:
            break
        seen.add(marker)
    return keys, prefixes


def list_futures_symbols(fetch: Fetch = http_get) -> set:
    _, prefixes = list_bucket(FUNDING_DIR, fetch)
    return {p.rstrip("/").rsplit("/", 1)[-1] for p in prefixes}


def resolve_futures_symbol(spot_symbol: str, available: Iterable[str]) -> Optional[str]:
    """Map a spot symbol (PEPEUSDT) to its USD-M perp (1000PEPEUSDT), or None."""
    avail = set(available)
    base = spot_symbol[:-4] if spot_symbol.endswith("USDT") else spot_symbol
    for pre in MULTIPLIER_PREFIXES:
        cand = f"{pre}{base}USDT"
        if cand in avail:
            return cand
    return None


_PERIOD_RE = re.compile(r"-(\d{4}-\d{2})(-\d{2})?\.zip$")


def key_period(key: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """[start, end) covered by a monthly or daily Vision file, from its name."""
    m = _PERIOD_RE.search(key)
    if not m:
        return None
    if m.group(2):
        start = pd.Timestamp(m.group(1) + m.group(2), tz="UTC")
        return start, start + pd.Timedelta(days=1)
    start = pd.Timestamp(m.group(1) + "-01", tz="UTC")
    return start, start + pd.offsets.MonthBegin(1)


def filter_keys(keys: Iterable[str], start: pd.Timestamp, end: pd.Timestamp) -> List[str]:
    """Zip files whose period overlaps [start, end]."""
    out = []
    for k in keys:
        if not k.endswith(".zip"):
            continue
        per = key_period(k)
        if per is not None and per[0] <= end and per[1] > start:
            out.append(k)
    return sorted(out)


# ------------------------------------------------------------------------ download


def _download_one(key: str, raw_root: Path, fetch: Fetch) -> str:
    dest = raw_root / key
    if dest.exists() and zipfile.is_zipfile(dest):
        return "cached"
    try:
        body = fetch(DATA_HOST + key)
    except FileNotFoundError:
        return "missing"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.part{os.getpid()}_{threading.get_ident()}")
    tmp.write_bytes(body)
    if not zipfile.is_zipfile(tmp):
        tmp.unlink()
        raise RemoteError(f"corrupt zip for {key}")
    os.replace(tmp, dest)
    return "downloaded"


def download_keys(keys: Sequence[str], raw_root: Path, fetch: Fetch = http_get,
                  workers: int = 16, log_every: int = 1000) -> Dict[str, int]:
    """Download missing keys into ``raw_root`` (mirrors the bucket layout)."""
    raw_root = Path(raw_root)
    counts = {"cached": 0, "downloaded": 0, "missing": 0, "error": 0}
    pending = []
    for k in keys:
        p = raw_root / k
        if p.exists() and zipfile.is_zipfile(p):
            counts["cached"] += 1
        else:
            pending.append(k)
    if not pending:
        return counts
    log.info("downloading %d files (%d cached)", len(pending), counts["cached"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_download_one, k, raw_root, fetch): k for k in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                counts[fut.result()] += 1
            except Exception as exc:  # keep going; summarise at the end
                counts["error"] += 1
                log.warning("failed %s: %s", futs[fut], exc)
                if counts["error"] >= 20 and counts["downloaded"] == 0:
                    for f in futs:
                        f.cancel()
                    raise RemoteError(f"20 failures and no successful download; last error: {exc}")
            if i % log_every == 0:
                log.info("  %d/%d done %s", i, len(pending), counts)
    return counts


# -------------------------------------------------------------------------- parse


def _has_header(first_line: str) -> bool:
    first = first_line.split(",", 1)[0].strip().strip('"').lstrip("\ufeff")
    return not any(ch.isdigit() for ch in first)


def read_vision_csv(source, default_columns: Sequence[str]) -> pd.DataFrame:
    """Read a Vision CSV (zipped or not, with or without header)."""
    raw = bytes(source) if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")] or zf.namelist()
            if not names:
                return pd.DataFrame(columns=list(default_columns))
            raw = zf.read(names[0])
    head = raw[:512].decode("utf-8", "replace").lstrip("\ufeff")
    if not head.strip():
        return pd.DataFrame(columns=list(default_columns))
    header = _has_header(head.splitlines()[0])
    df = pd.read_csv(io.BytesIO(raw), header=0 if header else None)
    if not header:
        cols = list(default_columns)[: df.shape[1]]
        cols += [f"col{i}" for i in range(len(cols), df.shape[1])]
        df.columns = cols
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def to_utc(values) -> pd.DatetimeIndex:
    """Epoch ms/us numbers or datetime strings -> UTC DatetimeIndex."""
    s = pd.Series(values).reset_index(drop=True)
    num = pd.to_numeric(s, errors="coerce")
    if len(s) and num.notna().mean() > 0.5:
        v = num.to_numpy(dtype="float64")
        v = np.where(v >= 1e14, v / 1000.0, v)  # microseconds (spot, 2025+) -> ms
        return pd.DatetimeIndex(pd.to_datetime(v, unit="ms", utc=True))
    return pd.DatetimeIndex(pd.to_datetime(s, utc=True))


def to_epoch_ms(index) -> np.ndarray:
    """UTC DatetimeIndex -> int64 epoch milliseconds (independent of resolution)."""
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return np.asarray((idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1),
                      dtype="int64")


_INTERVAL_RE = re.compile(r"^(\d+)([mhdw])$")


def interval_to_timedelta(interval: str) -> pd.Timedelta:
    m = _INTERVAL_RE.match(interval)
    if not m:
        raise ValueError(f"unsupported interval {interval!r}")
    unit = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}[m.group(2)]
    return pd.Timedelta(**{unit: int(m.group(1))})


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns), index=pd.DatetimeIndex([], tz="UTC", name="time"), dtype=float)


KLINE_FIELDS = ["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume"]


def parse_klines(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Vision kline rows -> numeric frame indexed by bar *close* time (UTC)."""
    if df.empty:
        return _empty(KLINE_FIELDS)
    idx = to_utc(df["open_time"]) + interval_to_timedelta(interval)
    out = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").to_numpy()
                        for c in KLINE_FIELDS if c in df.columns}, index=idx)
    out.index.name = "time"
    return out


def parse_funding(df: pd.DataFrame) -> pd.DataFrame:
    """Funding file -> frame indexed by settlement time with ``rate`` and ``interval_h``."""
    if df.empty:
        return _empty(["rate", "interval_h"])
    tcol = next((c for c in ("calc_time", "fundingTime", "funding_time") if c in df.columns), df.columns[0])
    rcol = next((c for c in ("last_funding_rate", "fundingRate", "funding_rate") if c in df.columns),
                df.columns[-1])
    idx = to_utc(df[tcol]).round(pd.offsets.Second())
    interval = (pd.to_numeric(df["funding_interval_hours"], errors="coerce").to_numpy()
                if "funding_interval_hours" in df.columns else np.full(len(df), 8.0))
    out = pd.DataFrame({"rate": pd.to_numeric(df[rcol], errors="coerce").to_numpy(),
                        "interval_h": interval}, index=idx)
    out.index.name = "time"
    return out


METRIC_FIELDS = {
    "sum_open_interest": "oi",
    "sum_open_interest_value": "oi_value",
    "sum_toptrader_long_short_ratio": "top_ls_ratio",
    "count_long_short_ratio": "ls_ratio",
    "sum_taker_long_short_vol_ratio": "taker_ls_ratio",
}


def parse_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Metrics file -> 5-minute snapshots indexed by create_time (UTC)."""
    if df.empty or "create_time" not in df.columns:
        return _empty(list(METRIC_FIELDS.values()))
    idx = to_utc(df["create_time"])
    out = pd.DataFrame({new: pd.to_numeric(df[old], errors="coerce").to_numpy()
                        for old, new in METRIC_FIELDS.items() if old in df.columns}, index=idx)
    out.index.name = "time"
    return out


def _combine(frames: List[pd.DataFrame], columns: Sequence[str]) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    if not frames:
        return _empty(columns)
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


def load_kline_files(paths: Iterable[Path], interval: str = "1h") -> pd.DataFrame:
    return _combine([parse_klines(read_vision_csv(p, KLINE_COLUMNS), interval) for p in paths], KLINE_FIELDS)


def load_funding_files(paths: Iterable[Path]) -> pd.DataFrame:
    return _combine([parse_funding(read_vision_csv(p, FUNDING_COLUMNS)) for p in paths], ["rate", "interval_h"])


def load_metrics_files(paths: Iterable[Path]) -> pd.DataFrame:
    return _combine([parse_metrics(read_vision_csv(p, METRICS_COLUMNS)) for p in paths],
                    list(METRIC_FIELDS.values()))


_KLINE_FILE_RE = r"^([A-Z0-9]+)-{interval}-\d{{4}}-\d{{2}}(?:-\d{{2}})?\.(?:zip|csv)$"


def scan_kline_files(directory, interval: str = "1h") -> Dict[str, List[Path]]:
    """Find Vision kline files (any folder layout) by their standard file names."""
    pat = re.compile(_KLINE_FILE_RE.format(interval=re.escape(interval)))
    found: Dict[str, List[Path]] = {}
    for p in Path(directory).rglob("*"):
        m = pat.match(p.name)
        if m and p.is_file():
            found.setdefault(m.group(1), []).append(p)
    return {k: sorted(v) for k, v in sorted(found.items())}


# ---------------------------------------------------------------- processed tables


def processed_path(root, kind: str, symbol: str, interval: str = "1h") -> Path:
    sub = f"klines_{interval}" if kind == "klines" else kind
    return Path(root) / "processed" / sub / f"{symbol}.csv.gz"


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.reset_index(drop=True)
    out.insert(0, "ts_ms", to_epoch_ms(df.index))
    tmp = path.with_name(path.name + ".tmp")
    out.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)


def read_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    idx = pd.DatetimeIndex(pd.to_datetime(df.pop("ts_ms").to_numpy(), unit="ms", utc=True), name="time")
    df.index = idx
    return df


# --------------------------------------------------------------------------- sync


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def sync(spot_symbols: Sequence[str], root, start, end, kinds: Sequence[str] = KINDS,
         interval: str = "1h", fetch: Fetch = http_get, workers: int = 16) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Download (incrementally) and rebuild per-symbol processed tables.

    Returns (report, download_counts).  The report is also written to
    ``root/processed/symbol_map.csv``.
    """
    root = Path(root)
    raw_root = root / "raw"
    start, end = _utc(start), _utc(end)
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown kinds {sorted(unknown)}")
    need_futures = bool({"funding", "metrics"} & set(kinds))
    available = list_futures_symbols(fetch) if need_futures else set()
    if need_futures:
        log.info("%d USD-M perps listed on Binance Vision", len(available))

    plan: Dict[str, Tuple[Optional[str], Dict[str, List[str]]]] = {}
    all_keys: List[str] = []
    for spot in spot_symbols:
        fut = resolve_futures_symbol(spot, available) if need_futures else None
        keys: Dict[str, List[str]] = {}
        if "klines" in kinds:
            keys["klines"] = filter_keys(list_bucket(f"{SPOT_KLINES_DIR}{spot}/{interval}/", fetch)[0], start, end)
        if fut and "funding" in kinds:
            keys["funding"] = filter_keys(list_bucket(f"{FUNDING_DIR}{fut}/", fetch)[0], start, end)
        if fut and "metrics" in kinds:
            keys["metrics"] = filter_keys(list_bucket(f"{METRICS_DIR}{fut}/", fetch)[0], start, end)
        plan[spot] = (fut, keys)
        for v in keys.values():
            all_keys.extend(v)
        log.info("%-12s -> %-16s %s", spot, fut or "-", {k: len(v) for k, v in keys.items()})

    counts = download_keys(all_keys, raw_root, fetch=fetch, workers=workers)

    loaders = {"klines": lambda ps: load_kline_files(ps, interval),
               "funding": load_funding_files, "metrics": load_metrics_files}
    rows = []
    for spot, (fut, keys) in plan.items():
        row = {"spot_symbol": spot, "futures_symbol": fut}
        for kind in KINDS:
            paths = [raw_root / k for k in keys.get(kind, []) if (raw_root / k).exists()]
            row[f"{kind}_files"] = len(paths)
            row[f"{kind}_rows"] = 0
            if not paths:
                continue
            table = loaders[kind](paths)
            table = table[(table.index >= start) & (table.index <= end + pd.Timedelta(days=1))]
            write_table(table, processed_path(root, kind, spot, interval))
            row[f"{kind}_rows"] = len(table)
            if len(table):
                row[f"{kind}_first"] = table.index[0].isoformat()
                row[f"{kind}_last"] = table.index[-1].isoformat()
        rows.append(row)
    report = pd.DataFrame(rows)
    out = root / "processed" / "symbol_map.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    return report, counts


# ------------------------------------------------------------------ research input


def load_research_inputs(root, interval: str = "1h", klines_dir=None,
                         symbols: Optional[Sequence[str]] = None
                         ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """(close panel, funding by symbol, metrics by symbol), all keyed by spot symbol.

    Prices come from ``root/processed/klines_<interval>`` unless ``klines_dir``
    points at existing Vision kline files (any folder layout).
    """
    root = Path(root)
    if klines_dir is not None:
        files = scan_kline_files(klines_dir, interval)
        if symbols:
            files = {s: v for s, v in files.items() if s in set(symbols)}
        closes = {s: load_kline_files(v, interval)["close"] for s, v in files.items()}
    else:
        kdir = root / "processed" / f"klines_{interval}"
        paths = sorted(kdir.glob("*.csv.gz")) if kdir.exists() else []
        closes = {p.name[: -len(".csv.gz")]: read_table(p)["close"] for p in paths}
        if symbols:
            closes = {s: v for s, v in closes.items() if s in set(symbols)}
    closes = {s: c for s, c in closes.items() if len(c)}
    close = pd.DataFrame(closes).sort_index() if closes else pd.DataFrame()

    def _load(kind: str) -> Dict[str, pd.DataFrame]:
        out = {}
        for s in close.columns:
            p = processed_path(root, kind, s, interval)
            if p.exists():
                t = read_table(p)
                if len(t):
                    out[s] = t
        return out

    return close, _load("funding"), _load("metrics")
