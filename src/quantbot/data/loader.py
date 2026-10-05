"""Load processed klines into the (time x pair) panels that strategies consume."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def kline_path(processed_dir: str | Path, market: str, interval: str, symbol: str) -> Path:
    return Path(processed_dir) / market / interval / f"{symbol}.parquet"


def load_klines(processed_dir: str | Path, market: str, interval: str, symbol: str) -> pd.DataFrame:
    return pd.read_parquet(kline_path(processed_dir, market, interval, symbol))


def load_panel(
    processed_dir: str | Path,
    market: str,
    interval: str,
    universe: dict[str, str],
    start: str | None = None,
    end: str | None = None,
    field: str = "close",
) -> pd.DataFrame:
    """Panel of `field` with Roostoo pair names as columns (end is exclusive)."""
    series = {}
    for pair, symbol in universe.items():
        path = kline_path(processed_dir, market, interval, symbol)
        if not path.exists():
            log.debug("no data for %s (%s)", pair, symbol)
            continue
        series[pair] = pd.read_parquet(path, columns=[field])[field]
    if not series:
        return pd.DataFrame()
    panel = pd.DataFrame(series).sort_index()
    if start:
        panel = panel[panel.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        panel = panel[panel.index < pd.Timestamp(end, tz="UTC")]
    return panel


OHLCV_FIELDS = ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote")


def load_ohlcv(processed_dir: str | Path, market: str, universe: dict[str, str], start: str | None = None,
               bar: str = "1h", fields: tuple[str, ...] = OHLCV_FIELDS) -> dict[str, pd.DataFrame]:
    """OHLCV panels (time x pair) from hourly klines, resampled to `bar` (bars labelled by open time).
    The last bar is dropped when resampling, as it may be incomplete."""
    raw = {f: load_panel(processed_dir, market, "1h", universe, start=start, field=f) for f in fields}
    if pd.Timedelta(bar) == pd.Timedelta(hours=1):
        return raw
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    out = {}
    for f, df in raw.items():
        r = df.resample(bar)
        out[f] = getattr(r, agg[f])() if f in agg else r.sum(min_count=1)
    return {f: v.iloc[:-1] for f, v in out.items()}
