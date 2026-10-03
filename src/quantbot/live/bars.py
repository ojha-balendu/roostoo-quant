"""Roostoo has no klines endpoint, so live bars are built from polled tickers."""

from __future__ import annotations

import re

import pandas as pd

_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def parse_interval(interval: str) -> pd.Timedelta:
    m = re.fullmatch(r"(\d+)([smhdw])", interval.strip())
    if not m:
        raise ValueError(f"unsupported bar interval {interval!r} (use e.g. 1m, 15m, 1h, 4h, 1d)")
    return pd.Timedelta(**{_UNITS[m.group(2)]: int(m.group(1))})


class BarBuilder:
    """Aggregates price ticks into OHLC bars labelled by bar OPEN time (Binance convention).

    A bar is emitted when the first tick of the next bar arrives. The first bar after a
    (re)start is partial.
    """

    def __init__(self, interval: str):
        self.interval_ms = int(parse_interval(interval).total_seconds() * 1000)
        self._bucket: int | None = None
        self._ohlc: dict[str, list[float]] = {}

    def update(self, ts_ms: int, prices: dict[str, float]) -> tuple[pd.Timestamp, pd.DataFrame] | None:
        bucket = ts_ms - ts_ms % self.interval_ms
        if self._bucket is not None and bucket < self._bucket:
            return None  # stale tick
        closed = None
        if self._bucket is not None and bucket > self._bucket:
            if self._ohlc:
                df = pd.DataFrame.from_dict(self._ohlc, orient="index", columns=["open", "high", "low", "close"])
                closed = (pd.Timestamp(self._bucket, unit="ms", tz="UTC"), df)
            self._ohlc = {}
        self._bucket = bucket
        for pair, px in prices.items():
            if not px or px <= 0:
                continue
            bar = self._ohlc.get(pair)
            if bar is None:
                self._ohlc[pair] = [px, px, px, px]
            else:
                bar[1] = max(bar[1], px)
                bar[2] = min(bar[2], px)
                bar[3] = px
        return closed
