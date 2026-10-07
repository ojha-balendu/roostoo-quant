"""Recent history for strategy warm-up from Binance's public market-data endpoint.

data-api.binance.vision serves public market data only (no keys). Binance prices
(USDT) are a close proxy for Roostoo (USD) prices; once running, the bot's own
recorded bars take precedence.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import requests

log = logging.getLogger(__name__)

KLINES_URL = "https://data-api.binance.vision/api/v3/klines"


def fetch_recent_klines(symbol: str, interval: str, n_bars: int,
                        session: requests.Session | None = None) -> pd.DataFrame:
    """Completed bars: close and quote_volume (USD traded), indexed by bar open time (UTC)."""
    session = session or requests.Session()
    rows: list[list] = []
    end_time: int | None = None
    remaining = n_bars
    while remaining > 0:
        params: dict = {"symbol": symbol, "interval": interval, "limit": min(1000, remaining)}
        if end_time is not None:
            params["endTime"] = end_time
        r = session.get(KLINES_URL, params=params, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows = batch + rows
        remaining -= len(batch)
        end_time = int(batch[0][0]) - 1
        if len(batch) < params["limit"]:
            break
    now_ms = time.time() * 1000
    closed = [r for r in rows if int(r[6]) < now_ms]  # completed bars only
    frame = pd.DataFrame(
        {"close": [float(r[4]) for r in closed], "quote_volume": [float(r[7]) for r in closed]},
        index=pd.to_datetime([int(r[0]) for r in closed], unit="ms", utc=True),
    )
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def fetch_recent_closes(symbol: str, interval: str, n_bars: int, session: requests.Session | None = None) -> pd.Series:
    return fetch_recent_klines(symbol, interval, n_bars, session)["close"].rename(symbol)


def warmup_panels(universe: dict[str, str], interval: str, n_bars: int) -> dict[str, pd.DataFrame]:
    """{"close": panel, "quote_volume": panel} (time x pair) from Binance's public klines."""
    session = requests.Session()
    closes, volumes = {}, {}
    for pair, symbol in universe.items():
        try:
            k = fetch_recent_klines(symbol, interval, n_bars, session)
        except (requests.RequestException, ValueError, IndexError) as exc:
            log.warning("warm-up: no Binance data for %s (%s): %s", pair, symbol, exc)
            continue
        closes[pair], volumes[pair] = k["close"], k["quote_volume"]
    if not closes:
        return {"close": pd.DataFrame(), "quote_volume": pd.DataFrame()}
    return {"close": pd.DataFrame(closes).sort_index(), "quote_volume": pd.DataFrame(volumes).sort_index()}


def warmup_close_panel(universe: dict[str, str], interval: str, n_bars: int) -> pd.DataFrame:
    return warmup_panels(universe, interval, n_bars)["close"]
