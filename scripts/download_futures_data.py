#!/usr/bin/env python3
"""Download Binance Vision funding rates, OI metrics and spot 1h klines (H5/H6 inputs).

Examples
    # the coins listed on Roostoo (stock tokens are skipped automatically: no USDT market)
    python scripts/download_futures_data.py --symbols-from-roostoo --start 2023-01-01

    # explicit list, or the symbols already present in your kline folder
    python scripts/download_futures_data.py --symbols BTCUSDT,ETHUSDT,SOLUSDT
    python scripts/download_futures_data.py --symbols-from-dir data/binance/klines

    # offline run against a local mirror (used for smoke tests)
    python scripts/download_futures_data.py --mirror /tmp/mirror --symbols BTCUSDT

Re-running only fetches new files.  Metrics are daily files (~1,300 per coin
since 2023), so the first full run takes a while; --workers controls parallelism.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402

from quantbot.data import futures_vision as fv  # noqa: E402

ROOSTOO_EXCHANGE_INFO = "https://mock-api.roostoo.com/v3/exchangeInfo"


def roostoo_symbols(url: str = ROOSTOO_EXCHANGE_INFO) -> list:
    with urllib.request.urlopen(url, timeout=20) as resp:
        info = json.loads(resp.read())
    pairs = info.get("TradePairs", {})
    return sorted({p.split("/")[0] + "USDT" for p in pairs if p.endswith("/USD")})


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--symbols", help="comma-separated spot symbols, e.g. BTCUSDT,ETHUSDT")
    src.add_argument("--symbols-from-dir", help="use symbols of Vision kline files found in this folder")
    src.add_argument("--symbols-from-roostoo", action="store_true", help="use Roostoo exchangeInfo pairs")
    ap.add_argument("--root", default="data/binance_vision", help="cache + processed output folder")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default=None, help="default: today (UTC)")
    ap.add_argument("--kinds", nargs="+", default=list(fv.KINDS), choices=list(fv.KINDS))
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--mirror", default=None, help="serve files from a local mirror instead of the internet")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.symbols_from_dir:
        symbols = sorted(fv.scan_kline_files(args.symbols_from_dir, args.interval))
    else:
        symbols = roostoo_symbols()
    if not symbols:
        print("no symbols found", file=sys.stderr)
        return 2
    end = args.end or pd.Timestamp.now(tz="UTC").normalize().isoformat()
    fetch = fv.http_get
    if args.mirror:
        from quantbot.data.synthetic_vision import LocalBucketFetcher
        fetch = LocalBucketFetcher(args.mirror)
    print(f"{len(symbols)} symbols, {args.start} -> {end}, kinds={args.kinds}")
    report, counts = fv.sync(symbols, args.root, args.start, end, kinds=args.kinds,
                             interval=args.interval, fetch=fetch, workers=args.workers)
    cols = [c for c in ("spot_symbol", "futures_symbol", "klines_rows", "funding_rows", "metrics_rows",
                        "metrics_first") if c in report.columns]
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(report[cols].to_string(index=False))
    print(f"downloads: {counts}")
    no_fut = report.loc[report["futures_symbol"].isna(), "spot_symbol"].tolist()
    if no_fut and set(args.kinds) & {"funding", "metrics"}:
        print(f"no USD-M perp found for: {', '.join(no_fut)}")
    print(f"symbol map: {Path(args.root) / 'processed' / 'symbol_map.csv'}")
    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
