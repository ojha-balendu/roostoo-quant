"""Command line entry point:  quantbot <command>  (or  python -m quantbot <command>)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

from quantbot.config import Config, load_config
from quantbot.logging_setup import setup_logging

log = logging.getLogger("quantbot")


def _client(cfg: Config):
    from quantbot.exchange.client import RoostooClient

    return RoostooClient.from_config(cfg.exchange)


def _parse_date(s: str | None, default: date) -> date:
    return date.fromisoformat(s) if s else default


# ------------------------------------------------------------------ commands
def cmd_check(cfg: Config, args) -> int:
    from quantbot.portfolio import fetch_snapshot

    client = _client(cfg)
    offset = client.sync_time()
    print(f"Roostoo reachable at {cfg.exchange.base_url}; local clock offset {offset:+d} ms")
    pairs = client.pairs()
    print(f"{len(pairs)} pairs listed, {sum(p.can_trade for p in pairs.values())} tradable")
    tickers = client.ticker()
    dead = sorted(set(pairs) - set(tickers))
    print(f"{len(tickers)} pairs have live prices" + (f"; no live price: {dead}" if dead else ""))
    top = sorted(tickers.values(), key=lambda t: t.quote_volume, reverse=True)[:8]
    print("\nTop pairs by 24h volume:")
    for t in top:
        print(f"  {t.pair:<12} bid {t.bid:<14g} ask {t.ask:<14g} spread {t.spread_bps:6.1f} bps")
    if not client.has_credentials:
        print("\nNo API keys in .env; skipping signed endpoints.")
        return 0
    snap = fetch_snapshot(client, tickers, cfg.accounting.short_collateral_in_usd_balance)
    print(f"\nAccount: {json.dumps(snap.summary())}")
    print(f"Pending orders: {client.pending_count()}")
    for pair, pos in snap.shorts.items():
        print(f"  short {pair}: qty {pos.qty} entry {pos.entry_price} uPnL {pos.unrealized_pnl:.2f}")
    return 0


def cmd_universe(cfg: Config, args) -> int:
    from quantbot.data.universe import build_universe, read_universe_file, save_universe

    pairs = _client(cfg).pairs()
    existing = read_universe_file(cfg.data.universe_path)
    mapping = build_universe(pairs, cfg.data.symbol_overrides, existing)
    save_universe(cfg.data.universe_path, mapping, pairs)
    print(f"wrote {len(mapping)} pairs to {cfg.data.universe_path}")
    print("Review it: tokenized stocks and renamed tickers need a manual symbol or null.")
    return 0


def cmd_download(cfg: Config, args) -> int:
    from quantbot.data.binance_vision import BinanceVisionDownloader, process_klines
    from quantbot.data.universe import load_universe

    interval = args.interval or cfg.data.interval
    market = args.market or cfg.data.market
    symbols = args.symbols or sorted(set(load_universe(cfg.data.universe_path).values()))
    if not symbols:
        print("no symbols: run `quantbot universe` or pass --symbols", file=sys.stderr)
        return 1
    start = _parse_date(args.start or cfg.data.start, date(2023, 1, 1))
    end = _parse_date(args.end or cfg.data.end, datetime.now(timezone.utc).date())
    dl = BinanceVisionDownloader(cfg.data.raw_dir, market=market, data_type=args.data_type, interval=interval,
                                 max_workers=cfg.data.max_workers, verify_checksum=cfg.data.verify_checksum)
    print(f"downloading {args.data_type} {interval} {market} for {len(symbols)} symbols, {start} -> {end}")
    report = dl.download(symbols, start, end)
    no_data = [s for s, c in report.items() if c.get("ok", 0) + c.get("exists", 0) == 0]
    failed = {s: c["failed"] for s, c in report.items() if c.get("failed")}
    if args.data_type == "klines" and not args.no_process:
        for sym in symbols:
            out = process_klines(cfg.data.raw_dir, cfg.data.processed_dir, market, sym, interval)
            if out:
                log.info("processed %s -> %s", sym, out)
    if failed:
        print(f"\nfailed downloads (re-run to retry): {failed}")
    if no_data:
        print(f"\nno Binance data at all for: {no_data}")
        print("Set these to null (or the correct symbol) in the universe file.")
    return 0


def cmd_backtest(cfg: Config, args) -> int:
    from quantbot.backtest.engine import run_backtest
    from quantbot.data.loader import load_panel
    from quantbot.data.universe import load_universe
    from quantbot.risk import clip_weights_frame
    from quantbot.strategy.registry import get_strategy

    name = args.strategy or cfg.strategy.name
    params = json.loads(args.params) if args.params else (cfg.strategy.params if name == cfg.strategy.name else {})
    interval = args.interval or cfg.data.interval
    universe = load_universe(cfg.data.universe_path)
    closes = load_panel(cfg.data.processed_dir, cfg.data.market, interval, universe,
                        args.start or cfg.data.start, args.end or cfg.data.end)
    if closes.empty:
        print("no data: run `quantbot download` first", file=sys.stderr)
        return 1
    strategy = get_strategy(name, params)
    weights = clip_weights_frame(strategy.generate_weights(closes.ffill()), cfg.risk)
    bt = cfg.backtest
    result = run_backtest(
        closes, weights, initial_capital=bt.initial_capital, fees=cfg.fees, long_fee=bt.long_fee,
        spread_bps=bt.spread_bps, delay_bars=bt.execution_delay_bars,
        min_trade_usd=cfg.execution.min_trade_usd, min_close_usd=cfg.execution.min_close_usd,
        threshold=cfg.execution.rebalance_threshold, cash_buffer=cfg.execution.cash_buffer,
    )
    print(f"strategy={name} params={params} interval={interval} pairs={closes.shape[1]} "
          f"bars={len(closes)} {closes.index[0]} -> {closes.index[-1]}\n")
    print(result.summary())
    out = Path(bt.report_dir) / f"{name}_{interval}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    result.equity.to_csv(out / "equity.csv")
    result.trades.to_csv(out / "trades.csv", index=False)
    (out / "metrics.json").write_text(json.dumps({"strategy": name, "params": params, **result.metrics}, indent=2))
    print(f"\nreport saved to {out}")
    return 0


def cmd_live(cfg: Config, args) -> int:
    from quantbot.data.universe import load_universe
    from quantbot.live.runner import LiveRunner
    from quantbot.live.store import StateStore
    from quantbot.strategy.registry import get_strategy

    if args.dry_run is not None:
        cfg.live.dry_run = args.dry_run
    client = _client(cfg)
    if not client.has_credentials:
        print("ROOSTOO_API_KEY / ROOSTOO_SECRET_KEY must be set (even dry-run reads the account)", file=sys.stderr)
        return 1
    universe = load_universe(cfg.data.universe_path)
    strategy = get_strategy(cfg.strategy.name, cfg.strategy.params)
    store = StateStore(cfg.live.db_path)
    LiveRunner(cfg, client, strategy, universe, store).run()
    return 0


def cmd_flatten(cfg: Config, args) -> int:
    from quantbot.execution.executor import Executor
    from quantbot.portfolio import fetch_snapshot

    client = _client(cfg)
    client.sync_time()

    def snap():
        return fetch_snapshot(client, None, cfg.accounting.short_collateral_in_usd_balance)

    ex = Executor(client, client.pairs(), cfg.execution, cfg.fees, dry_run=not args.execute)
    results = ex.rebalance({}, snap(), refresh=snap)
    print(f"{len(results)} actions ({'executed' if args.execute else 'dry-run; add --execute to send'})")
    return 0


def cmd_smoke_test(cfg: Config, args) -> int:
    """Exercise every endpoint once. With --trade, round-trips a tiny long and a tiny short."""
    from quantbot.exchange.rules import floor_to, fmt_decimal

    client = _client(cfg)
    print(f"clock offset: {client.sync_time():+d} ms")
    pairs = client.pairs()
    info = pairs[args.pair]
    tick = client.ticker(args.pair)[args.pair]
    print(f"{args.pair}: bid {tick.bid} ask {tick.ask}; precision qty={info.amount_precision} "
          f"price={info.price_precision}; min notional {info.min_notional}")
    bal = client.balance()
    print(f"USD free {bal.free.get('USD', 0):.2f} lock {bal.lock.get('USD', 0):.2f}")
    print(f"pending orders: {client.pending_count()}; open shorts: {len(client.short_positions())}")
    if not args.trade:
        print("read-only checks passed. Re-run with --trade to place tiny real orders.")
        return 0

    usd = args.usd
    qty = floor_to(usd / tick.ask, info.amount_precision)
    print(f"\n[1/4] LONG  BUY {qty} {args.pair}")
    print(json.dumps(client.place_order(args.pair, "BUY", fmt_decimal(qty, info.amount_precision)), indent=1))
    time.sleep(1)
    free = floor_to(client.balance().free.get(info.coin, 0.0), info.amount_precision)
    print(f"[2/4] SELL {free} {args.pair}")
    print(json.dumps(client.place_order(args.pair, "SELL", fmt_decimal(free, info.amount_precision)), indent=1))
    time.sleep(1)

    before = client.balance()
    print(f"[3/4] SHORT {args.pair} collateral {usd:.2f}")
    print(json.dumps(client.short_open(args.pair, fmt_decimal(usd, 2)), indent=1))
    time.sleep(1)
    after = client.balance()
    print(f"open shorts: {client.short_positions()}")
    drop = before.total("USD") - after.total("USD")
    in_balance = drop < 0.5 * usd
    print(f"USD free+lock dropped by {drop:.4f} after locking {usd:.2f} collateral")
    print(f"=> set accounting.short_collateral_in_usd_balance: {str(in_balance).lower()}")
    print(f"[4/4] CLOSE {args.pair}")
    print(json.dumps(client.short_close(args.pair), indent=1))
    return 0


# ------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quantbot", description="Roostoo competition trading bot")
    p.add_argument("--config", help="config file (default config/config.yaml or $QUANTBOT_CONFIG)")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="connectivity, exchange info, tickers and (with keys) account state")
    sub.add_parser("universe", help="write config/universe.yaml from Roostoo exchangeInfo")

    d = sub.add_parser("download", help="download Binance Vision history")
    d.add_argument("--symbols", nargs="*", help="Binance symbols (default: all in universe)")
    d.add_argument("--interval")
    d.add_argument("--market", choices=["spot", "um", "cm"])
    d.add_argument("--start")
    d.add_argument("--end")
    d.add_argument("--data-type", default="klines", choices=["klines", "aggTrades", "trades"])
    d.add_argument("--no-process", action="store_true", help="skip building parquet files")

    b = sub.add_parser("backtest", help="backtest a strategy on downloaded data")
    b.add_argument("--strategy")
    b.add_argument("--params", help='JSON, e.g. \'{"lookback_bars": 48}\'')
    b.add_argument("--interval")
    b.add_argument("--start")
    b.add_argument("--end")

    live = sub.add_parser("live", help="run the autonomous trading loop")
    live.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None,
                      help="override live.dry_run from config")

    f = sub.add_parser("flatten", help="close every long and short")
    f.add_argument("--execute", action="store_true", help="actually send orders (default: dry-run)")

    s = sub.add_parser("smoke-test", help="exercise every endpoint once")
    s.add_argument("--trade", action="store_true", help="round-trip a tiny long and short")
    s.add_argument("--pair", default="BTC/USD")
    s.add_argument("--usd", type=float, default=25.0)
    return p


COMMANDS = {
    "check": cmd_check,
    "universe": cmd_universe,
    "download": cmd_download,
    "backtest": cmd_backtest,
    "live": cmd_live,
    "flatten": cmd_flatten,
    "smoke-test": cmd_smoke_test,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg.live.log_dir if args.cmd == "live" else None, args.log_level)
    return COMMANDS[args.cmd](cfg, args)
