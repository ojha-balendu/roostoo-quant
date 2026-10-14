"""Does Roostoo's price lag Binance? Record both every second, then measure the lag and what it would pay.

    python research/29_latency_probe.py --minutes 30                 # record 30 minutes, then analyse
    python research/29_latency_probe.py --analyse reports/latency/probe_20261008_1200.csv

Each poll fetches ALL Roostoo prices (public ticker, no signature: the bot's account is not touched)
and ALL Binance best bid/ask prices (public data-api.binance.vision), one request each, stamped with
the local receive time and appended to a CSV (safe to stop with Ctrl+C; the analysis then runs).
PLACES NO ORDERS. Analysis per pair: how often Roostoo's price changes, the delay at which it best matches
Binance's earlier price, and the trades that delay would allow net of a 0.2% round trip (buy at
Roostoo's ask when Binance is above it by more than the costs, sell at Roostoo's bid a few seconds later).
Run it where the bot runs (the EC2 instance) for realistic network times.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from quantbot.analysis.latency import lag_estimate, margin_sweep, opportunities, panels, update_stats
from quantbot.config import load_config
from quantbot.data.universe import load_universe

BINANCE = "https://data-api.binance.vision/api/v3/ticker/bookTicker"
OUT = Path("reports/latency")
FIELDS = ["poll", "ts", "source", "pair", "bid", "ask", "last", "request_ms"]


def binance_fetcher(session: requests.Session, symbols: list[str]) -> Callable[[], dict[str, tuple[float, float]]]:
    """Returns fetch() -> {symbol: (bid, ask)}; asks for just our symbols, or everything if Binance rejects one."""
    params = {"symbols": json.dumps(symbols, separators=(",", ":"))}
    state = {"all": False}

    def fetch() -> dict[str, tuple[float, float]]:
        resp = session.get(BINANCE, params=None if state["all"] else params, timeout=5)
        if resp.status_code == 400 and not state["all"]:
            state["all"] = True
            resp = session.get(BINANCE, timeout=5)
        resp.raise_for_status()
        wanted = set(symbols)
        return {d["symbol"]: (float(d["bidPrice"]), float(d["askPrice"])) for d in resp.json() if d["symbol"] in wanted}
    return fetch


def record(roostoo_fetch, binance_fetch, pair_to_symbol: dict[str, str], n_polls: int, interval: float,
           path: Path, progress_every: int = 60) -> int:
    """Poll both sources n_polls times, appending rows to `path`. Returns the number of completed polls."""
    path.parent.mkdir(parents=True, exist_ok=True)
    symbol_to_pair = {s: p for p, s in pair_to_symbol.items()}
    done, r_ms, b_ms = 0, [], []
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        try:
            for poll in range(n_polls):
                start = time.time()
                try:
                    t0 = time.time()
                    rt = roostoo_fetch()
                    t1 = time.time()
                    bt = binance_fetch()
                    t2 = time.time()
                except Exception as exc:  # network hiccup: skip this poll
                    print(f"  poll {poll}: {type(exc).__name__}: {exc}")
                    time.sleep(max(0.0, interval - (time.time() - start)))
                    continue
                r_ms.append((t1 - t0) * 1000)
                b_ms.append((t2 - t1) * 1000)
                for pair, tk in rt.items():
                    if pair in pair_to_symbol and tk.bid > 0 and tk.ask > 0:
                        w.writerow([poll, f"{t1:.3f}", "roostoo", pair, tk.bid, tk.ask, tk.last, f"{r_ms[-1]:.0f}"])
                for sym, (bid, ask) in bt.items():
                    if sym in symbol_to_pair and bid > 0 and ask > 0:
                        w.writerow([poll, f"{t2:.3f}", "binance", symbol_to_pair[sym], bid, ask, (bid + ask) / 2,
                                    f"{b_ms[-1]:.0f}"])
                fh.flush()
                done += 1
                if progress_every and done % progress_every == 0:
                    print(f"  {done}/{n_polls} polls; request time Roostoo {np.median(r_ms):.0f} ms, "
                          f"Binance {np.median(b_ms):.0f} ms (medians)", flush=True)
                time.sleep(max(0.0, interval - (time.time() - start)))
        except KeyboardInterrupt:
            print("\nstopped; analysing what was recorded ...")
    return done


def analyse(path: Path, fee: float, margin_bps: float, max_lag_s: float) -> None:
    rec = pd.read_csv(path)
    if rec.empty:
        raise SystemExit("nothing recorded")
    p = panels(rec)
    sp = p["seconds_per_poll"]
    if not np.isfinite(sp) or sp <= 0:                                   # polls too close to time: count in polls
        sp = 1.0
    n_polls = rec["poll"].nunique()
    hours = n_polls * sp / 3600
    print(f"\n{path}: {n_polls} polls, {len(p['r_last'].columns)} pairs, one poll every {sp:.2f} s "
          f"({hours * 60:.0f} minutes); request time medians: Roostoo "
          f"{rec.loc[rec['source'] == 'roostoo', 'request_ms'].median():.0f} ms, Binance "
          f"{rec.loc[rec['source'] == 'binance', 'request_ms'].median():.0f} ms")
    upd = update_stats(p["r_last"], sp)
    lag = lag_estimate(p["r_last"], p["b_mid"], max_lag=max(1, int(round(max_lag_s / sp))))
    t = lag.join(upd)
    t["lag_seconds"] = t["lag_polls"] * sp
    t = t.sort_values("lag_seconds", ascending=False)
    pd.set_option("display.width", 200)
    print("\nPER PAIR: delay at which Roostoo best matches Binance's earlier price (0 = no lag at this resolution)")
    print(pd.DataFrame({
        "lag (s)": t["lag_seconds"].map("{:.0f}".format),
        "mismatch at that lag (bps)": t["error_at_lag_bps"].map("{:.1f}".format),
        "mismatch with no lag (bps)": t["error_at_zero_bps"].map("{:.1f}".format),
        "Roostoo price changed in": t["share_of_polls_changed"].map("{:.0%} of polls".format),
        "median s between changes": t["median_seconds_between_changes"].map(
            lambda v: f"{v:.0f}" if np.isfinite(v) else "n/a")}).to_string())
    lagged = t[t["lag_polls"] > 0]
    print(f"\n{len(lagged)} of {len(t)} pairs lag Binance; median lag {t['lag_seconds'].median():.0f} s.")

    print(f"\nTRADES THE LAG WOULD ALLOW (enter when Binance is more than {2 * fee * 1e4 + margin_bps:.0f} bps beyond "
          "Roostoo's quote; exit at Roostoo's quote; net of a 0.2% round trip; no execution delay assumed):")
    rows, top_text = [], None
    for h_s in (5, 10, 30, 60):
        h = max(1, int(round(h_s / sp)))
        ops = opportunities(p, fee, margin_bps, h)
        rows.append({"exit after": f"{h_s} s", "trades": len(ops), "per hour": len(ops) / hours if hours else np.nan,
                     "mean net (bps)": ops["net_bps"].mean() if len(ops) else np.nan,
                     "won": (ops["net_bps"] > 0).mean() if len(ops) else np.nan,
                     "pairs": ops["pair"].nunique() if len(ops) else 0})
        if h_s == 10 and len(ops):
            top = ops.groupby("pair")["net_bps"].agg(["count", "mean"]).sort_values("count", ascending=False).head(8)
            top_text = ", ".join(f"{k.split('/')[0]} {int(v['count'])}x {v['mean']:+.0f}bps" for k, v in top.iterrows())
    print(pd.DataFrame(rows).to_string(index=False, formatters={
        "per hour": "{:.1f}".format, "mean net (bps)": lambda v: f"{v:+.1f}" if np.isfinite(v) else "n/a",
        "won": lambda v: f"{v:.0%}" if np.isfinite(v) else "n/a"}))
    if top_text:
        print(f"  most frequent at 10 s: {top_text}")
    print("\nBefore relying on this: an order needs time to reach Roostoo (if its price updates first, the trade is\n"
          "gone), Roostoo may allow only one order per minute, and the competition rules may forbid exploiting\n"
          "stale prices. This probe measures prices only; check the rules before trading on any of it.")


def sweep_report(path: Path, fee: float, margins: list[float], horizons_s: list[float], cap: float,
                 kelly_scale: float, delays_s: list[float]) -> None:
    rec = pd.read_csv(path)
    p = panels(rec)
    sp = p["seconds_per_poll"]
    if not np.isfinite(sp) or sp <= 0:
        sp = 1.0
    hours = rec["poll"].nunique() * sp / 3600
    print(f"{path}: {hours:.1f} hours; margins {margins[0]:g}..{margins[-1]:g} bps (entry gap = {2 * fee * 1e4:.0f} bps "
          f"round trip + margin); size = {kelly_scale:g} x Kelly with the mean cut by one standard error, "
          f"capped at {cap:.0%} of equity per trade")
    for d_s in delays_s:
        d = int(round(d_s / sp))
        for h_s in horizons_s:
            h = max(1, int(round(h_s / sp)))
            t = margin_sweep(p, fee, margins, h, hours, cap, kelly_scale, d)
            print(f"\nENTRY {d_s:g} s AFTER THE SIGNAL, EXIT {h_s:g} s LATER:")
            print(pd.DataFrame({
                "entry gap": t["entry_gap_bps"].map("{:.0f} bps".format),
                "avg gap": t["mean_gap_bps"].map(lambda v: f"{v:.0f} bps" if np.isfinite(v) else "n/a"),
                "trades": t["trades"],
                "per hour": t["per_hour"].map("{:.1f}".format), "coins": t["pairs"],
                "won": t["win"].map(lambda v: f"{v:.0%}" if np.isfinite(v) else "n/a"),
                "mean net": t["mu"].map(lambda v: f"{v * 1e4:+.1f} bps" if np.isfinite(v) else "n/a"),
                "spread": t["sigma"].map(lambda v: f"{v * 1e4:.1f} bps" if np.isfinite(v) else "n/a"),
                "full Kelly": t["kelly"].map(lambda v: f"{v:.1f}x" if np.isfinite(v) else "n/a"),
                "cautious Kelly": t["kelly_cons"].map(lambda v: f"{v:.1f}x" if np.isfinite(v) else "n/a"),
                "size used": t["f"].map("{:.0%}".format),
                "profit/day": t["profit_day"].map("{:+.2%}".format),
                "growth/day": t["growth_day"].map("{:+.2%}".format)}).set_axis(
                    [f"margin {m:g}" for m in t["margin_bps"]]).to_string())
            best = t.loc[t["growth_day"].idxmax()] if (t["growth_day"] > 0).any() else None
            if best is None:
                print("  no margin has a positive edge after the cautious adjustment")
                continue
            warn = "  (few trades: confirm on a new recording)" if best["trades"] < 30 else ""
            print(f"  BEST (delay {d_s:g} s, exit {h_s:g} s): margin {best['margin_bps']:g} bps (enter at a {best['entry_gap_bps']:.0f} bps gap): "
                  f"{best['per_hour']:.1f} trades/hour, {best['win']:.0%} won, {best['mu'] * 1e4:+.1f} bps net per trade, "
                  f"size {best['f']:.0%} of equity -> about {best['growth_day']:+.2%} per day{warn}")
    print("\nDelay 0 enters at the Roostoo quote fetched just before the Binance price that triggered the trade\n"
          "(optimistic); delay 1 enters at the next poll's quote, about a second later (pessimistic: a real order\n"
          "arrives in ~0.2 s). Trust a margin only if it is profitable with the delay.\n"
          "'cautious Kelly' uses the average minus one standard error, so a margin with a few lucky trades is not\n"
          "favoured. Kelly far above 1x means the edge per trade is large relative to its spread; the cap keeps\n"
          "the size realistic. Picking the best margin on this recording flatters it: confirm on a new one.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between polls")
    ap.add_argument("--analyse", help="analyse an existing recording instead of recording")
    ap.add_argument("--fee", type=float, default=None, help="per side (default: taker fee from the config)")
    ap.add_argument("--margin-bps", type=float, default=5.0, help="extra gap required beyond the round-trip fee")
    ap.add_argument("--max-lag", type=float, default=30, help="longest lag to look for, seconds")
    ap.add_argument("--sweep", help="margin range for a Kelly sweep, e.g. 5:30 (with --analyse)")
    ap.add_argument("--horizons", default="10,60", help="exit times for the sweep, seconds")
    ap.add_argument("--delays", default="0,1", help="entry delays after the signal for the sweep, seconds")
    ap.add_argument("--cap", type=float, default=0.25, help="largest share of equity per trade")
    ap.add_argument("--kelly-scale", type=float, default=0.5, help="fraction of Kelly to use (0.5 = half Kelly)")
    args = ap.parse_args()
    cfg = load_config()
    fee = args.fee if args.fee is not None else cfg.fees.taker
    if args.analyse and args.sweep:
        lo, hi = (float(v) for v in args.sweep.split(":"))
        margins = [float(m) for m in np.arange(lo, hi + 1e-9, 1.0)]
        sweep_report(Path(args.analyse), fee, margins, [float(h) for h in args.horizons.split(",")], args.cap,
                     args.kelly_scale, [float(d) for d in args.delays.split(",")])
        return
    if args.analyse:
        analyse(Path(args.analyse), fee, args.margin_bps, args.max_lag)
        return
    from quantbot.exchange.client import RoostooClient

    client = RoostooClient.from_config(cfg.exchange)
    pairs = {p: s for p, s in load_universe(cfg.data.universe_path).items() if s}
    session = requests.Session()
    path = OUT / f"probe_{time.strftime('%Y%m%d_%H%M', time.gmtime())}.csv"
    n = int(args.minutes * 60 / args.interval)
    print(f"recording {len(pairs)} pairs every {args.interval:g} s for {args.minutes:g} minutes into {path} "
          "(Ctrl+C stops early and analyses); no orders are placed")
    record(lambda: client.ticker(), binance_fetcher(session, sorted(set(pairs.values()))), pairs, n, args.interval, path)
    analyse(path, fee, args.margin_bps, args.max_lag)


if __name__ == "__main__":
    main()
