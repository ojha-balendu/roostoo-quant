"""How many milliseconds after Binance moves does Roostoo's price follow? (crypto pairs)

    python research/30_ms_latency.py --minutes 20
    python research/30_ms_latency.py --analyse reports/latency/ms_20261015_0100

Streams Binance best bid/ask over a WebSocket (every change pushed within milliseconds, no polling delay)
and reads ALL Roostoo prices as fast as Roostoo allows, each stamped with the local receive time. Both run
until the time is up (Ctrl+C stops early). PLACES NO ORDERS. Then, for every moment Roostoo's price changed,
it finds how many milliseconds earlier Binance already showed that move -- the real reaction lag. Run it
where the bot runs (EC2) for realistic network times; a fast order path (~100-200 ms) only profits if the
lag is comfortably larger than that.

Writes two files per run under reports/latency/<name>/: binance.csv (streamed mid prices) and roostoo.csv
(polled mid prices). Re-analyse later with --analyse <dir>.
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.analysis.latency import lag_profile
from quantbot.config import load_config
from quantbot.data.universe import load_universe

OUT = Path("reports/latency")
WS_HOST = "wss://data-stream.binance.vision/stream"


class BinanceStream:
    """Background WebSocket reader: appends (recv_ts, pair, mid) for every bookTicker update."""

    def __init__(self, symbol_to_pair: dict[str, str], writer, stop: threading.Event):
        import websocket

        self._websocket = websocket
        self.symbol_to_pair = symbol_to_pair
        self.writer = writer
        self.stop = stop
        self.n = 0
        self.lock = threading.Lock()
        streams = "/".join(f"{s.lower()}@bookTicker" for s in symbol_to_pair)
        self.url = f"{WS_HOST}?streams={streams}"
        self.ws = None

    def _on_message(self, _ws, message: str) -> None:
        recv = time.time()
        try:
            d = json.loads(message)["data"]
            bid, ask = float(d["b"]), float(d["a"])
            pair = self.symbol_to_pair.get(d["s"])
        except (KeyError, ValueError, TypeError):
            return
        if pair and bid > 0 and ask > 0:
            with self.lock:
                self.writer.writerow([f"{recv:.3f}", pair, (bid + ask) / 2])
                self.n += 1

    def _on_error(self, _ws, err) -> None:
        print(f"  binance ws error: {type(err).__name__}: {err}", flush=True)

    def run(self) -> None:
        while not self.stop.is_set():
            self.ws = self._websocket.WebSocketApp(
                self.url, on_message=self._on_message, on_error=self._on_error)
            # run_forever returns on disconnect; loop reconnects until stop is set
            self.ws.run_forever(ping_interval=20, ping_timeout=10)
            if not self.stop.is_set():
                time.sleep(1)

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass


def record(client, pairs: dict[str, str], minutes: float, path: Path) -> tuple[int, int]:
    path.mkdir(parents=True, exist_ok=True)
    symbol_to_pair = {s: p for p, s in pairs.items()}
    stop = threading.Event()
    b_fh = (path / "binance.csv").open("w", newline="")
    r_fh = (path / "roostoo.csv").open("w", newline="")
    b_w, r_w = csv.writer(b_fh), csv.writer(r_fh)
    b_w.writerow(["ts", "pair", "mid"])
    r_w.writerow(["ts", "pair", "mid"])

    stream = BinanceStream(symbol_to_pair, b_w, stop)
    t = threading.Thread(target=stream.run, daemon=True)
    t.start()
    time.sleep(2)                                                        # let the stream connect

    deadline = time.time() + minutes * 60
    r_polls, r_ms = 0, []
    print(f"recording {len(pairs)} pairs for {minutes:g} min into {path} (Ctrl+C stops early); no orders",
          flush=True)
    try:
        while time.time() < deadline:
            t0 = time.time()
            try:
                ticks = client.ticker()
            except Exception as exc:
                print(f"  roostoo read error: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(0.5)
                continue
            recv = time.time()
            r_ms.append((recv - t0) * 1000)
            with stream.lock:
                for pair, tk in ticks.items():
                    if pair in pairs and tk.bid > 0 and tk.ask > 0:
                        r_w.writerow([f"{recv:.3f}", pair, (tk.bid + tk.ask) / 2])
                r_fh.flush()
                b_fh.flush()
            r_polls += 1
            if r_polls % 200 == 0:
                print(f"  {r_polls} roostoo reads ({np.median(r_ms):.0f} ms each), "
                      f"{stream.n} binance updates", flush=True)
    except KeyboardInterrupt:
        print("\nstopped; analysing ...", flush=True)
    finally:
        stop.set()
        stream.close()
        b_fh.close()
        r_fh.close()
    return r_polls, stream.n


def analyse(path: Path, window_s: float, fast_path_ms: float) -> None:
    binance = pd.read_csv(path / "binance.csv")
    reads = pd.read_csv(path / "roostoo.csv")
    if binance.empty or reads.empty:
        raise SystemExit("one of the recordings is empty")
    span = max(reads["ts"].max() - reads["ts"].min(), 1e-9)
    r_rate = len(reads) / reads["pair"].nunique() / span
    print(f"\n{path}: {span / 60:.1f} min; {len(binance)} binance updates, {len(reads)} roostoo reads "
          f"across {reads['pair'].nunique()} pairs (~{r_rate:.1f} reads/s per pair)")

    prof = lag_profile(reads, binance, window_s=window_s)
    n_ch, n_un = prof.attrs["n_changes"], prof.attrs["n_unmatched"]
    if prof.empty:
        print(f"\n{n_ch} roostoo price changes, none matched a Binance move in the last {window_s:g}s. "
              "Nothing to measure.")
        return
    lag_ms = prof["lag_s"].to_numpy() * 1000
    matched = len(prof)
    print(f"\n{n_ch} roostoo price changes; {matched} matched a prior Binance move, {n_un} did not "
          f"({matched / n_ch:.0%} matched).")
    print("\nREACTION LAG (ms) -- how long after Binance moved that Roostoo's price caught up:")
    for q in (10, 25, 50, 75, 90):
        print(f"  {q}th percentile: {np.percentile(lag_ms, q):6.0f} ms")
    print(f"  mean: {lag_ms.mean():.0f} ms")

    print(f"\nShare of moves where the lag exceeds a given order path (so a {fast_path_ms:.0f} ms order "
          "would still beat Roostoo's update):")
    for ms in (100, 200, 300, 500, 1000):
        print(f"  lag > {ms:5d} ms: {(lag_ms > ms).mean():5.0%}")
    beatable = (lag_ms > fast_path_ms).mean()
    print(f"\nWith a {fast_path_ms:.0f} ms order path, {beatable:.0%} of these moves are catchable before "
          "Roostoo updates.")
    by_pair = prof.assign(lag_ms=lag_ms).groupby("pair")["lag_ms"].agg(["count", "median"])
    by_pair = by_pair[by_pair["count"] >= 5].sort_values("median", ascending=False)
    if not by_pair.empty:
        print("\nSlowest-reacting pairs (median lag, pairs with >=5 moves):")
        for pair, row in by_pair.head(12).iterrows():
            print(f"  {pair:<14} {row['median']:5.0f} ms  ({int(row['count'])} moves)")
    print("\nThis measures prices only. A real order still has to arrive and fill at the stale quote; only a\n"
          "tiny-size live order confirms that. Reaction lag below your order path means no edge on that pair.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--analyse", help="analyse an existing recording directory instead of recording")
    ap.add_argument("--window", type=float, default=5.0, help="seconds before a Roostoo change to look back")
    ap.add_argument("--fast-path-ms", type=float, default=200.0, help="assumed order round-trip-to-exchange")
    args = ap.parse_args()
    if args.analyse:
        analyse(Path(args.analyse), args.window, args.fast_path_ms)
        return
    cfg = load_config()
    from quantbot.exchange.client import RoostooClient

    client = RoostooClient.from_config(cfg.exchange)
    pairs = {p: s for p, s in load_universe(cfg.data.universe_path).items() if s}
    path = OUT / f"ms_{time.strftime('%Y%m%d_%H%M', time.gmtime())}"
    record(client, pairs, args.minutes, path)
    analyse(path, args.window, args.fast_path_ms)


if __name__ == "__main__":
    main()
