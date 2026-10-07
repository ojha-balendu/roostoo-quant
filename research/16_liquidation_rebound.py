"""H5: after a liquidation flush, does the coin rebound? Event study on hourly bars.

    quantbot futures --what both              # once: funding + 5-minute open interest (last 365 days)
    python research/16_liquidation_rebound.py
    python research/16_liquidation_rebound.py --sigma 2.5 --oi-drop 0.03

Event at bar t (all known at the close of t):
  the coin's 1-hour return is below -sigma x its trailing volatility,
  its open interest fell by >= oi-drop during that hour (positions being closed: forced liquidations),
  and its funding was positive beforehand (longs were crowded).
Trade: buy at the close of bar t, sell `hold` hours later; costs = one round trip.
Compared with "big drop without an OI flush" (same price drop, OI not falling): if the flush matters,
those should rebound less. Events cluster on market-wide crash days, so the t-statistic treats each
day as one observation. Open-interest history covers only the downloaded days, so instead of a
separate holdout the two halves of the sample must both be positive.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.analysis.positioning import bar_oi_change, day_clustered, event_returns, flush_events
from quantbot.analysis.xsection import eligibility, residual_returns
from quantbot.config import load_config
from quantbot.data.futures import asof, load_futures_panel
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe

BTC = "BTC/USD"
OUT = Path("reports/research")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=3.0, help="drop size in trailing standard deviations")
    ap.add_argument("--oi-drop", type=float, default=0.05, help="open-interest fall within the hour")
    ap.add_argument("--any-funding", action="store_true", help="don't require positive funding beforehand")
    ap.add_argument("--holds", default="4,12,24", help="holding periods in hours")
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--min-net-bps", type=float, default=5.0)
    ap.add_argument("--min-t", type=float, default=3.0)
    ap.add_argument("--min-events", type=int, default=30)
    ap.add_argument("--min-days", type=int, default=15)
    args = ap.parse_args()
    holds = [int(x) for x in args.holds.split(",")]
    cost = args.cost_bps / 1e4

    cfg = load_config()
    universe = load_universe(cfg.data.universe_path)
    oi = load_futures_panel(cfg.data.processed_dir, "metrics", "oi")
    funding = load_futures_panel(cfg.data.processed_dir, "fundingRate", "rate")
    if oi.empty or funding.empty:
        raise SystemExit("need open interest and funding: run  quantbot futures --what both")
    start = oi.index.min().floor("h") - pd.Timedelta(days=31)               # extra month for volatility / betas
    close = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", universe, start=str(start.date())).ffill(limit=3)
    qv = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", universe, start=str(start.date()), field="quote_volume")
    coins = [c for c in close.columns if c in oi.columns and c in funding.columns and c != BTC]
    _, beta = residual_returns(close, BTC)
    close_c = close[coins]
    eligible = eligibility(close, qv)[coins]
    oi_chg = bar_oi_change(oi[coins], close.index, pd.Timedelta(hours=1))
    funding_before = asof(funding[coins], close.index)                     # latest settled rate at the bar's start
    ev = flush_events(close_c, oi_chg, funding_before, eligible, sigma_mult=args.sigma, oi_drop=args.oi_drop,
                      need_positive_funding=not args.any_funding)
    covered = oi_chg.notna()
    ev = {k: v & covered for k, v in ev.items()}
    span = close.index[covered.any(axis=1).to_numpy()]
    print(f"{len(coins)} coins with open interest; events searched {span.min():%Y-%m-%d} -> {span.max():%Y-%m-%d}; "
          f"drop < -{args.sigma:g} sigma, OI fall >= {args.oi_drop:.0%}"
          + ("" if args.any_funding else ", funding > 0 before") + f"; cost {args.cost_bps:g} bps per round trip")
    print(f"bars with a big drop: {int(ev['big_drop'].to_numpy().sum())}, of which with an OI flush: "
          f"{int(ev['flush'].to_numpy().sum())}, without (OI not falling): {int(ev['no_flush'].to_numpy().sum())}\n")

    rows = []
    mid = span.min() + (span.max() - span.min()) / 2
    for H in holds:
        for group in ("flush", "no_flush"):
            er = event_returns(ev[group], close_c, close[BTC], beta, H, cost)
            s = day_clustered(er, "net")
            adj = day_clustered(er, "market_adj") if len(er) else {"mean": np.nan, "t_day": np.nan}
            first = er[er["time"] < mid]["net"].mean() if len(er) else np.nan
            second = er[er["time"] >= mid]["net"].mean() if len(er) else np.nan
            rows.append({"hold": H, "group": group, **s, "gross": er["gross"].mean() if len(er) else np.nan,
                         "adj": adj["mean"], "adj_t": adj["t_day"], "first": first, "second": second})
    t = pd.DataFrame(rows)
    show = pd.DataFrame({
        "hold": t["hold"].map("{}h".format),
        "group": t["group"].map({"flush": "drop + OI flush", "no_flush": "drop, no flush"}),
        "events": t["events"], "days": t["days"],
        "gross bps": (t["gross"] * 1e4).map("{:+.0f}".format), "net bps": (t["mean"] * 1e4).map("{:+.0f}".format),
        "t (by day)": t["t_day"].map("{:+.1f}".format), "won": t["hit"].map("{:.0%}".format),
        "vs BTC bps": (t["adj"] * 1e4).map("{:+.0f}".format),
        "halves bps": [f"{a * 1e4:+.0f} / {b * 1e4:+.0f}" for a, b in zip(t["first"], t["second"], strict=True)]})
    pd.set_option("display.width", 200)
    print(show.to_string(index=False))
    print("net = after one round trip; 'vs BTC' = return minus beta x BTC's return over the same hold;\n"
          "t (by day) treats each crash day as one observation.")

    line = "=" * 100
    print(f"\n{line}\nVERDICT (net >= {args.min_net_bps:g} bps per event, t by day >= {args.min_t:g}, >= {args.min_events} "
          f"events on >= {args.min_days} days, both halves positive, and better than drops without a flush)")
    found = False
    for H in holds:
        f, n = t[(t["hold"] == H) & (t["group"] == "flush")].iloc[0], t[(t["hold"] == H) & (t["group"] == "no_flush")].iloc[0]
        checks = {"profit": f["mean"] * 1e4 >= args.min_net_bps, "confidence": f["t_day"] >= args.min_t,
                  "sample": f["events"] >= args.min_events and f["days"] >= args.min_days,
                  "both halves": f["first"] > 0 and f["second"] > 0,
                  "flush matters": not np.isfinite(n["mean"]) or f["mean"] > n["mean"]}
        failed = [k for k, v in checks.items() if not v]
        found |= not failed
        print(f"  hold {H:>2}h: {'PASSES' if not failed else 'fails: ' + ', '.join(failed)}")
    print("  -> worth simulating as an overlay next." if found else "  -> no tradeable rebound after costs.")
    print(line)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "liquidation_rebound.csv", index=False)


if __name__ == "__main__":
    main()
