#!/usr/bin/env python3
"""H5: liquidation-flush rebound event study.

Prerequisite: python scripts/download_futures_data.py ... (funding + metrics + klines)

    python scripts/run_h5_liquidation_flush.py                       # in-sample, default params
    python scripts/run_h5_liquidation_flush.py --grid                # + sensitivity grid (full condition)
    python scripts/run_h5_liquidation_flush.py --klines-dir data/binance/klines   # your own 1h files
    python scripts/run_h5_liquidation_flush.py --evaluate-holdout    # ONCE, after freezing params

Decision rule (pre-registered): the default parameters, condition "full",
must pass the protocol (net >= 5 bps, day-clustered t >= 3, >= 30 event days,
positive in both halves).  The grid is for judging robustness, not for picking
the best cell.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantbot.analysis import liquidation_flush as lf  # noqa: E402
from quantbot.analysis.study_window import claim_holdout, make_window  # noqa: E402
from quantbot.data.futures_vision import load_research_inputs  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/binance_vision")
    ap.add_argument("--klines-dir", default=None, help="use existing Vision 1h kline files instead")
    ap.add_argument("--symbols", default=None, help="comma-separated subset")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--holdout-days", type=int, default=365)
    ap.add_argument("--evaluate-holdout", action="store_true")
    ap.add_argument("--force", action="store_true", help="allow a second look at the holdout")
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("--oi-drop", type=float, default=0.05)
    ap.add_argument("--holds", type=int, nargs="+", default=[4, 12, 24])
    ap.add_argument("--entry-lag", type=int, default=0)
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--funding-n", type=int, default=3)
    ap.add_argument("--min-days", type=int, default=30)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--out", default="reports")
    return ap.parse_args(argv)


def _fmt(df: pd.DataFrame) -> str:
    with pd.option_context("display.width", 250, "display.max_columns", 50,
                           "display.float_format", lambda v: f"{v:,.2f}"):
        return df.to_string(index=False)


def main(argv=None) -> int:
    a = parse_args(argv)
    symbols = [s.strip().upper() for s in a.symbols.split(",")] if a.symbols else None
    close, funding, metrics = load_research_inputs(a.root, "1h", a.klines_dir, symbols)
    if close.empty:
        print("no price data found; run scripts/download_futures_data.py first", file=sys.stderr)
        return 2
    usable = sorted(set(close.columns) & set(funding) & set(metrics))
    if not usable:
        print("no symbol has prices, funding and OI metrics", file=sys.stderr)
        return 2
    close = close[usable]
    p = lf.FlushParams(z_threshold=a.z, oi_drop=a.oi_drop, holds=tuple(a.holds), entry_lag=a.entry_lag,
                       cost_bps=a.cost_bps, funding_n=a.funding_n)
    rules = lf.ProtocolRules(min_days=a.min_days)
    close, oi, fund = lf.prepare_inputs(close, metrics, funding, p)
    win = make_window(close.index[0], close.index[-1], a.holdout_days, a.start, a.evaluate_holdout)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if win.is_holdout:
        try:
            claim_holdout(out, "h5", a.force)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 3

    cov = pd.DataFrame({"oi": oi.notna().mean(), "funding": fund.notna().mean()})
    print(f"H5 liquidation-flush rebound | {len(usable)} symbols | {win.describe()}")
    print(f"data coverage (share of hours): OI median {cov['oi'].median():.0%}, "
          f"funding median {cov['funding'].median():.0%}")
    print(f"params: {p}\n")

    bench = lf.equal_weight_index(close)
    summary, rets = lf.run_study(close, oi, fund, p, win.start, win.end, benchmark=bench, rules=rules)
    if summary.empty:
        print("no events in window")
        return 0
    print(_fmt(summary))
    tag = "holdout" if win.is_holdout else "insample"
    summary.to_csv(out / f"h5_summary_{tag}.csv", index=False)
    rets.to_csv(out / f"h5_events_{tag}.csv", index=False)

    full = summary[summary["condition"] == "full"]
    verdict = "PASS" if full["passes_protocol"].any() else "FAIL"
    best = full.sort_values("t_days", ascending=False).head(1)
    print(f"\nverdict (condition=full, default params): {verdict}")
    if len(best):
        b = best.iloc[0]
        print(f"  strongest hold {int(b.hold_h)}h: net {b.net_bps:.1f} bps, t {b.t_days:.2f}, "
              f"{int(b.event_days)} event days; excess over EW market {b.excess_gross_bps:.1f} bps")
    print("  read 'price' vs 'price_oi' vs 'full': if the filters add nothing, the effect is plain "
          "crash reversal, not liquidations.")

    if a.entry_lag == 0:
        p1 = dataclasses.replace(p, entry_lag=1)
        s1, _ = lf.run_study(close, oi, fund, p1, win.start, win.end, conditions=("full",),
                             benchmark=bench, rules=rules)
        print("\nrobustness: enter one bar later (entry_lag=1), condition=full")
        print(_fmt(s1[["hold_h", "events", "event_days", "net_bps", "t_days", "passes_protocol"]]) if len(s1)
              else "  no events")

    if a.grid:
        rows = []
        for z in (2.5, 3.0, 4.0):
            for drop in (0.03, 0.05, 0.08):
                pg = dataclasses.replace(p, z_threshold=z, oi_drop=drop)
                sg, _ = lf.run_study(close, oi, fund, pg, win.start, win.end, conditions=("full",),
                                     benchmark=bench, rules=rules)
                for r in sg.itertuples(index=False):
                    rows.append({"z": z, "oi_drop": drop, "hold_h": r.hold_h, "event_days": r.event_days,
                                 "net_bps": r.net_bps, "t_days": r.t_days, "pass": r.passes_protocol})
        grid = pd.DataFrame(rows)
        if len(grid):
            print("\nsensitivity grid (condition=full): net bps / t")
            piv = grid.assign(cell=[f"{n:7.1f} / {t:5.2f}" if np.isfinite(t) else f"{n:7.1f} /   nan"
                                    for n, t in zip(grid.net_bps, grid.t_days)])
            print(piv.pivot_table(index=["z", "oi_drop"], columns="hold_h", values="cell", aggfunc="first")
                  .to_string())
            grid.to_csv(out / f"h5_grid_{tag}.csv", index=False)
    print(f"\nwrote {out}/h5_*_{tag}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
