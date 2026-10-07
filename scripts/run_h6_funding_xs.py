#!/usr/bin/env python3
"""H6: cross-sectional funding tilt (long low funding, short high funding, beta-neutral).

Prerequisite: python scripts/download_futures_data.py ... (funding + klines)

    python scripts/run_h6_funding_xs.py                         # in-sample, weekly + 3-day rebalances
    python scripts/run_h6_funding_xs.py --rebalance W-MON       # one schedule
    python scripts/run_h6_funding_xs.py --evaluate-holdout      # ONCE, after freezing params

Decision rule (pre-registered): residualised signal, weekly rebalance, must
pass the protocol (net >= 5 bps per period, t >= 3, >= 30 periods, positive
in both halves).  The raw (non-residualised) variant shows how much of any
result is just momentum/reversal in disguise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402

from quantbot.analysis import funding_xs as fx  # noqa: E402
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
    ap.add_argument("--window-days", type=int, default=7)
    ap.add_argument("--rebalance", nargs="+", default=["W-MON", "3D"])
    ap.add_argument("--quantile", type=float, default=0.2)
    ap.add_argument("--min-coins", type=int, default=10)
    ap.add_argument("--beta-days", type=int, default=60)
    ap.add_argument("--cost-bps-one-way", type=float, default=12.5)
    ap.add_argument("--no-beta-neutral", action="store_true")
    ap.add_argument("--min-periods", type=int, default=30)
    ap.add_argument("--out", default="reports")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    symbols = [s.strip().upper() for s in a.symbols.split(",")] if a.symbols else None
    close, funding, _ = load_research_inputs(a.root, "1h", a.klines_dir, symbols)
    usable = sorted(set(close.columns) & set(funding))
    if not usable:
        print("no symbol has both prices and funding; run scripts/download_futures_data.py first",
              file=sys.stderr)
        return 2
    daily = fx.daily_closes(close[usable])
    if daily.empty:
        print("no 00:00 UTC closes found", file=sys.stderr)
        return 2
    win = make_window(daily.index[0], daily.index[-1], a.holdout_days, a.start, a.evaluate_holdout)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if win.is_holdout:
        try:
            claim_holdout(out, "h6", a.force)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 3
    rules = fx.ProtocolRules(min_periods=a.min_periods)
    fund_daily = fx.trailing_funding(funding, daily.index, a.window_days)
    print(f"H6 cross-sectional funding | {len(usable)} symbols | {win.describe()}")
    print(f"coins with a funding signal per day: median {int(fund_daily.loc[win.start:win.end].notna().sum(axis=1).median())}\n")

    rows, tag = [], ("holdout" if win.is_holdout else "insample")
    for rb in a.rebalance:
        for resid in (True, False):
            p = fx.FundingXSParams(window_days=a.window_days, rebalance=rb, quantile=a.quantile,
                                   min_coins=a.min_coins, beta_days=a.beta_days, residualize=resid,
                                   beta_neutral=not a.no_beta_neutral, one_way_cost_bps=a.cost_bps_one_way)
            periods = fx.run_backtest(daily, fund_daily, p, win.start, win.end)
            name = f"{rb}_{'resid' if resid else 'raw'}"
            if not periods.empty:
                periods.to_csv(out / f"h6_periods_{name}_{tag}.csv", index=False)
            rows.append({"rebalance": rb, "signal": "residual" if resid else "raw",
                         **fx.summarize_periods(periods, rules)})
    summary = pd.DataFrame(rows)
    cols = ["rebalance", "signal", "periods", "per_leg", "mean_ic", "t_ic", "gross_bps", "cost_bps", "net_bps",
            "t_net", "net_bps_1st_half", "net_bps_2nd_half", "hit_rate", "ann_sharpe", "corr_funding_mom",
            "passes_protocol"]
    with pd.option_context("display.width", 250, "display.max_columns", 50,
                           "display.float_format", lambda v: f"{v:,.2f}"):
        print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))
    summary.to_csv(out / f"h6_summary_{tag}.csv", index=False)
    primary = summary[(summary["rebalance"] == a.rebalance[0]) & (summary["signal"] == "residual")]
    verdict = "PASS" if len(primary) and bool(primary["passes_protocol"].iloc[0]) else "FAIL"
    print(f"\nverdict (residual signal, {a.rebalance[0]}): {verdict}")
    print("  mean_ic > 0 means low-funding coins beat high-funding coins (the hypothesis).")
    print("  corr_funding_mom = rank corr of raw funding with 7-day return; if raw passes but residual "
          "fails, the 'edge' is momentum/reversal you already rejected.")
    print(f"wrote {out}/h6_*_{tag}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
