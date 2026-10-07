"""H6: do coins with crowded (high) funding underperform those with low funding? Beta-neutral, weekly.

    quantbot futures --what funding          # once: download funding rates
    python research/15_funding_tilt.py
    python research/15_funding_tilt.py --window 3D --holds 72,168

Signal: each coin's average funding rate over the trailing window (only settled rates), tested
  raw        - as is
  residual   - after removing what its past 7- and 30-day returns explain across coins (high funding
               mostly follows recent winners, and momentum/reversal were already tested and failed)
Every hold period: rank coins, long one fifth / short the other, beta-neutral via BTC, costs on
turnover. The data picks the direction. Same rules as research/06; the last --holdout-days are
checked only for combinations that pass on the research period.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.analysis.positioning import residualise, trailing_funding
from quantbot.analysis.xsection import XSRules, eligibility, judge, residual_returns, signal_periods
from quantbot.config import load_config
from quantbot.data.futures import load_futures_panel
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe

BTC = "BTC/USD"
OUT = Path("reports/research")


def mean_row_corr(a: pd.DataFrame, b: pd.DataFrame, every: int = 24) -> float:
    """Average correlation across coins between two panels, on every `every`-th row."""
    vals = []
    for i in range(0, len(a), every):
        x, y = a.iloc[i].to_numpy(dtype=float), b.iloc[i].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 5 and x[m].std() > 0 and y[m].std() > 0:
            vals.append(np.corrcoef(x[m], y[m])[0, 1])
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="7D", help="trailing funding window, e.g. 3D, 7D")
    ap.add_argument("--holds", default="72,168", help="holding periods in hours")
    ap.add_argument("--holdout-days", type=float, default=365)
    ap.add_argument("--quantile", type=float, default=0.2)
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--min-volume", type=float, default=200_000)
    args = ap.parse_args()
    rules = XSRules()
    holds = [int(x) for x in args.holds.split(",")]

    cfg = load_config()
    universe = load_universe(cfg.data.universe_path)
    close = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", universe, start=cfg.data.start).ffill(limit=3)
    qv = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", universe, start=cfg.data.start, field="quote_volume")
    funding = load_futures_panel(cfg.data.processed_dir, "fundingRate", "rate")
    if funding.empty:
        raise SystemExit("no funding data: run  quantbot futures --what funding")
    resid, beta = residual_returns(close, BTC)
    coins = [c for c in resid.columns if c in funding.columns]
    resid, beta = resid[coins], beta[coins]
    eligible = eligibility(close, qv)[coins]
    f = trailing_funding(funding[coins], close.index, args.window)
    logp = np.log(close[coins])
    ret7, ret30 = logp - logp.shift(168), logp - logp.shift(720)
    signals = {"raw funding": f, "residual funding": residualise(f, [ret7, ret30], eligible)}

    end = close.index.max()
    start = max(funding.dropna(how="all").index.min(), close.index.min()) + pd.Timedelta(days=40)
    holdout = end - pd.Timedelta(days=args.holdout_days)
    split = start + (holdout - start) / 2
    print(f"{len(coins)} coins with funding data; research {start:%Y-%m-%d} -> {holdout:%Y-%m-%d}, holdout -> "
          f"{end:%Y-%m-%d} (untouched); funding averaged over {args.window}; cost {args.cost_bps:g} bps round trip")
    corr = mean_row_corr(f.where(eligible), ret7.where(eligible))
    print(f"average cross-coin correlation of funding with past 7-day return: {corr:+.2f} "
          "(the momentum overlap that residualising removes)\n")

    rows = []
    for name, sig in signals.items():
        for H in holds:
            p = signal_periods(sig, resid, beta, eligible, horizon=H, quantile=args.quantile,
                               cost_one_way_bps=args.cost_bps / 2, start=start, end=holdout)
            if p.empty:
                continue
            d = 1 if p["spread_bps"].mean() > 0 else -1
            rows.append({"signal": name, "hold": H, "direction": d, **judge(p, d, split, rules, 14 * 24 / H)})
    grid = pd.DataFrame(rows)
    show = pd.DataFrame({
        "signal": grid["signal"], "hold": grid["hold"].map(lambda h: f"{h // 24}d" if h % 24 == 0 else f"{h}h"),
        "trade": np.where(grid["direction"] > 0, "long HIGH funding", "long LOW funding"),
        "IC": grid["ic"].map("{:+.3f}".format), "IC t": grid["ic_t"].map("{:+.1f}".format),
        "gross bps": grid["gross_bps"].map("{:+.1f}".format), "cost bps": grid["cost_bps"].map("{:.1f}".format),
        "net bps": grid["net_bps"].map("{:+.1f}".format), "t": grid["t"].map("{:+.1f}".format),
        "halves": [f"{a:+.0f} / {b:+.0f}" for a, b in zip(grid["net_first_half"], grid["net_second_half"], strict=True)],
        "periods": grid["periods"], "~14d on $100k": grid["est_14d"].map("{:+.1%}".format)})
    pd.set_option("display.width", 200)
    print(show.to_string(index=False))
    print("The hypothesis predicts 'long LOW funding' (crowded coins underperform). bps are per holding period,\n"
          "per $1 long + $1 short.")

    line = "=" * 100
    print(f"\n{line}\nVERDICT (rules: net >= {rules.min_net_bps:g} bps/period, t >= {rules.min_t:g}, enough periods, "
          "both halves positive)")
    winners = grid[grid["tradeable"]]
    if winners.empty:
        best = grid.sort_values("t", ascending=False).iloc[0]
        print(f"  NOTHING QUALIFIES. Closest: {best['signal']}, hold {best['hold']}h: net {best['net_bps']:+.1f} bps, "
              f"t {best['t']:.1f}. The holdout stays untouched.")
    for _, w in winners.iterrows():
        p = signal_periods(signals[w["signal"]], resid, beta, eligible, horizon=int(w["hold"]), quantile=args.quantile,
                           cost_one_way_bps=args.cost_bps / 2, start=holdout, end=end)
        h = judge(p, int(w["direction"]), holdout + (end - holdout) / 2, rules, 14 * 24 / w["hold"]) if len(p) else {}
        ok = bool(h) and h["net_bps"] > 0 and h["t"] >= rules.holdout_min_t
        print(f"  {w['signal']}, hold {w['hold']}h: research net {w['net_bps']:+.1f} bps (t {w['t']:.1f}) | holdout net "
              f"{h.get('net_bps', np.nan):+.1f} bps (t {h.get('t', np.nan):.1f}) -> {'CONFIRMED' if ok else 'NOT CONFIRMED'}")
    print(line)
    OUT.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT / "funding_tilt.csv", index=False)


if __name__ == "__main__":
    main()
