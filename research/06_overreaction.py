"""Do coins over-react (reverse) or keep going (momentum) relative to the market?

    python research/06_overreaction.py
    python research/06_overreaction.py --quantile 0.3 --min-volume 500000

Beta-neutral long/short test on hourly data. For each formation window F (how far back we
look at a coin's move relative to BTC) and holding period H, rank coins every H hours,
go long one extreme and short the other, and measure the next H hours after costs.

The last --holdout-days (default 365) are NOT used for research. Only combinations that
pass all rules on the research period are then checked once on the holdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.analysis.xsection import XSRules, eligibility, evaluate, residual_returns, spread_periods
from quantbot.config import load_config
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe

BTC = "BTC/USD"
FORMATIONS = (4, 12, 24, 72, 168)   # hours of past relative move
HORIZONS = (4, 12, 24, 72)          # hours held
OUT = Path("reports/research")


def hours_txt(h: int) -> str:
    return f"{h}h" if h < 24 else f"{h // 24}d"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-days", type=float, default=365)
    ap.add_argument("--quantile", type=float, default=0.2, help="share of coins in each leg")
    ap.add_argument("--cost-bps", type=float, default=25.0, help="round-trip cost per $ traded")
    ap.add_argument("--min-volume", type=float, default=200_000, help="min trailing median USD volume per hour")
    ap.add_argument("--min-net-bps", type=float, default=5.0, help="required net profit per holding period")
    args = ap.parse_args()
    rules = XSRules(min_net_bps=args.min_net_bps)

    cfg = load_config()
    universe = load_universe(cfg.data.universe_path)
    closes = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", universe, start=cfg.data.start)
    if closes.empty or BTC not in closes:
        raise SystemExit("no hourly data: run  quantbot download")
    volume = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", universe, start=cfg.data.start,
                        field="quote_volume")
    closes = closes.ffill(limit=3)
    end = closes.index.max()
    holdout_start = end - pd.Timedelta(days=args.holdout_days)

    resid, beta = residual_returns(closes, BTC)
    eligible = eligibility(closes, volume, min_usd_per_hour=args.min_volume)
    research_start = resid.index[min(len(resid) - 1, 24 * 45)]  # betas + history need ~1 month
    split = research_start + (holdout_start - research_start) / 2
    print(f"{len(resid.columns)} coins, hourly, research {research_start:%Y-%m-%d} -> {holdout_start:%Y-%m-%d} "
          f"(split {split:%Y-%m-%d}), holdout {holdout_start:%Y-%m-%d} -> {end:%Y-%m-%d} untouched")
    print(f"each leg = {args.quantile:.0%} of eligible coins; eligible = 30+ days of history and "
          f">= ${args.min_volume:,.0f}/hour median volume; cost {args.cost_bps:g} bps per round trip\n")

    rows, periods_cache = [], {}
    for F in FORMATIONS:
        for H in HORIZONS:
            p = spread_periods(resid, beta, eligible, formation=F, horizon=H, quantile=args.quantile,
                               cost_one_way_bps=args.cost_bps / 2, start=research_start, end=holdout_start)
            if p.empty:
                continue
            direction = 1 if p["spread_bps"].mean() > 0 else -1
            st = evaluate(p, direction, split)
            st.update(formation=F, hold=H, direction="momentum" if direction > 0 else "reversal",
                      coins=float(p["coins"].median()), turnover=float(p["turnover"].mean()))
            st["est_14d"] = st["net_bps"] / 1e4 * (14 * 24 / H) * 0.475  # $100k book ~ $47.5k per leg
            st["pass_profit"] = st["net_bps"] >= rules.min_net_bps
            st["pass_confidence"] = st["t"] >= rules.min_t
            st["pass_sample"] = (st["periods"] >= rules.min_periods and st["periods_first"] >= rules.min_periods_per_half
                                 and st["periods_second"] >= rules.min_periods_per_half)
            st["pass_stable"] = st["net_first_half"] > 0 and st["net_second_half"] > 0
            st["tradeable"] = all(st[k] for k in ("pass_profit", "pass_confidence", "pass_sample", "pass_stable"))
            rows.append(st)
            periods_cache[(F, H)] = direction
    grid = pd.DataFrame(rows).sort_values("t", ascending=False)

    show = pd.DataFrame({
        "look back": grid["formation"].map(hours_txt),
        "hold": grid["hold"].map(hours_txt),
        "direction": grid["direction"],
        "gross bps": grid["gross_bps"].map("{:+.1f}".format),
        "cost bps": grid["cost_bps"].map("{:.1f}".format),
        "net bps": grid["net_bps"].map("{:+.1f}".format),
        "t": grid["t"].map("{:+.1f}".format),
        "halves": [f"{a:+.0f} / {b:+.0f}" for a, b in zip(grid["net_first_half"], grid["net_second_half"], strict=True)],
        "~14d on $100k": grid["est_14d"].map("{:+.1%}".format),
        "coins": grid["coins"].map("{:.0f}".format),
    })
    pd.set_option("display.width", 200)
    print("All combinations on the research period (bps are per holding period, per $1 long + $1 short):")
    print(show.to_string(index=False))

    line = "=" * 100
    print(f"\n{line}\nTRADEABLE?  Rank coins by their move vs BTC over the look-back, long one end, short the other,")
    print(f"beta-neutral, rebalance every 'hold' hours. Tested {len(grid)} combinations on the research period.")
    steps = [("pass_profit", f"net profit >= {rules.min_net_bps:g} bps per holding period after costs"),
             ("pass_confidence", f"...and statistically confident (t >= {rules.min_t:g})"),
             ("pass_sample", f"...and enough periods (>= {rules.min_periods}, >= {rules.min_periods_per_half} per half)"),
             ("pass_stable", "...and profitable in BOTH halves of the research period")]
    mask = pd.Series(True, index=grid.index)
    for col, text in steps:
        mask &= grid[col].astype(bool)
        print(f"  {text:<78} {int(mask.sum()):>5}")

    winners = grid[grid["tradeable"]]
    if len(winners):
        print(f"\n{len(winners)} rule(s) pass on the research period. Checking each ONCE on the untouched holdout:")
        for _, w in winners.iterrows():
            F, H = int(w["formation"]), int(w["hold"])
            d = periods_cache[(F, H)]
            ph = spread_periods(resid, beta, eligible, formation=F, horizon=H, quantile=args.quantile,
                                cost_one_way_bps=args.cost_bps / 2, start=holdout_start)
            hmid = holdout_start + (end - holdout_start) / 2
            hs = evaluate(ph, d, hmid) if len(ph) else {"net_bps": np.nan, "t": np.nan, "periods": 0}
            confirmed = hs["net_bps"] > 0 and hs["t"] >= rules.holdout_min_t
            est = hs["net_bps"] / 1e4 * (14 * 24 / H) * 0.475
            print(f"  {w['direction'].upper()}: look back {hours_txt(F)}, hold {hours_txt(H)} -> research net "
                  f"{w['net_bps']:+.1f} bps/period (t {w['t']:.1f}, ~{w['est_14d']:+.1%} per 14d)")
            print(f"     holdout: net {hs['net_bps']:+.1f} bps/period, t {hs['t']:.1f}, {hs['periods']} periods, "
                  f"~{est:+.1%} per 14d  ->  {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}")
        print("\nOnly CONFIRMED rules are candidates for a strategy. Don't re-tune them on the holdout.")
    else:
        print("\nNOTHING QUALIFIES on the research period. The holdout stays untouched.")
        names = {"pass_profit": "profit", "pass_confidence": "confidence", "pass_sample": "periods",
                 "pass_stable": "both halves"}
        print("Closest candidates and the rules they fail:")
        for _, w in grid.sort_values("net_bps", ascending=False).head(5).iterrows():
            fails = ", ".join(n for c, n in names.items() if not w[c])
            print(f"  {w['direction']:>8}: look back {hours_txt(int(w['formation']))}, hold {hours_txt(int(w['hold']))}"
                  f" -> net {w['net_bps']:+.1f} bps/period, t {w['t']:.1f}  (fails: {fails})")
    print(line)

    OUT.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT / "overreaction_grid.csv", index=False)
    print(f"saved {OUT}/overreaction_grid.csv")


if __name__ == "__main__":
    main()
