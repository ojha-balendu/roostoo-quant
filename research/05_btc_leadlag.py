"""Does BTC lead the other coins? Measured on 1m, 5m, 15m and 1h returns.

    python research/05_btc_leadlag.py                 # last 365 days of 1m data
    python research/05_btc_leadlag.py --days 90

For each frequency:
  * cross-correlation of each coin's return with BTC's return k bars earlier (and the reverse)
  * regression  r_coin[t] = a + b * r_btc[t-1] + c * r_coin[t-1]   (robust t-stats)
  * catch-up after BTC's biggest 5% of moves: how far the coin follows next, in bps,
    versus the ~25 bps a round trip costs on Roostoo
  * stability across the two halves of the period
  * finally a TRADEABLE? verdict: every coin x bar size x BTC-move size x holding period
    is checked against five rules (profit after costs, confidence, enough trades,
    profitable in both halves, liquid), and passing rules are printed in plain English
Data is Binance's; Roostoo's own price feed may add (or remove) lag on top of this.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.analysis.leadlag import (
    TradeRules,
    catch_up,
    cross_correlations,
    lead_regression,
    stale_share,
    to_returns,
    tradability,
)
from quantbot.config import load_config
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe

BTC = "BTC/USD"
FREQS = ["1m", "5min", "15min", "1h"]
LABEL = {"1m": "1m", "5min": "5m", "15min": "15m", "1h": "1h"}
MINUTES = {"1m": 1, "5min": 5, "15min": 15, "1h": 60}
HORIZONS = (1, 3, 6)          # holding periods tested, in bars
QUANTILES = (0.95, 0.99)      # "big BTC move" = top 5% / top 1% of the previous 30 days
OUT = Path("reports/research")
ROUND_TRIP_BPS = 25  # 2 x 0.1% fee + ~5 bps of spread


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=365)
    ap.add_argument("--max-lag", type=int, default=5)
    ap.add_argument("--quantile", type=float, default=0.95, help="'big BTC move' threshold")
    ap.add_argument("--cost-bps", type=float, default=25.0, help="round-trip cost per trade")
    ap.add_argument("--min-net-bps", type=float, default=5.0, help="required profit per trade after cost")
    args = ap.parse_args()
    rules = TradeRules(cost_bps=args.cost_bps, min_net_bps=args.min_net_bps)

    cfg = load_config()
    universe = load_universe(cfg.data.universe_path)
    probe = load_panel(cfg.data.processed_dir, cfg.data.market, "1m", {BTC: universe[BTC]})
    if probe.empty:
        raise SystemExit("no 1m data: run  quantbot download --interval 1m --start <date>")
    start = probe.index.max() - pd.Timedelta(days=args.days)
    closes = load_panel(cfg.data.processed_dir, cfg.data.market, "1m", universe, start=str(start.date()))
    closes = closes.astype("float32").ffill(limit=5)
    print(f"{closes.shape[1] - 1} coins vs BTC, 1m data {closes.index.min():%Y-%m-%d} -> {closes.index.max():%Y-%m-%d}")

    stale = stale_share(closes).drop(BTC)
    per_coin, summary, ccf_median, halves, grids = {}, {}, {}, {}, []
    mid = closes.index.min() + (closes.index.max() - closes.index.min()) / 2
    for freq in FREQS:
        lab = LABEL[freq]
        r = to_returns(closes, freq).astype("float64")
        cc = cross_correlations(r, BTC, args.max_lag)
        reg = lead_regression(r, BTC)
        cu1 = catch_up(r, BTC, args.quantile, horizon=1)
        cu3 = catch_up(r, BTC, args.quantile, horizon=3)
        ccf_median[lab] = cc.median()
        first = cross_correlations(r.loc[:mid], BTC, 1)[1]
        second = cross_correlations(r.loc[mid:], BTC, 1)[1]
        halves[lab] = (first.median(), second.median(), float(np.corrcoef(first.align(second, join="inner"))[0, 1]))
        summary[lab] = {
            "corr same bar": cc[0].median(),
            "corr BTC 1 bar earlier": cc[1].median(),
            "corr BTC 2 bars earlier": cc[2].median(),
            "corr coin leads BTC by 1": cc[-1].median(),
            "lead coef b (median)": reg["lead_b"].median(),
            "coins with lead t > 3": f"{(reg['lead_t'] > 3).sum()}/{len(reg)}",
            "catch-up next bar (bps)": cu1["catchup_bps"].median(),
            "catch-up next 3 bars (bps)": cu3["catchup_bps"].median(),
            "follow rate next bar": cu1["hit_rate"].median(),
            "coins with catch-up > cost": f"{(cu1['catchup_bps'] > ROUND_TRIP_BPS).sum()}/{len(cu1)}",
        }
        per_coin[lab] = pd.DataFrame({"lag1_corr": cc[1], "lead_t": reg["lead_t"], "catchup_bps": cu1["catchup_bps"]})
        grids.append(tradability(r, BTC, freq_label=lab, horizons=HORIZONS, quantiles=QUANTILES,
                                 lookback_bars=int(30 * 24 * 60 / MINUTES[freq]), split=mid, stale=stale,
                                 rules=rules))
        print(f"  measured {lab}")

    pd.set_option("display.width", 200)
    s = pd.DataFrame(summary)
    for row in s.index:
        if row.startswith("corr") or row.startswith("lead coef"):
            s.loc[row] = s.loc[row].map(lambda v: f"{v:+.3f}")
        elif row.startswith("catch-up"):
            s.loc[row] = s.loc[row].map(lambda v: f"{v:+.1f}")
        elif row.startswith("follow"):
            s.loc[row] = s.loc[row].map(lambda v: f"{v:.0%}")
    print(f"\nMedian across coins (round trip on Roostoo costs ~{ROUND_TRIP_BPS} bps):")
    print(s.to_string())

    print("\nStability: median 'BTC 1 bar earlier' correlation in each half of the period,")
    print("and how similar the coin-by-coin pattern is between halves (1 = identical ranking):")
    for lab, (a, b, consistency) in halves.items():
        print(f"  {lab:>4}: first half {a:+.3f}, second half {b:+.3f}, coin pattern correlation {consistency:+.2f}")

    table = pd.concat(per_coin, axis=1)
    table[("1m", "stale_%")] = stale * 100
    table = table.sort_values(("1m", "lag1_corr"), ascending=False)
    fmt = table.copy()
    patterns = {"lag1_corr": "{:+.3f}", "lead_t": "{:+.1f}", "catchup_bps": "{:+.1f}", "stale_%": "{:.1f}"}
    for col in fmt.columns:
        fmt[col] = table[col].map(patterns[col[1]].format)
    fmt.index = [p.split("/")[0] for p in fmt.index]
    print("\nPer coin (sorted by 1m lag-1 correlation; catch-up in bps after BTC's biggest moves; "
          "stale_% = minutes with no price change):")
    print(fmt.to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    grid = pd.concat(grids, ignore_index=True)
    grid.to_csv(OUT / "btc_leadlag_tradability.csv", index=False)
    table.to_csv(OUT / "btc_leadlag_per_coin.csv")
    pd.DataFrame(ccf_median).to_csv(OUT / "btc_leadlag_ccf.csv")
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        for lab, ccf in ccf_median.items():
            ax.plot(ccf.index, ccf.to_numpy(), marker="o", label=lab)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("lag in bars (positive = BTC earlier)")
        ax.set_ylabel("median correlation across coins")
        ax.set_yscale("symlog", linthresh=0.01)
        ax.legend()
        ax.set_title("Cross-correlation of coin returns with BTC returns")
        fig.tight_layout()
        fig.savefig(OUT / "btc_leadlag_ccf.png", dpi=130)
        print(f"\nsaved {OUT}/btc_leadlag_per_coin.csv, btc_leadlag_ccf.csv and .png")
    except ImportError:
        print(f"\nsaved {OUT}/btc_leadlag_per_coin.csv and btc_leadlag_ccf.csv")
    print_verdict(grid, rules)


def _describe(row: pd.Series) -> str:
    minutes = row["hold_bars"] * {"1m": 1, "5m": 5, "15m": 15, "1h": 60}[row["freq"]]
    hold = f"{minutes:g} min" if minutes < 60 else f"{minutes / 60:g} h"
    coin = row["coin"].split("/")[0]
    return (f"{coin:<8} {row['freq']:>3} bars: after a {row['btc_move']} BTC move, trade {coin} in BTC's direction "
            f"for {row['hold_bars']} bar(s) ({hold}).\n"
            f"         net {row['net_bps']:+.1f} bps/trade after costs (gross {row['gross_bps']:+.1f}), t = {row['t']:.1f}, "
            f"{row['trades_per_year']:.0f} trades/yr, beats cost {row['hit_rate']:.0%} of the time, "
            f"halves {row['net_first_half']:+.1f} / {row['net_second_half']:+.1f} bps")


def print_verdict(grid: pd.DataFrame, rules: TradeRules) -> None:
    line = "=" * 100
    print(f"\n{line}\nTRADEABLE?  Rule: after a big BTC move, trade the coin in BTC's direction and hold for N bars.")
    print(f"Tested {len(grid)} combinations (coins x bar sizes {[f for f in LABEL.values() if f in set(grid['freq'])]} x BTC-move sizes "
          f"x holding periods {list(HORIZONS)} bars).")
    steps = [
        ("pass_profit", f"profitable after a {rules.cost_bps:g} bps round trip (net >= {rules.min_net_bps:g} bps)"),
        ("pass_confidence", f"...and statistically confident (t >= {rules.min_t:g})"),
        ("pass_sample", f"...and enough trades (>= {rules.min_events}, >= {rules.min_events_per_half} per half)"),
        ("pass_stable", "...and profitable in BOTH halves of the period"),
        ("pass_liquid", f"...and liquid (price updates in >= {1 - rules.max_stale:.0%} of minutes)"),
    ]
    mask = pd.Series(True, index=grid.index)
    for col, text in steps:
        mask &= grid[col].astype(bool)
        print(f"  {text:<78} {int(mask.sum()):>5}")
    winners = grid[grid["tradeable"]].sort_values("t", ascending=False)
    if len(winners):
        print(f"\n{len(winners)} TRADEABLE rule(s), strongest first:")
        for _, row in winners.iterrows():
            print("  " + _describe(row))
        print("\nCaveats: entry is assumed at the close of BTC's move bar; the live bot polls every ~10 s, "
              "so 1m results are optimistic.\nConfirm on Roostoo's own recorded prices before trading any of these.")
    else:
        print("\nNOTHING QUALIFIES: no coin/lag combination makes money reliably after costs.")
        near = grid[grid["pass_sample"].astype(bool)].sort_values("net_bps", ascending=False).head(8)
        if len(near):
            print("Closest candidates (best net profit per trade) and the rules they fail:")
            names = {"pass_profit": "profit", "pass_confidence": "confidence", "pass_sample": "trades",
                     "pass_stable": "both halves", "pass_liquid": "liquidity"}
            for _, row in near.iterrows():
                fails = ", ".join(n for c, n in names.items() if not row[c])
                print("  " + _describe(row) + f"\n         fails: {fails}")
    print(line)


if __name__ == "__main__":
    main()
