"""Lead-lag between BTC and tokenized gold (PAXG), in both directions, before and after a regime split.

    python research/23_paxg_btc_leadlag.py                       # split at 2026-01-01
    python research/23_paxg_btc_leadlag.py --split 2026-04-01 --a PAXG/USD --b BTC/USD

Uses CLOSE prices (VWAP averages overlap in time and create a fake lag).
  Part A, intraday (1-minute data: 1m, 5m, 15m, 1h) and Part B, long history (hourly data: 1h, 4h, 1d):
    cross-correlation of A's return with B's return k bars earlier (k > 0: B leads, k < 0: A leads),
    regression t-statistics for each direction, and how far A follows after B's biggest 1% of moves.
  Then the five-rule TRADEABLE? check (as research/05) for "after a big B move, trade A in B's direction",
  per regime. A stale A price (no trades) looks exactly like a lag: the liquidity rule uses the share of
  unchanged minutes (intraday part) or unchanged hours (long-history part).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
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

OUT = Path("reports/research")
MINUTES = {"1m": 1, "5min": 5, "15min": 15, "1h": 60, "4h": 240, "1D": 1440}
LABEL = {"1m": "1m", "5min": "5m", "15min": "15m", "1h": "1h", "4h": "4h", "1D": "1d"}


def analyse(closes: pd.DataFrame, a: str, b: str, freqs: list[str], max_lag: int, stale: pd.Series,
            rules: TradeRules) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    rows, ccfs, grids = [], {}, []
    for f in freqs:
        r = to_returns(closes[[a, b]], f).dropna(how="all")
        if r[[a, b]].dropna().shape[0] < 50:
            continue
        cc = cross_correlations(r, b, max_lag).loc[a]                     # corr(A[t], B[t - lag])
        ccfs[LABEL[f]] = cc
        fwd = lead_regression(r, b).loc[a]                               # B[t-1] -> A[t]
        rev = lead_regression(r, a).loc[b]                               # A[t-1] -> B[t]
        cu1, cu3 = catch_up(r, b, 0.99, 1), catch_up(r, b, 0.99, 3)
        rows.append({"bar": LABEL[f], "bars": int(r[[a, b]].dropna().shape[0]), "same bar": cc.get(0, np.nan),
                     "B leads by 1": cc.get(1, np.nan), "B leads by 2": cc.get(2, np.nan),
                     "A leads by 1": cc.get(-1, np.nan), "t (B->A)": fwd["lead_t"], "t (A->B)": rev["lead_t"],
                     "A follows next bar (bps)": cu1["catchup_bps"].get(a, np.nan) if len(cu1) else np.nan,
                     "next 3 bars (bps)": cu3["catchup_bps"].get(a, np.nan) if len(cu3) else np.nan})
        lookback = max(int(30 * 1440 / MINUTES[f]), 20)
        split = r.index.min() + (r.index.max() - r.index.min()) / 2
        g = tradability(r[[a, b]], b, freq_label=LABEL[f], horizons=(1, 3, 6), quantiles=(0.95, 0.99),
                        lookback_bars=lookback, split=split, stale=stale, rules=rules)
        grids.append(g)
    return pd.DataFrame(rows), ccfs, (pd.concat(grids, ignore_index=True) if grids else pd.DataFrame())


def show(t: pd.DataFrame, a: str, b: str) -> str:
    if t.empty:
        return "  (not enough data)"
    fmt = t.copy()
    for c in ("same bar", "B leads by 1", "B leads by 2", "A leads by 1"):
        fmt[c] = t[c].map("{:+.3f}".format)
    for c in ("t (B->A)", "t (A->B)"):
        fmt[c] = t[c].map("{:+.1f}".format)
    for c in ("A follows next bar (bps)", "next 3 bars (bps)"):
        fmt[c] = t[c].map(lambda v: f"{v:+.1f}" if np.isfinite(v) else "n/a")
    fmt.columns = [c.replace("A ", f"{a} ").replace("B ", f"{b} ").replace("(B->A)", f"({b}->{a})")
                   .replace("(A->B)", f"({a}->{b})") for c in fmt.columns]
    return "  " + fmt.to_string(index=False).replace("\n", "\n  ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="PAXG/USD", help="the follower being tested (default PAXG)")
    ap.add_argument("--b", default="BTC/USD", help="the possible leader (default BTC)")
    ap.add_argument("--split", default="2026-01-01", help="regime boundary (results before / after)")
    ap.add_argument("--max-lag", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    rules = TradeRules(cost_bps=args.cost_bps)

    cfg = load_config()
    universe = load_universe(cfg.data.universe_path)
    pair = {p: universe[p] for p in (args.a, args.b) if p in universe}
    if len(pair) < 2:
        raise SystemExit(f"both {args.a} and {args.b} must be in {cfg.data.universe_path}")
    A, B = args.a.split("/")[0], args.b.split("/")[0]
    split = pd.Timestamp(args.split, tz="UTC")
    pd.set_option("display.width", 230)

    sections = []
    m1 = load_panel(cfg.data.processed_dir, cfg.data.market, "1m", pair)
    if not m1.empty and m1[[args.a, args.b]].dropna().shape[0] > 1000:
        m1 = m1[[args.a, args.b]].dropna(how="all").ffill(limit=60)
        sections.append(("A. INTRADAY (1-minute data)", m1, ["1m", "5min", "15min", "1h"], stale_share(m1)))
    h1 = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", pair, start=cfg.data.start)
    if not h1.empty:
        h1 = h1[[args.a, args.b]].dropna(how="all").ffill(limit=3)
        sections.append(("B. LONG HISTORY (hourly data)", h1, ["1h", "4h", "1D"], stale_share(h1)))  # stale HOURS
    if not sections:
        raise SystemExit("no data: run  quantbot download  (and --interval 1m for the intraday part)")

    print(f"Does {B} lead {A} (or the reverse)? Close prices; regime split at {split:%Y-%m-%d}; cost {args.cost_bps:g} bps "
          "per round trip.")
    if sections[0][0].startswith("A."):
        st = sections[0][3]
        print(f"{A}'s price did not change in {st[args.a]:.1%} of minutes ({B}: {st[args.b]:.1%}). Unchanged minutes "
              f"make {A} look like it lags: the move shows up when it next trades.")
    print("Columns: correlation of the two returns in the same bar; with B k bars EARLIER (B leads) and with A one bar\n"
          f"earlier (A leads); regression t-statistics (|t| > 3: real); how far {A} moves in {B}'s direction after\n"
          f"{B}'s biggest 1% of moves.\n")

    curves, verdicts = {}, []
    for title, data, freqs, stale in sections:
        for regime, part in (("before", data.loc[data.index < split]), ("since", data.loc[data.index >= split])):
            if part.shape[0] < 100:
                continue
            label = f"{title.split(' (')[0]}, {regime} {split:%Y-%m-%d} ({part.index.min():%Y-%m-%d} -> {part.index.max():%Y-%m-%d})"
            t, ccf, grid = analyse(part, args.a, args.b, freqs, args.max_lag, stale, rules)
            print(label)
            print(show(t, A, B), "\n")
            curves[label] = ccf
            if not grid.empty:
                win = grid[grid["tradeable"]]
                best = grid.sort_values("net_bps", ascending=False).iloc[0]
                verdicts.append((label, len(grid), win, best))

    line = "=" * 100
    print(f"{line}\nTRADEABLE? rule: after a big {B} move, trade {A} in {B}'s direction; must clear {args.cost_bps:g} bps with\n"
          f"net >= {rules.min_net_bps:g} bps, t >= {rules.min_t:g}, enough trades, both halves positive, and {A} trading in "
          f">= {1 - rules.max_stale:.0%} of minutes.")
    for label, n, win, best in verdicts:
        if len(win):
            for _, w in win.iterrows():
                print(f"  {label}: PASSES on {w['freq']} bars, {w['btc_move']} {B} move, hold {w['hold_bars']}: net "
                      f"{w['net_bps']:+.1f} bps, t {w['t']:.1f}, {w['trades_per_year']:.0f} trades/yr")
        else:
            fails = [k for k in ("pass_profit", "pass_confidence", "pass_sample", "pass_stable", "pass_liquid") if not best[k]]
            print(f"  {label}: nothing of {n} combinations passes; best net {best['net_bps']:+.1f} bps "
                  f"({best['freq']}, hold {best['hold_bars']}), fails: {', '.join(f.replace('pass_', '') for f in fails)}")
    print("'n/a' in the follow columns = fewer than 20 big moves in that regime at that bar size.")
    print(line)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    groups = sorted({lab.split(",")[0] for lab in curves})
    fig, axes = plt.subplots(len(groups), 1, figsize=(13, 4.2 * len(groups)), squeeze=False)
    for ax, g in zip(axes[:, 0], groups, strict=True):
        regimes = [lab for lab in curves if lab.startswith(g)]
        freqs = sorted({f for lab in regimes for f in curves[lab]}, key=lambda f: list(LABEL.values()).index(f))
        width = 0.8 / max(1, len(regimes) * len(freqs))
        lags = np.arange(-args.max_lag, args.max_lag + 1)
        k = 0
        for lab in regimes:
            for f in freqs:
                if f in curves[lab]:
                    ax.bar(lags + (k - len(regimes) * len(freqs) / 2) * width, curves[lab][f].reindex(lags).to_numpy(),
                           width=width, label=f"{f}, {lab.split(', ')[1].split(' (')[0]}")
                    k += 1
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5, ls=":")
        ax.set_xticks(lags)
        ax.set_xlabel(f"lag in bars   (left: {A} earlier, {A} leads   |   right: {B} earlier, {B} leads)")
        ax.set_ylabel(f"corr({A} now, {B} at lag)")
        ax.set_title(g)
        ax.legend(fontsize=7, ncol=4)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{A}_{B}_leadlag.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"chart: {png}")
    if sys.platform == "darwin" and not args.no_open:
        subprocess.run(["open", str(png)], check=False)


if __name__ == "__main__":
    main()
