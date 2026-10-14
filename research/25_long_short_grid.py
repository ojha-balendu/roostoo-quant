"""Momentum sleeve with i longs x j shorts (0..8 each) and three ways to weight coins within a leg.

    python research/25_long_short_grid.py                                   # 3 x 80 = 240 backtests
    python research/25_long_short_grid.py --max-long 6 --max-short 6 --schemes equal,inverse_vol,exp_vol

Same ranking as the live strategy (16-day BTC-removed momentum / its volatility, daily at 00:00 UTC,
liquidity filter). For each cell: long the i best, short the j worst; the long leg and the short leg get
equal dollars; inside each leg coins are weighted by
    equal        1 / n
    inverse_vol  (1/sigma_i) / sum(1/sigma)
    mul_vol      sigma_i / sum(sigma)                      (most volatile coin gets the most)
    exp_vol      exp(-sigma_i / sum sigma) / sum_k exp(-sigma_k / sum sigma)   (via --schemes)
i = 0: shorts only (the BTC hedge ends up long); j = 0: longs only (BTC short); 0 x 0 is skipped.
(sigma = 16-day volatility after removing BTC's influence); BTC hedge to zero beta; 95% gross; the live
bot's 25% cap per asset (hedge included); rank buffer 2i for longs, 2j for shorts. Cell (5,5) with
"equal" is the live strategy.
Reported per weighting as matrices of the competition SCORE (0.4 Sortino + 0.3 Sharpe + 0.3 Calmar), the
SHARPE ratio and the net RETURN (after fees): for the last 30 days, and the median over every 14-day window
since the data starts (each a fresh competition). Charts: one image each for score, Sharpe and return.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.analysis.xsection import eligibility, residual_returns
from quantbot.backtest.composite_bt import CompositeConfig, run_metrics, simulate
from quantbot.backtest.momentum import MomentumConfig, momentum_inputs, sleeve_weights
from quantbot.config import load_config
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe
from quantbot.rl.sim import simple_returns

BTC = "BTC/USD"
OUT = Path("reports/research")
LIVE = MomentumConfig(residual_riskadj=True, buffer_k=10)          # what config/competition.yaml runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-long", type=int, default=8)
    ap.add_argument("--max-short", type=int, default=8)
    ap.add_argument("--min-long", type=int, default=0)
    ap.add_argument("--min-short", type=int, default=0)
    ap.add_argument("--schemes", default="equal,inverse_vol,mul_vol")
    ap.add_argument("--last-days", type=float, default=30)
    ap.add_argument("--window-days", type=float, default=14)
    ap.add_argument("--cap", type=float, default=0.25, help="per-asset cap, as the live risk layer")
    ap.add_argument("--capital", type=float, default=100_000)
    ap.add_argument("--config", default="config/competition.yaml")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    schemes = args.schemes.split(",")
    longs, shorts = range(args.min_long, args.max_long + 1), range(args.min_short, args.max_short + 1)
    cells = [(i, j) for i in longs for j in shorts if i + j > 0]

    cfg = load_config(args.config)
    universe = load_universe(cfg.data.universe_path)
    close = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", universe, start=cfg.data.start).ffill(limit=3)
    qv = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", universe, start=cfg.data.start, field="quote_volume")
    if close.empty or BTC not in close:
        raise SystemExit("no hourly data: run  quantbot download")
    resid, beta = residual_returns(close, BTC)
    coins = list(resid.columns)
    elig = eligibility(close, qv)[coins]
    x = {k: v.reindex(columns=coins).to_numpy(dtype=float) for k, v in momentum_inputs(close, resid, elig, LIVE).items()}
    idx = close.index
    P = close[coins + [BTC]].to_numpy(dtype=float)
    R, TR = simple_returns(P), np.isfinite(P)
    B, E = beta[coins].to_numpy(), elig.to_numpy(dtype=bool)
    daily = np.asarray((idx + pd.Timedelta(hours=1)).hour == 0)
    bc = CompositeConfig(capital=args.capital, fee=cfg.fees.taker, spread_bps=cfg.backtest.spread_bps)
    sim_cfg = bc.sim_config()
    end = len(idx) - 1
    t0 = int(idx.searchsorted(idx.min() + pd.Timedelta(days=45)))
    step = int(args.window_days * 24)
    starts = list(range(t0, end - step + 1, step))
    last_a = int(idx.searchsorted(idx[end] - pd.Timedelta(days=args.last_days)))
    n_runs = len(schemes) * len(cells)
    print(f"{len(coins)} coins; {n_runs} configurations; last {args.last_days:g} days "
          f"({idx[last_a]:%Y-%m-%d} -> {idx[end]:%Y-%m-%d}) and {len(starts)} windows of {args.window_days:g} days "
          f"since {idx[t0]:%Y-%m-%d}; ${args.capital:,.0f}, per-asset cap {args.cap:.0%}\n", flush=True)

    def decider(mc: MomentumConfig, first: int):
        def decide(t, cur):
            if not (daily[t] or t == first):
                return cur
            w = sleeve_weights(t, cur, x, B, E, R, mc)
            w[-1] = float(np.clip(w[-1], -args.cap, args.cap))            # the live risk layer caps the hedge too
            return w
        return decide

    rows = []
    for scheme in schemes:
        for i in longs:
            for j in (j for j in shorts if (i, j) in cells):
                mc = replace(LIVE, k=i, k_short=j, buffer_k=2 * i, buffer_short=2 * j, weighting=scheme,
                             max_weight=args.cap)
                last = run_metrics(simulate(P, idx, decider(mc, last_a), last_a, end, sim_cfg, R, TR), args.capital)
                wins = [run_metrics(simulate(P, idx, decider(mc, s), s, s + step, sim_cfg, R, TR), args.capital)
                        for s in starts]
                ws = pd.DataFrame(wins)
                rows.append({"scheme": scheme, "longs": i, "shorts": j, "last_score": last["score"],
                             "last_return": last["return"], "last_sharpe": last["sharpe"], "last_max_dd": last["max_dd"],
                             "w_median_score": ws["score"].median(), "w_median_return": ws["return"].median(),
                             "w_median_sharpe": ws["sharpe"].median(),
                             "w_positive": (ws["return"] > 0).mean(), "w_worst_dd": ws["max_dd"].min()})
            print(f"  {scheme}: {i} long(s) done", flush=True)
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "long_short_grid.csv", index=False)

    def matrix(scheme: str, col: str, fmt: str) -> pd.DataFrame:
        m = t[t["scheme"] == scheme].pivot(index="longs", columns="shorts", values=col)
        m.index = [f"{i} long" for i in m.index]
        m.columns = [f"{j} short" for j in m.columns]
        return m.map(lambda v: fmt.format(v) if np.isfinite(v) else "n/a")

    pd.set_option("display.width", 200)
    for scheme in schemes:
        print(f"\n{'=' * 100}\nWEIGHTING: {scheme}")
        for col, title, fmt in (
                ("last_score", f"SCORE, last {args.last_days:g} days", "{:+.2f}"),
                ("last_sharpe", f"SHARPE, last {args.last_days:g} days", "{:+.2f}"),
                ("last_return", f"RETURN (net of fees), last {args.last_days:g} days", "{:+.1%}"),
                ("w_median_score", f"MEDIAN SCORE over {len(starts)} windows of {args.window_days:g} days", "{:+.2f}"),
                ("w_median_sharpe", f"MEDIAN SHARPE over {len(starts)} windows of {args.window_days:g} days", "{:+.2f}"),
                ("w_median_return", f"MEDIAN RETURN (net) per {args.window_days:g}-day window", "{:+.2%}")):
            print(f"\n  {title}:")
            print("  " + matrix(scheme, col, fmt).to_string().replace("\n", "\n  "))

    print(f"\n{'=' * 100}\nCOMPARISON OF THE WEIGHTINGS (averages over all {len(cells)} cells):")
    best_by_cell = t.loc[t.groupby(["longs", "shorts"])["w_median_score"].idxmax(), "scheme"].value_counts()
    summary = t.groupby("scheme").agg(last_score=("last_score", "mean"), last_return=("last_return", "mean"),
                                      w_score=("w_median_score", "mean"), w_positive=("w_positive", "mean"),
                                      w_worst_dd=("w_worst_dd", "mean")).reindex(schemes)
    summary["cells where best (14-day)"] = best_by_cell.reindex(schemes).fillna(0).astype(int)
    print(pd.DataFrame({
        "avg score, last 30d": summary["last_score"].map("{:+.2f}".format),
        "avg return, last 30d": summary["last_return"].map("{:+.2%}".format),
        "avg median 14d score": summary["w_score"].map("{:+.2f}".format),
        "avg % windows positive": summary["w_positive"].map("{:.0%}".format),
        "avg worst 14d drawdown": summary["w_worst_dd"].map("{:.1%}".format),
        "cells where best (14d)": summary["cells where best (14-day)"]}).to_string())
    live = t[(t["scheme"] == "equal") & (t["longs"] == 5) & (t["shorts"] == 5)]
    if len(live):
        lv = live.iloc[0]
        print(f"\nLive strategy (5 long / 5 short, equal): last 30d score {lv['last_score']:+.2f} "
              f"({lv['last_return']:+.1%}), median 14-day score {lv['w_median_score']:+.2f}")
    top = t.sort_values("w_median_score", ascending=False).head(5)
    print("Top 5 cells by median 14-day score (how they did in the last 30 days):")
    for _, r in top.iterrows():
        print(f"  {r['scheme']:<12} {r['longs']} long / {r['shorts']} short: median 14d score {r['w_median_score']:+.2f}, "
              f"{r['w_positive']:.0%} windows positive, worst 14d DD {r['w_worst_dd']:.1%} | last 30d score "
              f"{r['last_score']:+.2f} ({r['last_return']:+.1%})")
    print(f"\nWith {n_runs} configurations, the best cell in the last-30-days matrix is partly luck. Trust a change only if\n"
          "the 14-day median matrix agrees and the neighbouring cells are also good.")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    pngs = []
    for metric, (last_col, win_col, label, pct) in {
            "score": ("last_score", "w_median_score", "score", False),
            "sharpe": ("last_sharpe", "w_median_sharpe", "Sharpe", False),
            "return": ("last_return", "w_median_return", "net return", True)}.items():
        fig, axes = plt.subplots(2, len(schemes), figsize=(5.4 * len(schemes), 10), squeeze=False)
        for c, scheme in enumerate(schemes):
            for r, (col, title) in enumerate(((last_col, f"{label}, last {args.last_days:g} days"),
                                              (win_col, f"median {label}, {len(starts)} x {args.window_days:g}-day windows"))):
                m = t[t["scheme"] == scheme].pivot(index="longs", columns="shorts", values=col)
                vals = t[col].replace([np.inf, -np.inf], np.nan).abs()
                lim = float(np.nanpercentile(vals, 95)) if vals.notna().any() else 1.0
                lim = lim or 1.0
                ax = axes[r, c]
                im = ax.imshow(m.to_numpy(dtype=float), cmap="RdYlGn", vmin=-lim, vmax=lim, origin="upper")
                ax.set_xticks(range(len(m.columns)), m.columns)
                ax.set_yticks(range(len(m.index)), m.index)
                ax.set_xlabel("shorts")
                ax.set_ylabel("longs")
                ax.set_title(f"{scheme}: {title}", fontsize=9)
                for (a, b), v in np.ndenumerate(m.to_numpy(dtype=float)):
                    if np.isfinite(v):
                        ax.text(b, a, f"{v * 100:+.1f}%" if pct else f"{v:+.1f}", ha="center", va="center", fontsize=6)
                if scheme == "equal" and 5 in m.index and 5 in m.columns:
                    ax.add_patch(plt.Rectangle((m.columns.get_loc(5) - 0.5, m.index.get_loc(5) - 0.5), 1, 1,
                                               fill=False, ec="black", lw=2))
                cb = fig.colorbar(im, ax=ax, fraction=0.046)
                if pct:
                    from matplotlib.ticker import PercentFormatter

                    cb.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        fig.suptitle(f"Momentum sleeve, i longs x j shorts: {label} by weighting (black box: the live strategy)",
                     fontsize=11)
        fig.tight_layout()
        png = OUT / f"long_short_grid_{metric}.png"
        fig.savefig(png, dpi=120)
        plt.close(fig)
        pngs.append(png)
    print("\ncharts: " + "\n        ".join(str(p) for p in pngs))
    if sys.platform == "darwin" and not args.no_open:
        subprocess.run(["open", *map(str, pngs)], check=False)


if __name__ == "__main__":
    main()
