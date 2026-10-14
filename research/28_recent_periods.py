"""Compare momentum books over the most recent k days (default 5, 10, 15, 20, 30).

    python research/28_recent_periods.py
    python research/28_recent_periods.py --days 3,7,14 --books 2/1/equal,5/5/equal,7/7/inverse_vol
    python research/28_recent_periods.py --from-flat           # each book starts fresh k days ago

By default each book runs continuously from --run-days ago (so its positions are already in place, as
they would be live), and the last k days of that run are measured on their own. With --from-flat, each
book instead starts with no positions k days ago (like a new competition starting then; includes the
cost of building the book). Books: longs/shorts/weighting with the live ranking, BTC hedge, 95% gross,
25% cap per asset, rank buffer 2i / 2j, daily at 00:00 UTC. Ratios are annualised: over a few days
they swing a lot, so read the return and drawdown first.
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
from quantbot.backtest.metrics import compute_metrics
from quantbot.backtest.momentum import MomentumConfig, momentum_inputs
from quantbot.backtest.switching import momentum_decider
from quantbot.config import load_config
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe
from quantbot.rl.sim import simple_returns

BTC = "BTC/USD"
OUT = Path("reports/research")
LIVE = MomentumConfig(residual_riskadj=True, buffer_k=10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default="2/1/equal,5/5/equal,7/7/inverse_vol", help="longs/shorts/weighting, comma-separated")
    ap.add_argument("--days", default="5,10,15,20,30")
    ap.add_argument("--run-days", type=float, default=60, help="continuous run length before measuring")
    ap.add_argument("--from-flat", action="store_true")
    ap.add_argument("--cap", type=float, default=0.25)
    ap.add_argument("--capital", type=float, default=100_000)
    ap.add_argument("--config", default="config/competition.yaml")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    ks = [float(k) for k in args.days.split(",")]

    cfg = load_config(args.config)
    universe = load_universe(cfg.data.universe_path)
    probe = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", {BTC: universe[BTC]})
    if probe.empty:
        raise SystemExit("no hourly data: run  quantbot download")
    start = (probe.index.max() - pd.Timedelta(days=args.run_days + 120)).strftime("%Y-%m-%d")
    close = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", universe, start=start).ffill(limit=3)
    qv = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", universe, start=start, field="quote_volume")
    if close.empty or BTC not in close:
        raise SystemExit("no recent hourly data: run  quantbot download")
    resid, beta = residual_returns(close, BTC)
    coins = list(resid.columns)
    elig = eligibility(close, qv)[coins]
    x = {k: v.reindex(columns=coins).to_numpy(dtype=float) for k, v in momentum_inputs(close, resid, elig, LIVE).items()}
    idx = close.index
    P = close[coins + [BTC]].to_numpy(dtype=float)
    R, TR = simple_returns(P), np.isfinite(P)
    B, E = beta[coins].to_numpy(), elig.to_numpy(dtype=bool)
    daily = np.asarray((idx + pd.Timedelta(hours=1)).hour == 0)
    sim_cfg = CompositeConfig(capital=args.capital, fee=cfg.fees.taker, spread_bps=cfg.backtest.spread_bps).sim_config()
    end = len(idx) - 1

    books = {}
    for spec in args.books.split(","):
        i, j, scheme = spec.split("/")
        i, j = int(i), int(j)
        label = f"{i}L/{j}S {scheme}" + (" (live)" if (i, j, scheme) == (5, 5, "equal") else "")
        books[label] = replace(LIVE, k=i, k_short=j, buffer_k=2 * i, buffer_short=2 * j, weighting=scheme,
                               max_weight=args.cap)

    def run(mc, a):
        return simulate(P, idx, momentum_decider(mc, x, B, E, R, daily, a, args.cap), a, end, sim_cfg, R, TR)

    run_a = int(idx.searchsorted(idx[end] - pd.Timedelta(days=args.run_days)))
    full = {} if args.from_flat else {name: run(mc, run_a) for name, mc in books.items()}
    mode = ("each book starts with no positions k days ago" if args.from_flat else
            f"one continuous run per book from {idx[run_a]:%Y-%m-%d}; the last k days measured on their own")
    print(f"{len(coins)} coins; data to {idx[end]:%Y-%m-%d %H:%M} UTC; {mode}; returns net of fees\n")

    def r(v: float) -> str:
        return f"{v:+.2f}" if np.isfinite(v) else "n/a"

    rows, curves = [], {}
    for k in ks:
        a = int(idx.searchsorted(idx[end] - pd.Timedelta(days=k)))
        for name, mc in books.items():
            if args.from_flat:
                out = run(mc, a)
                eq, m = out["equity"], run_metrics(out, args.capital)
                fees = out["fees"] / args.capital
            else:
                eq = full[name]["equity"].loc[idx[a]:]
                m = compute_metrics(eq)
                m = {"score": m.get("score", np.nan), "return": m.get("total_return", np.nan), "sharpe": m.get("sharpe", np.nan),
                     "sortino": m.get("sortino", np.nan), "calmar": m.get("calmar", np.nan),
                     "max_dd": m.get("max_drawdown", np.nan)}
                fees = np.nan
            rows.append({"k": k, "book": name, **{c: m[c] for c in ("score", "return", "sharpe", "sortino", "calmar", "max_dd")},
                         "fees": fees})
            curves[(k, name)] = eq / eq.iloc[0]
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for k in ks:
        sub = t[t["k"] == k].set_index("book")
        print(f"LAST {k:g} DAYS:")
        cols = {"SCORE": sub["score"].map(r), "return": sub["return"].map("{:+.2%}".format), "Sharpe": sub["sharpe"].map(r),
                "Sortino": sub["sortino"].map(r), "Calmar": sub["calmar"].map(r), "max DD": sub["max_dd"].map("{:.2%}".format)}
        if args.from_flat:
            cols["fees"] = sub["fees"].map("{:.2%}".format)
        print(pd.DataFrame(cols).to_string())
        print(f"  best return: {sub['return'].idxmax()}   best score: {sub['score'].idxmax()}\n")
    print("Returns by period (the quickest overview):")
    print(t.pivot(index="book", columns="k", values="return").rename(columns=lambda k: f"last {k:g}d")
          .map("{:+.2%}".format).to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "recent_periods.csv", index=False)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cols_n = min(len(ks), 3)
    rows_n = int(np.ceil(len(ks) / cols_n))
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(5.2 * cols_n, 3.9 * rows_n), squeeze=False)
    colors = dict(zip(books, ["tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:red", "tab:brown"], strict=False))
    for n, k in enumerate(ks):
        ax = axes[n // cols_n, n % cols_n]
        for name in books:
            c = curves[(k, name)]
            ax.plot(c.index, (c - 1) * 100, color=colors[name], lw=1.6,
                    label=f"{name} {t[(t['k'] == k) & (t['book'] == name)]['return'].iloc[0]:+.1%}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"last {k:g} days", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(labelsize=7)
    for n in range(len(ks), rows_n * cols_n):
        axes[n // cols_n, n % cols_n].axis("off")
    fig.suptitle("Return % over the most recent k days (net of fees)", fontsize=11)
    fig.tight_layout()
    png = OUT / "recent_periods.png"
    fig.savefig(png, dpi=125)
    plt.close(fig)
    print(f"\nchart: {png}")
    if sys.platform == "darwin" and not args.no_open:
        subprocess.run(["open", str(png)], check=False)


if __name__ == "__main__":
    main()
