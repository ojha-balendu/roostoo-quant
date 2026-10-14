"""Threshold momentum: long every coin with score > T, short every coin with score < -T, exit at T/2.

    python research/27_threshold_momentum.py                      # T = 0.9,1.0,1.1,1.2,1.5; hourly and daily checks
    python research/27_threshold_momentum.py --thresholds 1.0,1.5 --checks hourly

score = the live ranking number: a coin's 16-day return after removing BTC's influence, divided by its own
volatility over those 16 days (a t-statistic of its trend versus BTC). The number of longs and shorts
follows the market (at most 8 per side); positions keep their size while held; BTC hedge; 95% gross;
25% cap per asset. Sizes: fixed (5.9% each) or inverse volatility (5.9% x median sigma / sigma).
Checks: every hour (as fast as the hourly bars allow) or daily at 00:00 UTC.
Benchmarks: 2 long / 1 short equal, the live 5/5 equal, 7/7 inverse volatility.
Results: the last 30 days, every 14-day window, and the windows that looked like now.
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
from quantbot.backtest.momentum import MomentumConfig, momentum_inputs
from quantbot.backtest.switching import momentum_decider
from quantbot.backtest.threshold import ThresholdConfig, threshold_decider
from quantbot.config import load_config
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe
from quantbot.rl.sim import simple_returns

BTC = "BTC/USD"
OUT = Path("reports/research")
LIVE = MomentumConfig(residual_riskadj=True, buffer_k=10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", default="0.9,1.0,1.1,1.2,1.5")
    ap.add_argument("--sizings", default="fixed,inverse_vol")
    ap.add_argument("--checks", default="hourly,daily")
    ap.add_argument("--exit-frac", type=float, default=0.5, help="exit when the score falls below this x T")
    ap.add_argument("--max-per-side", type=int, default=8)
    ap.add_argument("--last-days", type=float, default=30)
    ap.add_argument("--window-days", type=float, default=14)
    ap.add_argument("--no-windows", action="store_true", help="only the last 30 days (fast)")
    ap.add_argument("--similar-threshold", type=float, default=0.10)
    ap.add_argument("--cap", type=float, default=0.25)
    ap.add_argument("--capital", type=float, default=100_000)
    ap.add_argument("--config", default="config/competition.yaml")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

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
    hourly = np.ones(len(idx), dtype=bool)
    sim_cfg = CompositeConfig(capital=args.capital, fee=cfg.fees.taker, spread_bps=cfg.backtest.spread_bps).sim_config()
    end = len(idx) - 1
    unit = 0.95 / (2 * args.max_per_side)

    def bench(i, j, scheme):
        mc = replace(LIVE, k=i, k_short=j, buffer_k=2 * i, buffer_short=2 * j, weighting=scheme, max_weight=args.cap)
        return lambda first, log=None: momentum_decider(mc, x, B, E, R, daily, first, args.cap)

    variants = {"2L/1S equal": bench(2, 1, "equal"), "5L/5S equal (live)": bench(5, 5, "equal"),
                "7L/7S inverse_vol": bench(7, 7, "inverse_vol")}
    for check in args.checks.split(","):
        for sizing in args.sizings.split(","):
            for T in (float(v) for v in args.thresholds.split(",")):
                tc = ThresholdConfig(threshold=T, exit_frac=args.exit_frac, max_per_side=args.max_per_side, unit=unit,
                                     sizing=sizing, cap=args.cap)
                mask = hourly if check == "hourly" else daily
                variants[f"T={T:g} {sizing} {check}"] = (
                    lambda first, log=None, tc=tc, mask=mask: threshold_decider(tc, x["score"], x["risk"], B, E, mask,
                                                                               first, log))

    def run(make, a, b):
        log: list = []
        out = simulate(P, idx, make(a, log), a, b, sim_cfg, R, TR)
        m = run_metrics(out, args.capital)
        nl = float(np.mean([v[0] for v in log])) if log else np.nan
        ns = float(np.mean([v[1] for v in log])) if log else np.nan
        return out, m, nl, ns, log

    def r(v: float) -> str:
        return f"{v:+.2f}" if np.isfinite(v) else "n/a"

    # ---------------------------------------------------------------- last N days
    last_a = int(idx.searchsorted(idx[end] - pd.Timedelta(days=args.last_days)))
    days = (idx[end] - idx[last_a]).total_seconds() / 86400
    print(f"{len(coins)} coins; {len(variants)} strategies; last {args.last_days:g} days {idx[last_a]:%Y-%m-%d %H:%M} -> "
          f"{idx[end]:%Y-%m-%d %H:%M} UTC; ${args.capital:,.0f}; exit at {args.exit_frac:g} x T; at most "
          f"{args.max_per_side} per side, {unit:.1%} each (fixed)\n", flush=True)
    rows, curves, logs = {}, {}, {}
    for k, make in variants.items():
        out, m, nl, ns, log = run(make, last_a, end)
        rows[k] = {**m, "orders_day": out["trades"] / days, "longs": nl, "shorts": ns}
        curves[k], logs[k] = out["equity"], log
    t = pd.DataFrame(rows).T.sort_values("score", ascending=False)
    pd.set_option("display.width", 230)
    print(f"LAST {args.last_days:g} DAYS, sorted by SCORE (returns net of fees):")
    print(pd.DataFrame({
        "SCORE": t["score"].map(r), "return": t["return"].map("{:+.2%}".format), "Sharpe": t["sharpe"].map(r),
        "Sortino": t["sortino"].map(r), "Calmar": t["calmar"].map(r), "max DD": t["max_dd"].map("{:.2%}".format),
        "fees": t["fees_pct"].map("{:.2%}".format), "orders/day": t["orders_day"].map("{:.1f}".format),
        "avg longs": t["longs"].map(lambda v: f"{v:.1f}" if np.isfinite(v) else "fixed"),
        "avg shorts": t["shorts"].map(lambda v: f"{v:.1f}" if np.isfinite(v) else "fixed")}).to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "threshold_last.csv")

    # ---------------------------------------------------------------- every 14-day window
    if not args.no_windows:
        step = int(args.window_days * 24)
        t0 = int(idx.searchsorted(idx.min() + pd.Timedelta(days=75)))
        starts = list(range(t0, end - step + 1, step))
        two_one = variants["2L/1S equal"]
        prev = {st: run(two_one, int(idx.searchsorted(idx[st] - pd.Timedelta(days=30))), st)[1]["return"] for st in starts}
        similar = {st for st in starts if prev[st] > args.similar_threshold}
        wrows = []
        for k, make in variants.items():
            per = []
            for st in starts:
                out, m, nl, ns, _ = run(make, st, st + step)
                per.append({"start": st, "score": m["score"], "sharpe": m["sharpe"], "return": m["return"],
                            "max_dd": m["max_dd"], "orders_day": out["trades"] / args.window_days})
            df = pd.DataFrame(per)
            for label, sub in (("all", df), ("like now", df[df["start"].isin(similar)])):
                if len(sub):
                    wrows.append({"variant": k, "set": label, "n": len(sub), "score": sub["score"].median(),
                                  "sharpe": sub["sharpe"].median(), "ret": sub["return"].median(),
                                  "pos": (sub["return"] > 0).mean(), "worst": sub["return"].min(),
                                  "worst_dd": sub["max_dd"].min(), "orders": sub["orders_day"].mean()})
            print(f"  windows done: {k}", flush=True)
        w = pd.DataFrame(wrows)
        for label in ("all", "like now"):
            sub = w[w["set"] == label].sort_values("score", ascending=False)
            if sub.empty:
                print(f"\nNo windows where 2L/1S made more than {args.similar_threshold:.0%} in the previous 30 days.")
                continue
            title = (f"EVERY {args.window_days:g}-DAY WINDOW ({int(sub['n'].iloc[0])} windows)" if label == "all" else
                     f"WINDOWS LIKE NOW: 2L/1S made > {args.similar_threshold:.0%} in the 30 days before "
                     f"({int(sub['n'].iloc[0])} windows)")
            print(f"\n{title}, sorted by median score:")
            print(pd.DataFrame({
                "median SCORE": sub["score"].map(r), "median Sharpe": sub["sharpe"].map(r),
                "median return": sub["ret"].map("{:+.2%}".format), "% positive": sub["pos"].map("{:.0%}".format),
                "worst return": sub["worst"].map("{:+.1%}".format), "worst DD": sub["worst_dd"].map("{:.1%}".format),
                "orders/day": sub["orders"].map("{:.1f}".format)}).set_axis(sub["variant"]).to_string())
        w.to_csv(OUT / "threshold_windows.csv", index=False)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2.3, 1]})
    show = [k for k in t.index if k.startswith("T=")][:5] + ["2L/1S equal", "5L/5S equal (live)", "7L/7S inverse_vol"]
    for k in show:
        eq = curves[k]
        bench_line = not k.startswith("T=")
        ax.plot(eq.index, (eq / eq.iloc[0] - 1) * 100, lw=1.2 if bench_line else 1.8, ls="--" if bench_line else "-",
                label=f"{k} (score {r(t.loc[k, 'score'])}, {t.loc[k, 'return']:+.1%})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("return % (net of fees)")
    ax.set_title(f"Threshold momentum, last {args.last_days:g} days: the 5 best thresholds vs fixed-count books")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    best = next((k for k in t.index if k.startswith("T=") and "hourly" in k), None)
    if best and logs[best]:
        counts = np.array(logs[best])
        ts = idx[last_a:last_a + len(counts)]
        ax2.step(ts, counts[:, 0], where="post", color="tab:green", label="longs held")
        ax2.step(ts, -counts[:, 1], where="post", color="tab:red", label="shorts held (shown negative)")
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_ylabel("positions")
        ax2.set_title(f"How many coins {best} held (it follows the market's strength)", fontsize=9)
        ax2.legend(fontsize=8, loc="upper left")
        ax2.grid(alpha=0.3)
    loc = mdates.AutoDateLocator()
    ax2.xaxis.set_major_locator(loc)
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.tight_layout()
    png = OUT / "threshold_momentum.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"\nchart: {png}")
    if sys.platform == "darwin" and not args.no_open:
        subprocess.run(["open", str(png)], check=False)


if __name__ == "__main__":
    main()
