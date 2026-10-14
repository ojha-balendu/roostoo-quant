"""Rule 1, stop-and-switch: start in book A (2 long / 1 short, equal), switch for good to book B
(7 long / 7 short, inverse volatility) the first time the account falls X% below its peak.

    python research/26_stop_switch.py                                   # last 30 days, stops 3/5/8%
    python research/26_stop_switch.py --stops 0.02,0.04,0.06 --a 2/1/equal --b 7/7/inverse_vol
    python research/26_stop_switch.py --windows                        # also every 14-day window since 2023

Books use the live ranking (16-day BTC-removed momentum / volatility, daily at 00:00 UTC), equal dollars
per leg, BTC hedge, 95% gross, 25% cap per asset (hedge included), rank buffer 2i / 2j. The drawdown is
checked on the account's equity (after fees) at every hourly close; on a breach the whole book is
switched to B at that close. Compared with A only, B only and the live 5/5 equal book.
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
from quantbot.backtest.composite_bt import CompositeConfig, run_metrics
from quantbot.backtest.momentum import MomentumConfig, momentum_inputs
from quantbot.backtest.switching import momentum_decider, simulate_with_stop
from quantbot.config import load_config
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe
from quantbot.rl.sim import simple_returns

BTC = "BTC/USD"
OUT = Path("reports/research")
LIVE = MomentumConfig(residual_riskadj=True, buffer_k=10)


def book(spec: str, cap: float) -> tuple[str, MomentumConfig]:
    i, j, scheme = spec.split("/")
    i, j = int(i), int(j)
    return (f"{i}L/{j}S {scheme}",
            replace(LIVE, k=i, k_short=j, buffer_k=2 * i, buffer_short=2 * j, weighting=scheme, max_weight=cap))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="2/1/equal", help="starting book: longs/shorts/weighting")
    ap.add_argument("--b", default="7/7/inverse_vol", help="book after the stop: longs/shorts/weighting")
    ap.add_argument("--stops", default="0.03,0.05,0.08", help="drawdowns that trigger the switch")
    ap.add_argument("--last-days", type=float, default=30)
    ap.add_argument("--windows", action="store_true", help="also run every 14-day window since the data starts")
    ap.add_argument("--window-days", type=float, default=14)
    ap.add_argument("--similar-threshold", type=float, default=0.10,
                    help="windows 'like now': book A made more than this in the 30 days before the window")
    ap.add_argument("--cap", type=float, default=0.25)
    ap.add_argument("--capital", type=float, default=100_000)
    ap.add_argument("--config", default="config/competition.yaml")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    stops = [float(s) for s in args.stops.split(",")]

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
    sim_cfg = CompositeConfig(capital=args.capital, fee=cfg.fees.taker, spread_bps=cfg.backtest.spread_bps).sim_config()
    end = len(idx) - 1
    name_a, mc_a = book(args.a, args.cap)
    name_b, mc_b = book(args.b, args.cap)
    name_live, mc_live = "5L/5S equal (live)", replace(LIVE, max_weight=args.cap)

    def dec(mc, first):
        return momentum_decider(mc, x, B, E, R, daily, first, args.cap)

    def run(a, b, mc_first, stop, mc_second=None):
        second = mc_second or mc_b
        return simulate_with_stop(P, idx, dec(mc_first, a), lambda t: dec(second, t), a, b, sim_cfg, stop, R, TR)

    variants = {f"{name_a} only": (mc_a, None), f"{name_b} only": (mc_b, None), name_live: (mc_live, None)}
    for s in stops:
        variants[f"{name_a} -> {name_b} at -{s:.0%}"] = (mc_a, s)

    # ------------------------------------------------------------------ last N days
    last_a = int(idx.searchsorted(idx[end] - pd.Timedelta(days=args.last_days)))
    print(f"{len(coins)} coins; last {args.last_days:g} days {idx[last_a]:%Y-%m-%d %H:%M} -> {idx[end]:%Y-%m-%d %H:%M} UTC; "
          f"${args.capital:,.0f}; book A = {name_a}, book B = {name_b}\n")
    res = {k: run(last_a, end, mc, stop) for k, (mc, stop) in variants.items()}
    pure_a = res[f"{name_a} only"]["equity"]

    def r(v: float) -> str:
        return f"{v:+.2f}" if np.isfinite(v) else "n/a"

    rows = {}
    for k, out in res.items():
        m = run_metrics(out, args.capital)
        rows[k] = {**m, "switched": out["switched_at"], "dd_at_switch": out["drawdown_at_switch"]}
    t = pd.DataFrame(rows).T
    pd.set_option("display.width", 220)
    print(f"LAST {args.last_days:g} DAYS (each run starts flat with ${args.capital:,.0f}; returns are net of fees):")
    print(pd.DataFrame({
        "SCORE": t["score"].map(r), "return": t["return"].map("{:+.2%}".format), "Sharpe": t["sharpe"].map(r),
        "Sortino": t["sortino"].map(r), "Calmar": t["calmar"].map(r), "max DD": t["max_dd"].map("{:.2%}".format),
        "fees": t["fees_pct"].map("{:.2%}".format), "orders": t["trades"].map("{:.0f}".format),
        "switched at (UTC)": t["switched"].map(lambda v: f"{v:%m-%d %H:%M}" if v is not None and pd.notna(v) else "no")}
        ).to_string())

    print("\nSWITCHES:")
    for s in stops:
        k = f"{name_a} -> {name_b} at -{s:.0%}"
        out = res[k]
        if out["switched_at"] is None:
            print(f"  stop -{s:.0%}: never triggered; the deepest drawdown of {name_a} was "
                  f"{(pure_a / pure_a.cummax() - 1).min():.2%}, so the run stayed in {name_a} throughout")
            continue
        eq, ts = out["equity"], out["switched_at"]
        a_after = pure_a.iloc[-1] / pure_a.loc[ts] - 1
        s_after = eq.iloc[-1] / eq.loc[ts] - 1
        print(f"  stop -{s:.0%}: switched on {ts:%Y-%m-%d %H:%M} UTC at a drawdown of {out['drawdown_at_switch']:.2%} "
              f"(equity {eq.loc[ts] / eq.iloc[0] - 1:+.2%} at that point).")
        print(f"      after the switch: {name_b} made {s_after:+.2%}; staying in {name_a} would have made {a_after:+.2%} "
              f"-> switching {'helped' if s_after > a_after else 'hurt'} by {abs(s_after - a_after):.2%}")
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "stop_switch_last.csv")

    # ------------------------------------------------------------------ every 14-day window (optional)
    if args.windows:
        step = int(args.window_days * 24)
        t0 = int(idx.searchsorted(idx.min() + pd.Timedelta(days=75)))   # room for A's previous 30 days
        starts = list(range(t0, end - step + 1, step))
        prev = {}
        for st in starts:
            p0 = int(idx.searchsorted(idx[st] - pd.Timedelta(days=30)))
            prev[st] = run_metrics(run(p0, st, mc_a, None), args.capital)["return"]
        similar = [st for st in starts if prev[st] > args.similar_threshold]
        wrows = []
        for k, (mc, stop) in variants.items():
            per = []
            for st in starts:
                out = run(st, st + step, mc, stop)
                m = run_metrics(out, args.capital)
                per.append({"start": st, "score": m["score"], "sharpe": m["sharpe"], "return": m["return"],
                            "max_dd": m["max_dd"], "switched": out["switched_at"] is not None})
            df = pd.DataFrame(per)
            for label, sub in (("all", df), ("like now", df[df["start"].isin(similar)])):
                if len(sub):
                    wrows.append({"variant": k, "windows": label, "n": len(sub), "median score": sub["score"].median(),
                                  "median Sharpe": sub["sharpe"].median(), "median return": sub["return"].median(),
                                  "worst return": sub["return"].min(), "worst DD": sub["max_dd"].min(),
                                  "switched in": sub["switched"].mean()})
        w = pd.DataFrame(wrows)
        for label in ("all", "like now"):
            sub = w[w["windows"] == label]
            if sub.empty:
                print(f"\nNo windows where {name_a} made more than {args.similar_threshold:.0%} in the previous 30 days.")
                continue
            title = (f"EVERY {args.window_days:g}-DAY WINDOW ({int(sub['n'].iloc[0])} windows)" if label == "all" else
                     f"WINDOWS LIKE NOW: {name_a} made > {args.similar_threshold:.0%} in the 30 days before "
                     f"({int(sub['n'].iloc[0])} windows)")
            print(f"\n{title}:")
            print(pd.DataFrame({
                "median SCORE": sub["median score"].map(r), "median Sharpe": sub["median Sharpe"].map(r),
                "median return": sub["median return"].map("{:+.2%}".format),
                "worst return": sub["worst return"].map("{:+.1%}".format), "worst DD": sub["worst DD"].map("{:.1%}".format),
                "switched in": sub["switched in"].map("{:.0%} of windows".format)}).set_axis(sub["variant"]).to_string())
        w.to_csv(OUT / "stop_switch_windows.csv", index=False)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(14, 6.5))
    for k, out in res.items():
        eq = out["equity"]
        pure = k.endswith("only") or k == name_live
        line, = ax.plot(eq.index, (eq / eq.iloc[0] - 1) * 100, lw=1.3 if pure else 2.0, ls="--" if pure else "-",
                        label=f"{k} (score {r(t.loc[k, 'score'])}, {t.loc[k, 'return']:+.1%})")
        if out["switched_at"] is not None:
            ts = out["switched_at"]
            ax.scatter([ts], [(eq.loc[ts] / eq.iloc[0] - 1) * 100], color=line.get_color(), s=60, zorder=5, marker="v")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("return % (net of fees)")
    ax.set_title(f"Stop-and-switch, last {args.last_days:g} days (triangles: the moment of the switch)")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.tight_layout()
    png = OUT / "stop_switch.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"\nchart: {png}")
    if sys.platform == "darwin" and not args.no_open:
        subprocess.run(["open", str(png)], check=False)


if __name__ == "__main__":
    main()
