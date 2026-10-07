"""Composite indicator signal: v1 (top/bottom decile) vs v2 (trade only if expected profit > fee).

    python research/14_composite.py                     # 1-hour bars ("next tick" = next hour)
    python research/14_composite.py --bar 4h --max-hold-bars 6
    python research/14_composite.py --normalize raw     # literal weighted sum of raw indicator values

Signal: weighted sum of the indicators (coefficients in quantbot/analysis/composite.py), each first
normalised across coins (z-score), then mapped to [-1, 1] across the eligible coins every bar.
Both strategies are beta-neutral via BTC, start flat with $1M, pay 0.1% + half the spread per trade.
Reported: the older period and the last year separately, every 14-day window as its own competition,
a placebo (signal shuffled across coins: must lose about the fees), runs without fees (how much the
signal earns before costs), and a chart. See backtest/composite_bt.py for the exact rules.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.analysis.composite import COEFFS, affine_unit, composite, term_shares
from quantbot.analysis.indicators import compute_named
from quantbot.analysis.xsection import eligibility, residual_returns
from quantbot.backtest.composite_bt import (
    CompositeConfig,
    calibrate,
    run_metrics,
    simulate,
    v1_weights,
    v2_weights,
)
from quantbot.backtest.windows import summarize
from quantbot.config import load_config
from quantbot.data.loader import load_ohlcv
from quantbot.data.universe import load_universe

BTC = "BTC/USD"
OUT = Path("reports/research")


def row_rank_ic(sp: pd.DataFrame, nxt: pd.DataFrame) -> pd.Series:
    a, b = sp.rank(axis=1), nxt.where(sp.notna()).rank(axis=1)
    a, b = a.where(b.notna()), b.where(a.notna())
    ac, bc = a.sub(a.mean(axis=1), axis=0), b.sub(b.mean(axis=1), axis=0)
    den = np.sqrt((ac ** 2).sum(axis=1) * (bc ** 2).sum(axis=1))
    return ((ac * bc).sum(axis=1) / den.where(den > 0)).dropna()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/competition.yaml")
    ap.add_argument("--bar", default="1h")
    ap.add_argument("--normalize", default="zscore", choices=["zscore", "rank", "raw"])
    ap.add_argument("--decile", type=float, default=0.10)
    ap.add_argument("--calib-days", type=float, default=90)
    ap.add_argument("--max-hold-bars", type=int, default=24)
    ap.add_argument("--band", type=float, default=0.005)
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--spread-bps", type=float, default=2.0)
    ap.add_argument("--holdout-days", type=float, default=365)
    ap.add_argument("--window-days", type=float, default=14)
    ap.add_argument("--min-volume", type=float, default=200_000, help="min median USD volume per hour")
    ap.add_argument("--allow-flip", action="store_true", help="v2 may trade the signal inverted")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    cfg_file = load_config(args.config)
    bar = pd.Timedelta(args.bar)
    bpd = pd.Timedelta(days=1) / bar
    universe = load_universe(cfg_file.data.universe_path)
    f = load_ohlcv(cfg_file.data.processed_dir, cfg_file.data.market, universe, cfg_file.data.start, args.bar)
    c = f["close"]
    if c.empty or BTC not in c:
        raise SystemExit("no hourly data: run  quantbot download")
    resid, beta = residual_returns(c, BTC, beta_hours=int(30 * bpd))
    coins = list(resid.columns)
    eligible = eligibility(c, f["quote_volume"], min_history_hours=int(30 * bpd),
                           min_usd_per_hour=args.min_volume * (bar / pd.Timedelta(hours=1)))
    named = {k: v[coins] for k, v in compute_named(f["open"], f["high"], f["low"], c, f["volume"],
                                                     f["quote_volume"]).items()}
    s = composite(named, eligible, COEFFS, args.normalize)
    sp = affine_unit(s)
    cfg = CompositeConfig(capital=args.capital, fee=cfg_file.fees.taker, spread_bps=args.spread_bps,
                          decile=args.decile, band=args.band, calib_bars=int(args.calib_days * bpd),
                          max_hold_bars=args.max_hold_bars, allow_flip=args.allow_flip)
    cal = calibrate(sp, resid, cfg)

    idx = c.index
    P = c[coins + [BTC]].to_numpy(dtype=float)
    S, E, B, A = sp.to_numpy(), eligible.to_numpy(dtype=bool), beta[coins].to_numpy(), cal["A"].to_numpy()
    enough = (np.isfinite(S) & E).sum(axis=1) >= cfg.min_coins
    t0 = int(np.flatnonzero(enough & cal["b"].notna().to_numpy())[0]) if (enough & cal["b"].notna().to_numpy()).any() \
        else None
    if t0 is None:
        raise SystemExit("not enough history for the calibration window: reduce --calib-days or download more")
    end = len(idx) - 1
    t_hold = int(idx.searchsorted(idx[-1] - pd.Timedelta(days=args.holdout_days)))
    t_hold = max(t_hold, t0 + 1)
    print(f"{len(coins)} coins on {args.bar} bars ({args.normalize} normalisation); median "
          f"{int(np.median((np.isfinite(S) & E).sum(axis=1)[t0:]))} eligible per bar; trading from "
          f"{idx[t0]:%Y-%m-%d}; last year starts {idx[t_hold]:%Y-%m-%d}; ${args.capital:,.0f}, cost "
          f"{cfg.cost * 1e4:.0f} bps per side\n")

    shares = term_shares(named, eligible, COEFFS, args.normalize)
    print("What drives the composite (share of its cross-coin variation):")
    print("  " + ", ".join(f"{k} {v:.0%}" for k, v in shares.head(8).items()) + (", ..." if len(shares) > 8 else ""))
    nxt = resid.shift(-1)
    for name, (a, b) in {"older period": (t0, t_hold), "last year": (t_hold, end)}.items():
        ic = row_rank_ic(sp.iloc[a:b], nxt.iloc[a:b])
        print(f"  signal quality, {name}: rank IC with next-bar relative return {ic.mean():+.4f} "
              f"(t {ic.mean() / ic.std() * np.sqrt(len(ic)):+.1f}, positive in {(ic > 0).mean():.0%} of bars)")
    bb = cal["b"].iloc[t0:] * 1e4
    print(f"  v2 calibration: expected next-bar edge of the top coin vs average ~{bb.median():+.2f} bps (median), "
          f"positive in {(bb > 0).mean():.0%} of bars; signal persistence rho {cal['rho'].iloc[t0:].mean():.2f}; "
          f"entry + exit costs {2 * cfg.cost * 1e4:.0f} bps\n")

    rng = np.random.default_rng(0)
    S_perm = S.copy()
    for t in range(len(S_perm)):
        ok = np.flatnonzero(np.isfinite(S_perm[t]))
        S_perm[t, ok] = S_perm[t, rng.permutation(ok)]

    def hold_btc(start):
        target = np.zeros(P.shape[1])
        target[-1] = cfg.gross
        return lambda t, cur: target if t == start else cur

    def strategies(start):
        return {
            "v1 decile cutoff": (lambda t, cur: v1_weights(S[t], E[t], B[t], cfg), False),
            "v2 profit > fee": (lambda t, cur: v2_weights(A[t], cur, E[t], B[t], cfg), False),
            "hold BTC": (hold_btc(start), False),
            "v1 placebo (shuffled)": (lambda t, cur: v1_weights(S_perm[t], E[t], B[t], cfg), False),
            "v1 without fees": (lambda t, cur: v1_weights(S[t], E[t], B[t], cfg), True),
            "v2 without fees": (lambda t, cur: v2_weights(A[t], cur, E[t], B[t], cfg), True),
        }

    pd.set_option("display.width", 220)
    curves = {}
    for name, (a, b) in {"OLDER PERIOD": (t0, t_hold), "LAST YEAR": (t_hold, end)}.items():
        rows = {}
        for sname, (decide, zero) in strategies(a).items():
            res = simulate(P, idx, decide, a, b, cfg.sim_config(zero_cost=zero))
            rows[sname] = run_metrics(res, cfg.capital)
            curves.setdefault(sname, []).append(res["equity"])
        t = pd.DataFrame(rows).T
        show = pd.DataFrame({
            "score": t["score"].map("{:+.2f}".format),
            "return": t["return"].map("{:+.1%}".format),
            "CAGR": t["cagr"].map(lambda v: f"{v:+.1%}" if np.isfinite(v) else "n/a"),
            "Sharpe": t["sharpe"].map("{:.2f}".format), "Sortino": t["sortino"].map("{:.2f}".format),
            "Calmar": t["calmar"].map("{:.2f}".format), "max DD": t["max_dd"].map("{:.1%}".format),
            "P&L before fees": t["gross_pnl"].map("{:+.1%}".format), "fees": t["fees_pct"].map("{:.1%}".format),
            "turnover/yr": t["turnover_yr"].map("{:.0f}x".format), "trades": t["trades"].map("{:,.0f}".format),
            "avg coins held": t["avg_positions"].map("{:.1f}".format)})
        print(f"{name}: {idx[a]:%Y-%m-%d} -> {idx[b]:%Y-%m-%d}, one continuous run from flat")
        print(show.to_string())
        pl = rows["v1 placebo (shuffled)"]
        print(f"  placebo check: shuffled signal made {pl['gross_pnl']:+.1%} before fees and paid {pl['fees_pct']:.1%} "
              "in fees (before-fee result should be near zero: the backtest is not finding money in noise)\n")

    step = int(round(args.window_days * bpd))
    wins = {k: [] for k in ("v1 decile cutoff", "v2 profit > fee", "hold BTC")}
    for a in range(t0, end - step + 1, step):
        for sname in wins:
            decide, zero = strategies(a)[sname]
            res = simulate(P, idx, decide, a, a + step, cfg.sim_config(zero_cost=zero))
            m = run_metrics(res, cfg.capital)
            wins[sname].append({"return": m["return"], "max_dd": m["max_dd"], "sharpe": m["sharpe"],
                                "sortino": m["sortino"], "calmar": m["calmar"], "score": m["score"], "fees": res["fees"],
                                "trades": res["trades"], "avg_gross": res["avg_gross"], "beta_btc": np.nan})
    wdf = {k: pd.DataFrame(v) for k, v in wins.items()}
    summ = pd.DataFrame({k: summarize(v, versus=wdf["hold BTC"]) for k, v in wdf.items()})
    keep = ["windows", "median_return", "mean_return", "worst_return", "pct_positive", "median_max_dd", "worst_max_dd",
            "median_sharpe", "median_sortino", "median_calmar", "median_score", "avg_fees", "pct_beats_benchmark"]
    pct = {"median_return", "mean_return", "worst_return", "pct_positive", "median_max_dd", "worst_max_dd",
           "pct_beats_benchmark"}
    show = summ.loc[keep].astype(object)
    for r in keep:
        for col in show.columns:
            v = summ.loc[r, col]
            show.loc[r, col] = f"{v:.1%}" if r in pct else (f"{v:,.0f}" if r in ("windows", "avg_fees") else f"{v:.2f}")
    print(f"EVERY {args.window_days:g}-DAY WINDOW as a separate ${args.capital:,.0f} competition (all periods):")
    print(show.rename(index={"pct_beats_benchmark": "pct_beats_btc"}).to_string())

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    colors = {"v1 decile cutoff": "tab:blue", "v2 profit > fee": "tab:green", "hold BTC": "tab:orange",
              "v1 placebo (shuffled)": "tab:gray"}
    for sname, col in colors.items():
        parts, level, xs, ys = curves[sname], 1.0, [], []
        for eq in parts:   # older period then last year, chained (the last year restarts flat)
            e = eq / eq.iloc[0] * level
            xs.extend(e.index)
            ys.extend(e.to_numpy())
            level = e.iloc[-1]
        y = np.array(ys)
        axes[0].plot(xs, y, label=f"{sname} ({level - 1:+.0%})", color=col,
                     lw=2 if sname.startswith("v") and "placebo" not in sname else 1.3)
        if sname != "v1 placebo (shuffled)":
            axes[1].plot(xs, (y / np.maximum.accumulate(y) - 1) * 100, color=col, lw=1.1, label=sname)
    axes[2].plot(cal["b"].index[t0:], cal["b"].iloc[t0:] * 1e4, color="tab:green", lw=1)
    axes[2].axhline(2 * cfg.cost * 1e4 / max(cal["persistence"].iloc[t0:].median(), 1e-9), color="tab:red", ls="--",
                    lw=1, label="edge needed to open a position at typical persistence")
    axes[2].axhline(0, color="k", lw=0.5)
    for ax in axes:
        ax.axvline(idx[t_hold], color="k", ls=":", lw=1)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_title(f"Composite indicator signal, {args.bar} bars, ${args.capital:,.0f}, fees included "
                      f"(dotted line: last year, restarted flat)")
    axes[0].set_yscale("log")
    axes[0].axhline(1.0, color="k", lw=0.5)
    axes[0].set_ylabel("equity / start (log scale)")
    axes[1].set_ylabel("drawdown %")
    axes[2].set_ylabel("v2 calibration\n(bps per unit s')")
    loc = mdates.AutoDateLocator()
    axes[2].xaxis.set_major_locator(loc)
    axes[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"composite_{args.bar}.png"
    fig.savefig(png, dpi=130)
    print(f"\nchart: {png}")
    if sys.platform == "darwin" and not args.no_open:
        subprocess.run(["open", str(png)], check=False)


if __name__ == "__main__":
    main()
