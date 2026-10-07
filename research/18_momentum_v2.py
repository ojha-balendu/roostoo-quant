"""Momentum sleeve, better executed: each improvement added one at a time (A = research/17's sleeve).

    python research/18_momentum_v2.py

  A  long 5 / short 5 by raw 16-day return, equal weight, daily (= research/17 part 2)
  B  + rank by residual (BTC removed), risk-adjusted momentum
  C  + rank buffer: enter in the top 5, exit only outside the top 10
  D  + inverse-volatility weights, 20% portfolio volatility target, 15% cap per coin
  E  + blend of 7/16/30-day lookbacks, skipping the most recent day
  F  + trend consistency: >= 55% of days moved in the trend's direction ("frog in the pan")
  G  + proximity to the 30-day high added to the score
Selection rule, fixed in advance: the best median competition score (0.4 Sortino + 0.3 Sharpe + 0.3 Calmar)
over the 14-day windows of the OLDER period.
The last year is reported for confirmation only (research/17 already showed A's last year).
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
from quantbot.backtest.metrics import SCORE_WEIGHTS
from quantbot.backtest.momentum import MomentumConfig, momentum_inputs, sleeve_weights
from quantbot.backtest.windows import summarize
from quantbot.config import load_config
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe
from quantbot.rl.sim import simple_returns

BTC = "BTC/USD"
OUT = Path("reports/research")


def variants() -> dict[str, tuple[str, MomentumConfig]]:
    a = MomentumConfig()
    b = replace(a, residual_riskadj=True)
    c = replace(b, buffer_k=10)
    d = replace(c, weighting="inverse_vol", vol_target=0.20, max_weight=0.15)
    e = replace(d, lookbacks_days=[7, 16, 30], skip_days=1)
    f = replace(e, min_consistency=0.55)
    g = replace(f, proximity=True)
    return {"A": ("H3 as is", a), "B": ("+ residual risk-adj", b), "C": ("+ rank buffer", c),
            "D": ("+ inv-vol, vol target", d), "E": ("+ multi-lookback", e), "F": ("+ consistency", f),
            "G": ("+ 30d-high proximity", g)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-days", type=float, default=365)
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    universe = load_universe(cfg.data.universe_path)
    close = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", universe, start=cfg.data.start).ffill(limit=3)
    qv = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", universe, start=cfg.data.start, field="quote_volume")
    resid, beta = residual_returns(close, BTC)
    coins = list(resid.columns)
    eligible = eligibility(close, qv)[coins]
    idx = close.index
    P = close[coins + [BTC]].to_numpy(dtype=float)
    rets = simple_returns(P)
    E, Bv = eligible.to_numpy(dtype=bool), beta[coins].to_numpy()
    closes_at = idx + pd.Timedelta(hours=1)
    daily = np.asarray((closes_at.hour == 0) & (closes_at.minute == 0))
    bc = CompositeConfig(capital=args.capital, fee=cfg.fees.taker)
    t0 = int(idx.searchsorted(idx.min() + pd.Timedelta(days=45)))
    th = int(idx.searchsorted(idx.max() - pd.Timedelta(days=args.holdout_days)))
    end = len(idx) - 1
    step = 14 * 24
    print(f"{len(coins)} coins, ${args.capital:,.0f}, daily rebalance at 00:00 UTC; older period "
          f"{idx[t0]:%Y-%m-%d} -> {idx[th]:%Y-%m-%d} (used to choose), last year -> {idx[end]:%Y-%m-%d} (confirmation)\n")

    def runner(mc: MomentumConfig):
        x = {k: v.reindex(columns=coins).to_numpy(dtype=float) for k, v in momentum_inputs(close, resid, eligible, mc).items()}

        def make(first):
            return lambda t, cur: (sleeve_weights(t, cur, x, Bv, E, rets, mc) if (daily[t] or t == first) else cur)
        return make

    def hold_btc(first):
        tgt = np.zeros(P.shape[1])
        tgt[-1] = bc.gross
        return lambda t, cur: tgt if t == first else cur

    def windows(make, a, b):
        rows = []
        for s in range(a, b - step + 1, step):
            r = simulate(P, idx, make(s), s, s + step, bc.sim_config())
            m = run_metrics(r, args.capital)
            rows.append({"return": m["return"], "max_dd": m["max_dd"], "sharpe": m["sharpe"], "sortino": m["sortino"],
                         "calmar": m["calmar"], "score": m["score"], "fees": r["fees"], "trades": r["trades"], "avg_gross": r["avg_gross"],
                         "beta_btc": np.nan})
        return pd.DataFrame(rows)

    results, curves = {}, {}
    makers = {name: runner(mc) for name, (_, mc) in variants().items()}
    makers["BTC"] = hold_btc
    for name, make in makers.items():
        row = {}
        for period, (a, b) in {"old": (t0, th), "last": (th, end)}.items():
            r = simulate(P, idx, make(a), a, b, bc.sim_config())
            m = run_metrics(r, args.capital)
            row.update({f"{period}_{k}": m[k] for k in ("score", "return", "sharpe", "sortino", "calmar", "max_dd", "fees_pct")})
            row[f"{period}_orders_day"] = r["trades"] / ((idx[b] - idx[a]).total_seconds() / 86400)
            curves.setdefault(name, []).append(r["equity"])
            w = summarize(windows(make, a, b))
            row.update({f"{period}_w_score": w["median_score"], f"{period}_w_sharpe": w["median_sharpe"],
                        f"{period}_w_calmar": w["median_calmar"],
                        f"{period}_w_sortino": w["median_sortino"], f"{period}_w_return": w["median_return"],
                        f"{period}_w_worst": w["worst_return"], f"{period}_w_worst_dd": w["worst_max_dd"],
                        f"{period}_w_positive": w["pct_positive"]})
        results[name] = row
        print(f"  ran {name}", flush=True)
    t = pd.DataFrame(results).T
    labels = {k: v[0] for k, v in variants().items()} | {"BTC": "hold BTC"}

    def table(period: str, title: str) -> None:
        print(f"\n{title}")
        print(pd.DataFrame({
            "variant": [f"{k} {labels[k]}" for k in t.index],
            "SCORE": t[f"{period}_score"].map("{:+.2f}".format),
            "return": t[f"{period}_return"].map("{:+.1%}".format), "Sortino": t[f"{period}_sortino"].map("{:.2f}".format),
            "Sharpe": t[f"{period}_sharpe"].map("{:.2f}".format), "Calmar": t[f"{period}_calmar"].map("{:.2f}".format),
            "max DD": t[f"{period}_max_dd"].map("{:.1%}".format), "fees": t[f"{period}_fees_pct"].map("{:.1%}".format),
            "orders/day": t[f"{period}_orders_day"].map("{:.1f}".format),
            "| 14d median SCORE": t[f"{period}_w_score"].map("{:+.2f}".format),
            "Sortino ": t[f"{period}_w_sortino"].map("{:+.2f}".format), "Sharpe ": t[f"{period}_w_sharpe"].map("{:+.2f}".format),
            "Calmar ": t[f"{period}_w_calmar"].map("{:+.2f}".format),
            "median ret": t[f"{period}_w_return"].map("{:+.1%}".format), "% positive": t[f"{period}_w_positive"].map("{:.0%}".format),
            "worst DD": t[f"{period}_w_worst_dd"].map("{:.1%}".format),
        }).to_string(index=False))

    pd.set_option("display.width", 260)
    w = SCORE_WEIGHTS
    print(f"SCORE = {w['sortino']} x Sortino + {w['sharpe']} x Sharpe + {w['calmar']} x Calmar (annualised). Left: one "
          "continuous run; right: medians over every 14-day window\n(Calmar annualises a 14-day return, so it is often "
          "large and drives the 14-day score).")
    table("old", f"OLDER PERIOD {idx[t0]:%Y-%m-%d} -> {idx[th]:%Y-%m-%d} (continuous run, then every 14-day window):")
    table("last", f"LAST YEAR {idx[th]:%Y-%m-%d} -> {idx[end]:%Y-%m-%d} (confirmation):")

    cand = t.drop(index="BTC")
    best = cand["old_w_score"].astype(float).idxmax()
    line = "=" * 100
    print(f"\n{line}\nCHOSEN on the older period (best median 14-day SCORE): {best} {labels[best]} "
          f"({cand.loc[best, 'old_w_score']:+.2f} vs A {cand.loc['A', 'old_w_score']:+.2f}, hold BTC "
          f"{t.loc['BTC', 'old_w_score']:+.2f})")
    print(f"  last year: median 14-day SCORE {cand.loc[best, 'last_w_score']:+.2f} (A {cand.loc['A', 'last_w_score']:+.2f}, "
          f"hold BTC {t.loc['BTC', 'last_w_score']:+.2f}); return {cand.loc[best, 'last_return']:+.1%} "
          f"(A {cand.loc['A', 'last_return']:+.1%}, BTC {t.loc['BTC', 'last_return']:+.1%}); worst 14-day drawdown "
          f"{cand.loc[best, 'last_w_worst_dd']:.1%}")
    better = cand.loc[best, "last_w_score"] > cand.loc["A", "last_w_score"]
    print("  -> the improvement holds up in the last year." if better and best != "A"
          else "  -> not confirmed in the last year: treat the differences as noise.")
    print(line)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "momentum_v2.csv")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(13, 6))
    for name, col in (("A", "tab:gray"), (best, "tab:green"), ("BTC", "tab:orange")):
        level, xs, ys = 1.0, [], []
        for eq in curves[name]:
            e = eq / eq.iloc[0] * level
            xs.extend(e.index)
            ys.extend(e.to_numpy())
            level = e.iloc[-1]
        ax.plot(xs, ys, color=col, lw=2 if name == best else 1.3, label=f"{name} {labels[name]} ({level - 1:+.0%})")
    ax.axvline(idx[th], color="k", ls=":", lw=1)
    ax.set_yscale("log")
    ax.axhline(1, color="k", lw=0.5)
    ax.set_ylabel("equity / start (log)")
    ax.set_title("Momentum sleeve: as specified (A) vs chosen variant vs BTC (dotted: last year, restarted)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    png = OUT / "momentum_v2.png"
    fig.savefig(png, dpi=130)
    print(f"chart: {png}")
    if sys.platform == "darwin" and not args.no_open:
        subprocess.run(["open", str(png)], check=False)


if __name__ == "__main__":
    main()
