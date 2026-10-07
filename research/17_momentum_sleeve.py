"""H3: relative momentum sleeve: long the 5 strongest coins, short the 5 weakest by 16-day return.

    python research/17_momentum_sleeve.py
    python research/17_momentum_sleeve.py --lookback-days 10 --k 4

Part 1 - the long/short rules test (as research/06): rank by trailing return every 1, 3 or 7 days,
         beta-neutral, costs on turnover, research period then the untouched last year.
Part 2 - the sleeve itself as a portfolio: rebalanced daily at 00:00 UTC (only changed names trade),
         BTC hedge, 95% gross, $1M; every 14-day window as a separate competition, vs holding BTC,
         plus orders per day.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.analysis.xsection import XSRules, eligibility, judge, residual_returns, signal_periods
from quantbot.backtest.composite_bt import CompositeConfig, run_metrics, simulate, topk_weights
from quantbot.backtest.windows import summarize
from quantbot.config import load_config
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe

BTC = "BTC/USD"
OUT = Path("reports/research")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback-days", type=float, default=16)
    ap.add_argument("--k", type=int, default=5, help="coins per side")
    ap.add_argument("--holds", default="24,72,168", help="rules test: holding periods in hours")
    ap.add_argument("--holdout-days", type=float, default=365)
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--capital", type=float, default=1_000_000)
    args = ap.parse_args()
    rules = XSRules()
    lb = int(args.lookback_days * 24)

    cfg = load_config()
    universe = load_universe(cfg.data.universe_path)
    close = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", universe, start=cfg.data.start).ffill(limit=3)
    qv = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", universe, start=cfg.data.start, field="quote_volume")
    resid, beta = residual_returns(close, BTC)
    coins = list(resid.columns)
    eligible = eligibility(close, qv)[coins]
    sig = np.log(close[coins]) - np.log(close[coins]).shift(lb)            # trailing return, known at the bar close
    end = close.index.max()
    start = close.index.min() + pd.Timedelta(days=max(45, args.lookback_days + 5))
    holdout = end - pd.Timedelta(days=args.holdout_days)
    split = start + (holdout - start) / 2
    print(f"{len(coins)} coins; long {args.k} strongest / short {args.k} weakest by {args.lookback_days:g}-day return; "
          f"research {start:%Y-%m-%d} -> {holdout:%Y-%m-%d}, holdout -> {end:%Y-%m-%d}\n")

    rows = []
    for H in [int(x) for x in args.holds.split(",")]:
        p = signal_periods(sig, resid, beta, eligible, horizon=H, k=args.k, cost_one_way_bps=args.cost_bps / 2,
                           start=start, end=holdout)
        d = 1 if p["spread_bps"].mean() > 0 else -1
        rows.append({"hold": H, "direction": d, **judge(p, d, split, rules, 14 * 24 / H)})
    g = pd.DataFrame(rows)
    print("PART 1 - rules test on the research period (bps per holding period, per $1 long + $1 short):")
    print(pd.DataFrame({
        "rebalance": g["hold"].map(lambda h: f"every {h // 24}d"),
        "data prefers": np.where(g["direction"] > 0, "momentum", "reversal"),
        "IC": g["ic"].map("{:+.3f}".format), "gross bps": g["gross_bps"].map("{:+.1f}".format),
        "cost bps": g["cost_bps"].map("{:.1f}".format), "net bps": g["net_bps"].map("{:+.1f}".format),
        "t": g["t"].map("{:+.1f}".format),
        "halves": [f"{a:+.0f} / {b:+.0f}" for a, b in zip(g["net_first_half"], g["net_second_half"], strict=True)],
        "passes": np.where(g["tradeable"], "YES", "no")}).to_string(index=False))
    for _, w in g[g["tradeable"]].iterrows():
        p = signal_periods(sig, resid, beta, eligible, horizon=int(w["hold"]), k=args.k,
                           cost_one_way_bps=args.cost_bps / 2, start=holdout, end=end)
        h = judge(p, int(w["direction"]), holdout + (end - holdout) / 2, rules, 14 * 24 / w["hold"])
        ok = h["net_bps"] > 0 and h["t"] >= rules.holdout_min_t
        print(f"  holdout, rebalance every {w['hold'] // 24}d: net {h['net_bps']:+.1f} bps (t {h['t']:.1f}) -> "
              f"{'CONFIRMED' if ok else 'NOT CONFIRMED'}")

    # PART 2: the sleeve as specified (momentum), daily rebalance at 00:00 UTC
    bc = CompositeConfig(capital=args.capital, fee=cfg.fees.taker)
    P = close[coins + [BTC]].to_numpy(dtype=float)
    S, E, B = sig.to_numpy(), eligible.to_numpy(dtype=bool), beta[coins].to_numpy()
    closes_at = close.index + pd.Timedelta(hours=1)
    daily = np.asarray((closes_at.hour == 0) & (closes_at.minute == 0))
    idx = close.index

    def sleeve(first):
        return lambda t, cur: topk_weights(S[t], E[t], B[t], args.k, bc.gross) if (daily[t] or t == first) else cur

    def hold_btc(first):
        tgt = np.zeros(P.shape[1])
        tgt[-1] = bc.gross
        return lambda t, cur: tgt if t == first else cur

    t0, th = int(idx.searchsorted(start)), int(idx.searchsorted(holdout))
    print("\nPART 2 - the sleeve as a $1M portfolio (momentum, daily rebalance, BTC-hedged):")
    for name, (a, b) in {"older period": (t0, th), "last year": (th, len(idx) - 1)}.items():
        res = {k: simulate(P, idx, f(a), a, b, bc.sim_config()) for k, f in (("sleeve", sleeve), ("hold BTC", hold_btc))}
        days = (idx[b] - idx[a]).total_seconds() / 86400
        for k, r in res.items():
            m = run_metrics(r, args.capital)
            print(f"  {name:<12} {k:<9} score {m['score']:+6.2f}  return {m['return']:+7.1%}  Sharpe {m['sharpe']:5.2f}  "
                  f"Sortino {m['sortino']:5.2f}  Calmar {m['calmar']:5.2f}  max DD {m['max_dd']:6.1%}  fees {m['fees_pct']:5.1%}"
                  + (f"  orders/day {r['trades'] / days:.1f}" if k == "sleeve" else ""))
    step = 14 * 24
    wins = {"sleeve": [], "hold BTC": []}
    for a in range(t0, len(idx) - 1 - step, step):
        for k, f in (("sleeve", sleeve), ("hold BTC", hold_btc)):
            r = simulate(P, idx, f(a), a, a + step, bc.sim_config())
            m = run_metrics(r, args.capital)
            wins[k].append({"return": m["return"], "max_dd": m["max_dd"], "sharpe": m["sharpe"], "sortino": m["sortino"],
                            "calmar": m["calmar"], "score": m["score"], "fees": r["fees"], "trades": r["trades"], "avg_gross": r["avg_gross"],
                            "beta_btc": np.nan})
    w = {k: pd.DataFrame(v) for k, v in wins.items()}
    summ = pd.DataFrame({k: summarize(v, versus=w["hold BTC"]) for k, v in w.items()})
    keep = ["windows", "median_return", "mean_return", "worst_return", "pct_positive", "worst_max_dd",
            "median_sharpe", "median_sortino", "median_calmar", "median_score", "avg_fees", "pct_beats_benchmark"]
    pct = {"median_return", "mean_return", "worst_return", "pct_positive", "worst_max_dd", "pct_beats_benchmark"}
    out = summ.loc[keep].astype(object)
    for r in keep:
        for c in out.columns:
            v = summ.loc[r, c]
            out.loc[r, c] = f"{v:.1%}" if r in pct else (f"{v:,.0f}" if r in ("windows", "avg_fees") else f"{v:.2f}")
    print("\n  every 14-day window since the start, as a separate $1M competition:")
    print("  " + out.rename(index={"pct_beats_benchmark": "pct_beats_btc"}).to_string().replace("\n", "\n  "))
    OUT.mkdir(parents=True, exist_ok=True)
    g.to_csv(OUT / "momentum_sleeve_rules.csv", index=False)


if __name__ == "__main__":
    main()
