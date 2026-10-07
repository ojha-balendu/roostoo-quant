"""Hourly RSI with extreme thresholds (default 20/80), recomputed every minute so no extreme is missed.

    python research/21_hourly_rsi.py                          # last 30 days, RSI(14) on hourly bars, 20/80
    python research/21_hourly_rsi.py --low 25 --high 75 --exit 50 --days 14

Variants (15 majors, $1M each, decisions at each minute's close, trades at that close + half spread):
  hourly RSI, checked at hour closes          the standard way: only looks at xx:59
  hourly RSI, checked every minute            RSI over the last 14 hours ending at THIS minute, so an
                                              extreme is acted on as soon as it appears
  ... every minute + BTC hedge                14 non-BTC coins, market exposure offset with BTC
  relative hourly RSI every minute + hedge    RSI of coin/BTC: fires only when a coin moves against the market
  minute RSI + BTC hedge                      the previous script's best idea, for reference
Long below --low, short above --high, exit when RSI returns to --exit (default 50).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.analysis.indicators import rsi
from quantbot.analysis.xsection import residual_returns
from quantbot.backtest.composite_bt import CompositeConfig, run_metrics, simulate
from quantbot.backtest.minute_signals import (
    at_bar_closes,
    holding_minutes,
    rsi_states,
    signal_check,
    sliding_bar_rsi,
    state_decider,
)
from quantbot.config import load_config
from quantbot.data.futures import asof
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe

BTC = "BTC/USD"
MAJORS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "BNB/USD", "DOGE/USD", "ADA/USD", "LINK/USD",
          "AVAX/USD", "LTC/USD", "DOT/USD", "SUI/USD", "NEAR/USD", "UNI/USD", "AAVE/USD"]
OUT = Path("reports/research")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=30)
    ap.add_argument("--low", type=float, default=20)
    ap.add_argument("--high", type=float, default=80)
    ap.add_argument("--exit", type=float, default=50, help="close when RSI returns to this level")
    ap.add_argument("--period", type=int, default=14)
    ap.add_argument("--bar", type=int, default=60, help="RSI bar length in minutes (60 = hourly)")
    ap.add_argument("--warmup-days", type=float, default=20)
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--config", default="config/competition.yaml")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    universe = {p: s for p, s in load_universe(cfg.data.universe_path).items() if p in MAJORS}
    probe = load_panel(cfg.data.processed_dir, cfg.data.market, "1m", {BTC: universe[BTC]})
    if probe.empty:
        raise SystemExit("no 1-minute data: run  quantbot download --interval 1m --start <date>")
    end_ts = probe.index.max()
    start_ts = end_ts - pd.Timedelta(days=args.days)
    raw = load_panel(cfg.data.processed_dir, cfg.data.market, "1m", universe,
                     start=str((start_ts - pd.Timedelta(days=args.warmup_days)).date()))
    full_idx = pd.date_range(raw.index.min(), raw.index.max(), freq="1min")   # gap-free: sliding RSI needs it
    close = raw.reindex(full_idx).ffill(limit=30)
    allc = [c for c in MAJORS if c in close.columns]
    coins = [c for c in allc if c != BTC]
    close = close[allc]
    idx = close.index
    a, end = int(idx.searchsorted(start_ts)), len(idx) - 1
    days = (idx[end] - idx[a]).total_seconds() / 86400
    hourly = close.resample("1h").last()
    _, beta_h = residual_returns(hourly, BTC, beta_hours=24 * 14)
    beta_h.index = beta_h.index + pd.Timedelta(hours=1)
    beta = asof(beta_h[coins], idx).to_numpy()

    lo, hi, ex = args.low, args.high, args.exit
    s_all = sliding_bar_rsi(close, args.period, args.bar)
    s_coins = s_all[coins]
    s_rel = sliding_bar_rsi(close[coins].div(close[BTC], axis=0), args.period, args.bar)
    variants = {
        f"hourly RSI {lo:g}/{hi:g}, checked at hour closes": (rsi_states(at_bar_closes(s_all, args.bar), lo, hi, ex), False),
        f"hourly RSI {lo:g}/{hi:g}, checked every minute": (rsi_states(s_all, lo, hi, ex), False),
        f"hourly RSI {lo:g}/{hi:g} every minute + hedge": (rsi_states(s_coins, lo, hi, ex), True),
        f"relative hourly RSI {lo:g}/{hi:g} + hedge": (rsi_states(s_rel, lo, hi, ex), True),
        f"minute RSI {lo:g}/{hi:g} + hedge": (rsi_states(rsi(close[coins], args.period), lo, hi, ex), True),
    }
    P_all, P_hedged = close[allc].to_numpy(dtype=float), close[coins + [BTC]].to_numpy(dtype=float)
    bc = CompositeConfig(capital=args.capital, fee=cfg.fees.taker, spread_bps=cfg.backtest.spread_bps, band=0.0)
    cost = 2 * bc.cost * 1e4
    print(f"{len(allc)} majors; RSI({args.period}) on {args.bar}-minute bars, long < {lo:g}, short > {hi:g}, exit at "
          f"{ex:g}; window {idx[a]:%Y-%m-%d %H:%M} -> {idx[end]:%Y-%m-%d %H:%M} UTC ({days:.1f} days); "
          f"${args.capital:,.0f}; round trip ~{cost:.0f} bps\n")

    horizons = (15, 60, 240, 720, 1440)
    print(f"SIGNAL CHECK: average move in the trade's direction after each entry, in bps (needs more than ~{cost:.0f} "
          "bps to pay a round trip). Hedged variants: relative to beta x BTC.")
    checks = []
    for name, (S, hedged) in variants.items():
        cl = close[coins] if hedged else close[allc]
        sc = signal_check(S[a:end], cl.iloc[a:end], close[BTC].iloc[a:end], beta[a:end] if hedged else None, horizons)
        col = "vs_btc_bps" if hedged else "raw_bps"
        lab = {15: "15m", 60: "1h", 240: "4h", 720: "12h", 1440: "24h"}
        checks.append({"variant": name, "entries": int(sc["entries"].iloc[0]),
                       **{f"after {lab[h]}": (f"{v:+.1f}" if np.isfinite(v) else "n/a")
                          for h, v in zip(sc["horizon_min"], sc[col], strict=True)},
                       "won after 4h": (lambda h: f"{h:.0%}" if np.isfinite(h) else "n/a")(
                           sc.loc[sc["horizon_min"] == 240, "hit"].iloc[0])})
    pd.set_option("display.width", 230)
    print(pd.DataFrame(checks).to_string(index=False))
    print("Few entries = few trades: the book is mostly flat and its result rests on a handful of signals.")

    rows, curves, free_curves = {}, {}, {}
    for name, (S, hedged) in variants.items():
        n = len(coins) if hedged else len(allc)
        unit = bc.gross / (n * 2.2) if hedged else bc.gross / n
        P = P_hedged if hedged else P_all
        log: dict[int, int] = {}
        net = simulate(P, idx, state_decider(S, unit, beta if hedged else None, bc.gross, a, order_log=log),
                       a, end, bc.sim_config())
        free = simulate(P, idx, state_decider(S, unit, beta if hedged else None, bc.gross, a), a, end,
                        bc.sim_config(zero_cost=True))
        opm = np.zeros(end - a, dtype=int)
        for tt, k in log.items():
            opm[tt - a] = k
        rows[name] = {**run_metrics(net, args.capital), "free_return": run_metrics(free, args.capital)["return"],
                      "orders_day": net["trades"] / days, "max_per_min": int(opm.max()),
                      "multi_min": float((opm > 1).mean()), "hold_h": holding_minutes(S[a:end]) / 60}
        curves[name], free_curves[name] = net["equity"], free["equity"]
    btc_t = np.zeros(len(allc))
    btc_t[allc.index(BTC)] = bc.gross
    res = simulate(P_all, idx, lambda t, cur: btc_t if t == a else cur, a, end, bc.sim_config())
    rows["hold BTC"] = {**run_metrics(res, args.capital), "free_return": np.nan, "orders_day": res["trades"] / days,
                        "max_per_min": 1, "multi_min": 0.0, "hold_h": np.nan}
    curves["hold BTC"] = res["equity"]

    def ratio(v: float) -> str:
        return f"{v:+.2f}" if np.isfinite(v) else "n/a"

    t = pd.DataFrame(rows).T.sort_values("score", ascending=False, na_position="last")
    print(f"\nRESULTS over the last {days:.0f} days, sorted by SCORE = 0.4 Sortino + 0.3 Sharpe + 0.3 Calmar:")
    print(pd.DataFrame({
        "SCORE": t["score"].map(ratio), "return": t["return"].map("{:+.2%}".format),
        "without fees": t["free_return"].map(lambda v: f"{v:+.2%}" if np.isfinite(v) else ""),
        "Sortino": t["sortino"].map(ratio), "Sharpe": t["sharpe"].map(ratio), "Calmar": t["calmar"].map(ratio),
        "max DD": t["max_dd"].map("{:.2%}".format), "fees": t["fees_pct"].map("{:.2%}".format),
        "orders/day": t["orders_day"].map("{:,.1f}".format),
        "avg hold": t["hold_h"].map(lambda v: f"{v:.1f}h" if np.isfinite(v) else ""),
        "max orders in 1 min": t["max_per_min"].astype(int),
        "minutes with >1 order": t["multi_min"].map("{:.2%}".format)}).to_string())
    print("Compare the first two rows to see what checking every minute adds. 'without fees' = the same trades at\n"
          "zero cost. One 30-day window is a single draw: check --days 14 and other windows before relying on it.")
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / f"hourly_rsi_{args.days:g}d.csv")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = [n for n in t.index if n in free_curves]
    cols, rows_n = 3, int(np.ceil(len(names) / 3))
    fig, axes = plt.subplots(rows_n * 2, cols, figsize=(15, 3.8 * rows_n), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1] * rows_n})
    axes = np.atleast_2d(axes)
    btc_eq = curves["hold BTC"]
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        top, bot = axes[2 * r, c], axes[2 * r + 1, c]
        eq, fe = curves[name], free_curves[name]
        top.plot(btc_eq.index, (btc_eq / btc_eq.iloc[0] - 1) * 100, color="tab:orange", lw=0.8, alpha=0.5, label="BTC")
        top.plot(fe.index, (fe / fe.iloc[0] - 1) * 100, color="tab:green", lw=1.1, ls="--", label="without fees")
        top.plot(eq.index, (eq / eq.iloc[0] - 1) * 100, color="tab:blue", lw=1.6, label="with fees")
        top.axhline(0, color="k", lw=0.4)
        m = t.loc[name]
        top.set_title(f"{name}\nscore {ratio(m['score'])} | ret {m['return']:+.1%} (no fees {m['free_return']:+.1%}) | "
                      f"{m['orders_day']:,.1f} orders/day", fontsize=8)
        top.grid(alpha=0.3)
        top.tick_params(labelsize=7)
        if i == 0:
            top.legend(fontsize=7, loc="lower left")
        dd = (eq / eq.cummax() - 1) * 100
        bot.fill_between(dd.index, dd, 0, color="tab:red", alpha=0.4)
        bot.set_ylabel("DD %", fontsize=7)
        bot.grid(alpha=0.3)
        bot.tick_params(labelsize=7)
        bot.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for j in range(len(names), rows_n * cols):
        r, c = divmod(j, cols)
        axes[2 * r, c].axis("off")
        axes[2 * r + 1, c].axis("off")
    fig.suptitle(f"Hourly RSI {lo:g}/{hi:g}: blue with fees, green dashed without fees, orange BTC; red: drawdown",
                 fontsize=11)
    fig.tight_layout()
    png = OUT / f"hourly_rsi_{args.days:g}d.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"\nchart: {png}")
    if sys.platform == "darwin" and not args.no_open:
        subprocess.run(["open", str(png)], check=False)


if __name__ == "__main__":
    main()
