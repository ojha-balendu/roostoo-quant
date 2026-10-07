"""RSI and MACD on 1-minute bars for the 15 majors, as specified and with BTC hedges.

    python research/20_minute_rsi_macd.py               # last 30 days
    python research/20_minute_rsi_macd.py --days 14

Variants (all start flat with $1M, decisions at each minute's close, trades at that close + half spread):
  RSI as specified      RSI(14) < 30 -> long, > 70 -> short, exit when RSI is back at 50 (all 15 coins)
  RSI + BTC hedge       same on the 14 non-BTC coins, net BTC beta hedged with a BTC position (the hedge is
                        adjusted only when it is more than 5% of equity off, to save orders and fees)
  RSI relative + hedge  RSI of each coin's price relative to BTC (coin/BTC), hedged
  RSI 20/80 + hedge     stricter thresholds, hedged
  MACD as specified     MACD(12,26,9): long while MACD > signal line, short while below (all 15 coins)
  MACD + BTC hedge      same on the 14 non-BTC coins, hedged
Each is run with fees and without (to show what the signal earns before costs). Also reported: the
move in the trade's direction after entries (signal check), orders per day, the most orders needed in
a single minute and how often more than one is needed (Roostoo may allow only one per minute).
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
    macd_states,
    relative_rsi,
    rsi_states,
    signal_check,
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
    ap.add_argument("--warmup-days", type=float, default=20, help="history for betas before the window")
    ap.add_argument("--rsi-period", type=int, default=14)
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
    close = load_panel(cfg.data.processed_dir, cfg.data.market, "1m", universe,
                       start=str((start_ts - pd.Timedelta(days=args.warmup_days)).date())).ffill(limit=5)
    allc = [c for c in MAJORS if c in close.columns]
    coins = [c for c in allc if c != BTC]
    close = close[allc]
    idx = close.index
    a, end = int(idx.searchsorted(start_ts)), len(idx) - 1
    days = (idx[end] - idx[a]).total_seconds() / 86400

    hourly = close.resample("1h").last()
    _, beta_h = residual_returns(hourly, BTC, beta_hours=24 * 14)
    beta_h.index = beta_h.index + pd.Timedelta(hours=1)          # an hour's beta is known when the hour ends
    beta = asof(beta_h[coins], idx).to_numpy()

    lo = max(0, a - 300)                                            # indicators need a few hours of warm-up

    def full(states_slice: np.ndarray, n: int) -> np.ndarray:
        out = np.zeros((len(idx), n), dtype=np.int8)
        out[lo:] = states_slice
        return out

    r_all, r_coins = rsi(close.iloc[lo:], args.rsi_period), rsi(close[coins].iloc[lo:], args.rsi_period)
    rel = relative_rsi(close[coins].iloc[lo:], close[BTC].iloc[lo:], args.rsi_period)
    states = {
        "RSI as specified": (full(rsi_states(r_all), len(allc)), False),
        "RSI + BTC hedge": (full(rsi_states(r_coins), len(coins)), True),
        "RSI relative + hedge": (full(rsi_states(rel), len(coins)), True),
        "RSI 20/80 + hedge": (full(rsi_states(r_coins, 20, 80), len(coins)), True),
        "MACD as specified": (full(macd_states(close.iloc[lo:]), len(allc)), False),
        "MACD + BTC hedge": (full(macd_states(close[coins].iloc[lo:]), len(coins)), True),
    }
    P_all = close[allc].to_numpy(dtype=float)
    P_hedged = close[coins + [BTC]].to_numpy(dtype=float)
    bc = CompositeConfig(capital=args.capital, fee=cfg.fees.taker, spread_bps=cfg.backtest.spread_bps, band=0.0)
    print(f"{len(allc)} majors on 1-minute bars; window {idx[a]:%Y-%m-%d %H:%M} -> {idx[end]:%Y-%m-%d %H:%M} UTC "
          f"({days:.1f} days); ${args.capital:,.0f}; cost per round trip ~{2 * bc.cost * 1e4:.0f} bps\n")

    print("SIGNAL CHECK: average move in the trade's direction after each entry, in bps (a trade needs more than "
          f"~{2 * bc.cost * 1e4:.0f} bps to pay its round trip). Hedged variants: move relative to beta x BTC.")
    checks = []
    for name, (S, hedged) in states.items():
        cl = close[coins] if hedged else close[allc]
        sc = signal_check(S[a:end], cl.iloc[a:end], close[BTC].iloc[a:end], beta[a:end] if hedged else None)
        col = "vs_btc_bps" if hedged else "raw_bps"
        checks.append({"variant": name, "entries": int(sc["entries"].iloc[0]),
                       **{f"after {h}m": f"{v:+.1f}" for h, v in zip(sc["horizon_min"], sc[col], strict=True)},
                       "won after 15m": f"{sc.loc[sc['horizon_min'] == 15, 'hit'].iloc[0]:.0%}"})
    pd.set_option("display.width", 220)
    print(pd.DataFrame(checks).to_string(index=False))

    rows, curves, gross_curves = {}, {}, {}
    for name, (S, hedged) in states.items():
        n = len(coins) if hedged else len(allc)
        unit = bc.gross / (n * 2.2) if hedged else bc.gross / n
        P = P_hedged if hedged else P_all
        log: dict[int, int] = {}
        net = simulate(P, idx, state_decider(S, unit, beta if hedged else None, bc.gross, a, order_log=log),
                       a, end, bc.sim_config())
        free = simulate(P, idx, state_decider(S, unit, beta if hedged else None, bc.gross, a), a, end,
                        bc.sim_config(zero_cost=True))
        opm = np.zeros(end - a, dtype=int)                                  # orders actually sent, every minute
        for tt, k in log.items():
            opm[tt - a] = k
        m = run_metrics(net, args.capital)
        rows[name] = {**m, "free_return": run_metrics(free, args.capital)["return"], "trades_day": net["trades"] / days,
                      "max_per_min": int(opm.max()), "multi_min": float((opm > 1).mean())}
        curves[name], gross_curves[name] = net["equity"], free["equity"]
    btc_t = np.zeros(len(allc))
    btc_t[allc.index(BTC)] = bc.gross
    for name, dec in (("hold BTC", lambda t, cur: btc_t if t == a else cur),
                      ("never trade", lambda t, cur: np.zeros(len(allc)))):
        res = simulate(P_all, idx, dec, a, end, bc.sim_config())
        rows[name] = {**run_metrics(res, args.capital), "free_return": np.nan, "trades_day": res["trades"] / days,
                      "max_per_min": 1 if name == "hold BTC" else 0, "multi_min": 0.0}
        curves[name] = res["equity"]

    def ratio(v: float) -> str:
        return f"{v:+.2f}" if np.isfinite(v) else "n/a"

    t = pd.DataFrame(rows).T.sort_values("score", ascending=False, na_position="last")
    print(f"\nRESULTS over the last {days:.0f} days, sorted by SCORE = 0.4 Sortino + 0.3 Sharpe + 0.3 Calmar:")
    print(pd.DataFrame({
        "SCORE": t["score"].map(ratio), "return": t["return"].map("{:+.2%}".format),
        "without fees": t["free_return"].map(lambda v: f"{v:+.2%}" if np.isfinite(v) else ""),
        "Sortino": t["sortino"].map(ratio), "Sharpe": t["sharpe"].map(ratio), "Calmar": t["calmar"].map(ratio),
        "max DD": t["max_dd"].map("{:.2%}".format), "fees": t["fees_pct"].map("{:.1%}".format),
        "orders/day": t["trades_day"].map("{:,.0f}".format), "max orders in 1 min": t["max_per_min"].astype(int),
        "minutes needing >1 order": t["multi_min"].map("{:.1%}".format)}).to_string())
    print("'without fees' = the same trades at zero cost: what the signal itself earned. If it is positive but the\n"
          "real return is not, the idea has an edge that fees remove; if both are negative, it has none.")
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / f"minute_rsi_macd_{args.days:g}d.csv")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        return
    order = list(t.index)
    fig, ax = plt.subplots(figsize=(14, 6.5))
    for i, name in enumerate(order):
        eq = curves[name]
        ax.plot(eq.index, (eq / eq.iloc[0] - 1) * 100, lw=2 if i == 0 else 1.2,
                ls="--" if name in ("hold BTC", "never trade") else "-",
                color="tab:orange" if name == "hold BTC" else ("black" if name == "never trade" else None),
                label=f"{name} (score {ratio(t.loc[name, 'score'])}, {t.loc[name, 'return']:+.1%})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("return % (fees included)")
    ax.set_title(f"Minute RSI / MACD strategies, last {days:.0f} days, ${args.capital:,.0f} each")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.tight_layout()
    p1 = OUT / f"minute_rsi_macd_{args.days:g}d_all.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    names = [n for n in order if n in gross_curves]
    cols, rows_n = 3, int(np.ceil(len(names) / 3))
    fig, axes = plt.subplots(rows_n * 2, cols, figsize=(15, 3.8 * rows_n), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1] * rows_n})
    axes = np.atleast_2d(axes)
    btc_eq = curves["hold BTC"]
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        top, bot = axes[2 * r, c], axes[2 * r + 1, c]
        eq, fe = curves[name], gross_curves[name]
        top.plot(btc_eq.index, (btc_eq / btc_eq.iloc[0] - 1) * 100, color="tab:orange", lw=0.8, alpha=0.5, label="BTC")
        top.plot(fe.index, (fe / fe.iloc[0] - 1) * 100, color="tab:green", lw=1.1, ls="--", label="without fees")
        top.plot(eq.index, (eq / eq.iloc[0] - 1) * 100, color="tab:blue", lw=1.6, label="with fees")
        top.axhline(0, color="k", lw=0.4)
        m = t.loc[name]
        top.set_title(f"{name}\nscore {ratio(m['score'])} | ret {m['return']:+.1%} (no fees {m['free_return']:+.1%}) | "
                      f"{m['trades_day']:,.0f} orders/day", fontsize=8)
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
    fig.suptitle("Each minute strategy: blue with fees, green dashed without fees, orange BTC; red: drawdown", fontsize=11)
    fig.tight_layout()
    p2 = OUT / f"minute_rsi_macd_{args.days:g}d_each.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    print(f"\ncharts: {p1}\n        {p2}")
    if sys.platform == "darwin" and not args.no_open:
        subprocess.run(["open", str(p1), str(p2)], check=False)


if __name__ == "__main__":
    main()
