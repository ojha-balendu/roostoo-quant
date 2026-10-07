"""Combined portfolio: momentum C + BTC trend + RSI 15/85, at several capital splits, on 1-minute bars.

    python research/24_combined_portfolio.py
    python research/24_combined_portfolio.py --allocations "0.5/0.3/0.2,0.6/0.4/0" --rsi-low 15 --rsi-high 85 --rsi-exit 55

Sleeves (each a fixed share of total equity; the account trades only the NET of all sleeves):
  momentum C   long 5 / short 5 by residual risk-adjusted 16-day momentum, rank buffer (top/bottom 10),
               all eligible coins, rebalanced daily at 00:00 UTC, BTC-hedged (research/18 variant C)
  BTC trend    long BTC only while its 3/7/14-day trend is up and BTC is above its 7-day average, 30%
               volatility target, flat otherwise; checked hourly, re-sized only on >10% changes
  RSI          hourly RSI(14) recomputed every minute on the 15 majors: long < low, short > high, exit at exit
Allocations "C/BTC/RSI" are shares of equity. Results: every 14-day window inside the 1-minute history (each
a fresh competition), then the last 30 days. Ranked by median 14-day score (0.4 Sortino + 0.3 Sharpe + 0.3 Calmar).
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.analysis.xsection import eligibility, residual_returns
from quantbot.backtest.combined import (
    BtcTrendSleeve,
    HoldSleeve,
    MomentumSleeve,
    RsiSleeve,
    last_closed_row,
    simulate_combined,
)
from quantbot.backtest.composite_bt import CompositeConfig, run_metrics
from quantbot.backtest.minute_signals import rsi_states, sliding_bar_rsi
from quantbot.backtest.momentum import momentum_inputs
from quantbot.backtest.windows import summarize
from quantbot.config import load_config
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe
from quantbot.rl.sim import simple_returns
from quantbot.strategy.trend_vol import TrendVolTarget

BTC = "BTC/USD"
MAJORS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "BNB/USD", "DOGE/USD", "ADA/USD", "LINK/USD",
          "AVAX/USD", "LTC/USD", "DOT/USD", "SUI/USD", "NEAR/USD", "UNI/USD", "AAVE/USD"]
DEFAULT_ALLOCATIONS = "1/0/0,0/1/0,0/0/1,0.5/0.3/0.2,0.6/0.2/0.2,0.4/0.4/0.2,0.6/0.4/0,0.7/0.3/0"
OUT = Path("reports/research")


def momentum_c():
    spec = importlib.util.spec_from_file_location("momentum_v2", Path(__file__).with_name("18_momentum_v2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.variants()["C"][1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allocations", default=DEFAULT_ALLOCATIONS, help="comma-separated C/BTC/RSI shares")
    ap.add_argument("--rsi-low", type=float, default=15)
    ap.add_argument("--rsi-high", type=float, default=85)
    ap.add_argument("--rsi-exit", type=float, default=55)
    ap.add_argument("--window-days", type=float, default=14)
    ap.add_argument("--last-days", type=float, default=30)
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--config", default="config/competition.yaml")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    allocs = [tuple(float(v) for v in a.split("/")) for a in args.allocations.split(",")]

    cfg = load_config(args.config)
    universe = load_universe(cfg.data.universe_path)
    probe = load_panel(cfg.data.processed_dir, cfg.data.market, "1m", {BTC: universe[BTC]})
    if probe.empty:
        raise SystemExit("no 1-minute data: run  quantbot download --interval 1m --start <date>")
    m_start, m_end = probe.index.min(), probe.index.max()

    # hourly data (from 60 days before the minute data starts) for momentum, eligibility, betas, BTC trend
    h_from = str((m_start - pd.Timedelta(days=60)).date())
    hc = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", universe, start=h_from).ffill(limit=3)
    hq = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", universe, start=h_from, field="quote_volume")
    resid, beta = residual_returns(hc, BTC)
    print("loading 1-minute data for all coins (this takes a moment) ...", flush=True)
    m1 = load_panel(cfg.data.processed_dir, cfg.data.market, "1m", universe, start=str(m_start.date()))
    coins = [c for c in resid.columns if c in m1.columns]
    assets = coins + [BTC]
    idx = pd.date_range(m1.index.min(), m1.index.max(), freq="1min")
    close = m1[assets].reindex(idx).ffill(limit=60)
    P = close.to_numpy(dtype=float)
    R, TRADABLE = simple_returns(P), np.isfinite(P)                  # computed once for every simulation
    del m1
    btc_col = len(coins)

    # momentum C on hourly rows
    mc = momentum_c()
    elig_h = eligibility(hc, hq)[coins]
    x = {k: v.reindex(columns=coins).to_numpy(dtype=float)
         for k, v in momentum_inputs(hc, resid[coins], elig_h, mc).items()}
    rets_h = simple_returns(hc[coins + [BTC]].to_numpy(dtype=float))
    hour_row = last_closed_row(hc.index + pd.Timedelta(hours=1), idx)
    closes_at = idx + pd.Timedelta(minutes=1)
    midnight = np.asarray((closes_at.hour == 0) & (closes_at.minute == 0))
    hour_end = np.asarray(closes_at.minute == 0)

    # BTC trend: long-only, regime-filtered, 30% vol target (hourly)
    trend = TrendVolTarget(assets=[BTC], lookbacks_hours=[72, 168, 336], target_vol=0.30, max_gross=0.95,
                           max_weight=0.95, long_in_bear=0.0, short_in_bull=0.0, regime_ma_hours=168,
                           regime_band=0.01)
    btc_target = trend.generate_weights(hc[[BTC]])[BTC].clip(lower=0.0).to_numpy()

    # RSI on the 15 majors (minute bars)
    majors = [c for c in MAJORS if c in assets]
    rsi_cols = np.array([assets.index(c) for c in majors])
    states = rsi_states(sliding_bar_rsi(close[majors], 14, 60), args.rsi_low, args.rsi_high, args.rsi_exit)
    rsi_unit = 0.95 / len(majors)

    def sleeves(alloc):
        c, b, r = alloc
        out = []
        if c > 0:
            out.append(MomentumSleeve(c, mc, x, beta[coins].to_numpy(), elig_h.to_numpy(dtype=bool), rets_h,
                                      hour_row, midnight, np.arange(len(coins)), btc_col))
        if b > 0:
            out.append(BtcTrendSleeve(b, btc_target, hour_row, hour_end, btc_col))
        if r > 0:
            out.append(RsiSleeve(r, states, rsi_cols, rsi_unit))
        return out

    hold = np.zeros(len(assets))
    hold[btc_col] = 0.95
    bc = CompositeConfig(capital=args.capital, fee=cfg.fees.taker, spread_bps=cfg.backtest.spread_bps, band=0.0)
    sim_cfg = bc.sim_config()
    first = int(idx.searchsorted(max(m_start + pd.Timedelta(days=3), idx[0])))
    step = int(args.window_days * 1440)
    starts = list(range(first, len(idx) - 1 - step + 1, step))
    last_a = int(idx.searchsorted(m_end - pd.Timedelta(days=args.last_days)))
    print(f"{len(coins)} coins for momentum, {len(majors)} majors for RSI ({args.rsi_low:g}/{args.rsi_high:g}, exit "
          f"{args.rsi_exit:g}); 1-minute history {idx[0]:%Y-%m-%d} -> {idx[-1]:%Y-%m-%d}: {len(starts)} windows of "
          f"{args.window_days:g} days, plus the last {args.last_days:g} days; ${args.capital:,.0f}\n", flush=True)

    candidates = {f"C {c:.0%} / BTC {b:.0%} / RSI {r:.0%}": (lambda a=(c, b, r): sleeves(a)) for c, b, r in allocs}
    candidates["hold BTC (benchmark)"] = lambda: [HoldSleeve(1.0, hold)]
    rows, last_curves = {}, {}
    for name, make in candidates.items():
        wins = []
        for s in starts:
            res = simulate_combined(P, idx, make(), s, s + step, sim_cfg, R, TRADABLE)
            m = run_metrics(res, args.capital)
            wins.append({"return": m["return"], "max_dd": m["max_dd"], "sharpe": m["sharpe"], "sortino": m["sortino"],
                         "calmar": m["calmar"], "score": m["score"], "fees": res["fees"], "trades": res["trades"],
                         "avg_gross": res["avg_gross"], "beta_btc": np.nan})
        w = summarize(pd.DataFrame(wins))
        last = simulate_combined(P, idx, make(), last_a, len(idx) - 1, sim_cfg, R, TRADABLE)
        lm = run_metrics(last, args.capital)
        days = (idx[-1] - idx[last_a]).total_seconds() / 86400
        rows[name] = {**{f"w_{k}": v for k, v in w.items()}, **{f"l_{k}": lm[k] for k in
                      ("score", "return", "sortino", "sharpe", "calmar", "max_dd", "fees_pct")},
                      "l_orders_day": last["trades"] / days, "l_gross": last["avg_gross"]}
        last_curves[name] = last["equity"]
        print(f"  done: {name}", flush=True)

    def r(v: float) -> str:
        return f"{v:+.2f}" if np.isfinite(v) else "n/a"

    t = pd.DataFrame(rows).T.sort_values("w_median_score", ascending=False)
    pd.set_option("display.width", 240)
    print(f"\nEVERY {args.window_days:g}-DAY WINDOW ({len(starts)} windows, each a fresh ${args.capital:,.0f}), sorted by median score:")
    print(pd.DataFrame({
        "median SCORE": t["w_median_score"].map(r), "mean score": t["w_mean_score"].map(r),
        "median return": t["w_median_return"].map("{:+.2%}".format), "mean return": t["w_mean_return"].map("{:+.2%}".format),
        "worst": t["w_worst_return"].map("{:+.1%}".format), "% positive": t["w_pct_positive"].map("{:.0%}".format),
        "worst DD": t["w_worst_max_dd"].map("{:.1%}".format), "median Sortino": t["w_median_sortino"].map(r),
        "median Calmar": t["w_median_calmar"].map(r), "avg fees $": t["w_avg_fees"].map("{:,.0f}".format)}).to_string())
    print(f"\nLAST {args.last_days:g} DAYS ({idx[last_a]:%Y-%m-%d} -> {idx[-1]:%Y-%m-%d}), one continuous run:")
    print(pd.DataFrame({
        "SCORE": t["l_score"].map(r), "return": t["l_return"].map("{:+.2%}".format), "Sortino": t["l_sortino"].map(r),
        "Sharpe": t["l_sharpe"].map(r), "Calmar": t["l_calmar"].map(r), "max DD": t["l_max_dd"].map("{:.2%}".format),
        "fees": t["l_fees_pct"].map("{:.2%}".format), "orders/day": t["l_orders_day"].map("{:,.1f}".format),
        "avg exposure": t["l_gross"].map("{:.2f}".format)}).to_string())
    best = t.drop(index="hold BTC (benchmark)").index[0]
    print(f"\nBest median 14-day score: {best}. With only {len(starts)} windows and {len(allocs)} splits tried, prefer a\n"
          "split that also does well on worst return / worst drawdown and isn't far from its neighbours.")
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "combined_portfolio.csv")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(14, 6.5))
    for name in t.index:
        eq = last_curves[name]
        bench = name.startswith("hold BTC")
        ax.plot(eq.index, (eq / eq.iloc[0] - 1) * 100, lw=2.2 if name == best else (1.4 if bench else 1.0),
                ls="--" if bench else "-", color="tab:orange" if bench else None,
                label=f"{name} (score {r(t.loc[name, 'l_score'])}, {t.loc[name, 'l_return']:+.1%})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("return % (fees included)")
    ax.set_title(f"Combined portfolios, last {args.last_days:g} days, ${args.capital:,.0f} each")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.tight_layout()
    png = OUT / "combined_portfolio.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"chart: {png}")
    if sys.platform == "darwin" and not args.no_open:
        subprocess.run(["open", str(png)], check=False)


if __name__ == "__main__":
    main()
