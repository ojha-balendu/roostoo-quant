"""Backtest the ETH/BTC spread trade on 1-minute candles.

    python research/03_pairs_btc_eth.py                      # last 15 days, defaults
    python research/03_pairs_btc_eth.py --days 180           # longer test
    python research/03_pairs_btc_eth.py --z-entry 2.5 --fee-multiple 3 --lookback-hours 48

Prints: whether the spread mean-reverts (half-life), every trade with its exit reason and
P&L after fees, and summary metrics versus simply holding BTC.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.backtest.engine import run_backtest
from quantbot.config import load_config
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe
from quantbot.strategy.pairs_spread import PairsSpread, known_time_bars

OUT = Path("reports/research")


def half_life_bars(spread: pd.Series) -> tuple[float, float]:
    """AR(1) fit  d(spread) = a + b * spread[t-1]. Returns (b, half-life in bars); b >= 0 means no reversion."""
    s = spread.dropna()
    ds, lag = s.diff().iloc[1:], s.shift(1).iloc[1:]
    b = np.polyfit(lag.to_numpy(), ds.to_numpy(), 1)[0]
    return float(b), (float(np.log(0.5) / np.log(1 + b)) if -1 < b < 0 else float("inf"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=15)
    ap.add_argument("--leg-a", default="ETH/USD")
    ap.add_argument("--leg-b", default="BTC/USD")
    ap.add_argument("--bar-minutes", type=int, default=5)
    ap.add_argument("--lookback-hours", type=float, default=72)
    ap.add_argument("--z-entry", type=float, default=2.0)
    ap.add_argument("--z-exit", type=float, default=0.0)
    ap.add_argument("--z-stop", type=float, default=4.0)
    ap.add_argument("--max-hold-hours", type=float, default=48)
    ap.add_argument("--fee-multiple", type=float, default=2.0)
    ap.add_argument("--spread-bps", type=float, default=2.0)
    args = ap.parse_args()

    cfg = load_config()
    universe = load_universe(cfg.data.universe_path)
    pair = {p: universe[p] for p in (args.leg_a, args.leg_b)}
    closes = load_panel(cfg.data.processed_dir, cfg.data.market, "1m", pair).ffill().dropna()
    if closes.empty:
        raise SystemExit("no 1m data for these pairs: run  quantbot download --interval 1m --start <date>")
    end = closes.index.max()
    test_start = end - pd.Timedelta(days=args.days)
    closes = closes.loc[closes.index >= test_start - pd.Timedelta(hours=args.lookback_hours + 24)]

    round_trip = 2 * cfg.fees.taker  # long leg: buy + sell; short leg: short_open + short_close (same 0.1% each)
    strat = PairsSpread(
        leg_a=args.leg_a, leg_b=args.leg_b, bar_minutes=args.bar_minutes, lookback_hours=args.lookback_hours,
        z_entry=args.z_entry, z_exit=args.z_exit, z_stop=args.z_stop, max_hold_hours=args.max_hold_hours,
        fee_multiple=args.fee_multiple, round_trip_fee=round_trip, spread_cost=args.spread_bps / 1e4,
    )
    sig = strat.signals(closes)
    weights = strat.generate_weights(closes)

    # --- does the spread mean-revert in the test window? -----------------------------
    test_sig = sig.loc[sig.index > test_start]
    px = known_time_bars(closes, args.bar_minutes).loc[test_sig.index]
    beta_fixed = test_sig["beta"].median()
    fixed_spread = np.log(px[args.leg_a]) - beta_fixed * np.log(px[args.leg_b])
    b, hl = half_life_bars(fixed_spread)
    print(f"{args.leg_a} vs {args.leg_b}, {args.bar_minutes}m bars, test {test_start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M}")
    print(f"hedge ratio beta: median {beta_fixed:.2f}, range {test_sig['beta'].min():.2f}-{test_sig['beta'].max():.2f}")
    hl_txt = f"{hl * args.bar_minutes / 60:.1f} hours" if np.isfinite(hl) else "none (no mean reversion)"
    print(f"spread mean reversion: AR coefficient {b:+.4f}, half-life {hl_txt}")
    print(f"time with |z| >= {args.z_entry}: {(test_sig['z'].abs() >= args.z_entry).mean():.1%}; "
          f"round-trip cost per $1 of {args.leg_a.split('/')[0]} leg: {test_sig['cost'].median():.3%}, "
          f"entry needs edge >= {args.fee_multiple * test_sig['cost'].median():.3%}")

    # --- backtest: trade only when the target changes (no leg rebalancing mid-trade) -----
    test = closes.index >= test_start
    c, w = closes.loc[test], weights.loc[test]
    trade_on = (w.diff().abs().sum(axis=1) > 0).shift(1, fill_value=False).to_numpy()
    res = run_backtest(
        c, w, initial_capital=cfg.backtest.initial_capital, fees=cfg.fees, long_fee="taker",
        spread_bps=args.spread_bps, delay_bars=1, min_trade_usd=cfg.execution.min_trade_usd,
        min_close_usd=cfg.execution.min_close_usd, threshold=0.0, cash_buffer=cfg.execution.cash_buffer,
        trade_on=trade_on,
    )
    eq = res.equity

    # --- per-trade P&L (from equity at the execution bars) -------------------------------
    trades = []
    ev = test_sig[test_sig["event"] != ""]
    open_row = None
    for t, row in ev.iterrows():
        exec_i = min(c.index.searchsorted(t), len(c) - 1)  # decision known at t executes on bar t
        if row["event"] == "enter":
            open_row = (t, exec_i, "short spread" if row["position"] < 0 else "long spread",
                        row["z"], row["edge"], row["cost"])
        elif open_row is not None:
            t0, i0, side, z0, edge0, cost0 = open_row
            pnl = eq.iloc[min(exec_i + 1, len(eq) - 1)] - eq.iloc[i0]
            trades.append({"entry": t0, "side": side, "z_in": z0, "edge_%": edge0 * 100, "cost_%": cost0 * 100,
                           "exit": row["event"], "hours": (t - t0).total_seconds() / 3600, "pnl_$": pnl})
            open_row = None
    if open_row is not None:
        t0, i0, side, z0, edge0, cost0 = open_row
        trades.append({"entry": t0, "side": side, "z_in": z0, "edge_%": edge0 * 100, "cost_%": cost0 * 100,
                       "exit": "still open", "hours": (end - t0).total_seconds() / 3600,
                       "pnl_$": eq.iloc[-1] - eq.iloc[i0]})
    tdf = pd.DataFrame(trades)
    pd.set_option("display.width", 160)
    print(f"\nTrades ({len(tdf)}):")
    if len(tdf):
        print(tdf.assign(entry=tdf["entry"].dt.strftime("%m-%d %H:%M")).round(3).to_string(index=False))
        closed = tdf[tdf["exit"] != "still open"]
        if len(closed):
            print(f"\nclosed trades: {len(closed)}, win rate {(closed['pnl_$'] > 0).mean():.0%}, "
                  f"avg ${closed['pnl_$'].mean():,.0f}, best ${closed['pnl_$'].max():,.0f}, "
                  f"worst ${closed['pnl_$'].min():,.0f}")
            print("exits by reason: " + ", ".join(f"{k} {v}" for k, v in closed["exit"].value_counts().items()))

    btc = run_backtest(c, pd.DataFrame(0.0, index=c.index, columns=c.columns).assign(**{args.leg_b: 0.95}),
                       fees=cfg.fees, spread_bps=args.spread_bps, threshold=0.0, delay_bars=0,
                       trade_on=np.r_[True, np.zeros(len(c) - 1, dtype=bool)])
    m, mb = res.metrics, btc.metrics
    print(f"\n{'':14}{'return':>9}{'max DD':>9}{'fees $':>9}")
    print(f"{'pairs trade':14}{m['total_return']:>9.2%}{m['max_drawdown']:>9.2%}{m['fees_paid']:>9,.0f}")
    print(f"{'hold ' + args.leg_b.split('/')[0]:14}{mb['total_return']:>9.2%}{mb['max_drawdown']:>9.2%}{mb['fees_paid']:>9,.0f}")
    hourly = pd.concat([eq.resample("1h").last().pct_change(), c[args.leg_b].resample("1h").last().pct_change()],
                       axis=1).dropna()
    if len(hourly) > 10:
        print(f"correlation of pairs-trade hourly returns with {args.leg_b.split('/')[0]}: "
              f"{hourly.iloc[:, 0].corr(hourly.iloc[:, 1]):+.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    tdf.to_csv(OUT / "pairs_trades.csv", index=False)
    eq.to_csv(OUT / "pairs_equity.csv")
    print(f"\nsaved {OUT}/pairs_trades.csv and pairs_equity.csv")


if __name__ == "__main__":
    main()
