"""Does volatility management improve Sharpe / Sortino / Calmar over 14-day windows?

    python research/07_vol_managed.py
    python research/07_vol_managed.py --target-vol 0.25 --rebalance-hours 12

Compares a volatility-managed portfolio (exposure = target vol / predicted vol) with the
SAME base portfolio held at a CONSTANT exposure equal to the managed version's average
exposure. Same capital on average; the only difference is timing. Every 14-day window
since the start of the data is scored as a fresh competition (flat start, fees included).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.backtest.windows import evaluate_windows, summarize
from quantbot.config import load_config
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe
from quantbot.strategy.trend_vol import HOURS_PER_YEAR
from quantbot.strategy.vol_managed import VolManaged

BTC = "BTC/USD"
MAJORS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "BNB/USD", "DOGE/USD", "ADA/USD", "LINK/USD",
          "AVAX/USD", "LTC/USD", "DOT/USD", "SUI/USD", "NEAR/USD", "UNI/USD", "AAVE/USD"]
HALFLIVES = (24, 72, 168)
OUT = Path("reports/research")


def paired(a: pd.Series, b: pd.Series) -> dict[str, float]:
    d = (a - b).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(d)
    half = n // 2
    return {"wins": float((d > 0).mean()) if n else np.nan,
            "t": float(d.mean() / (d.std() / np.sqrt(n))) if n > 2 and d.std() > 0 else np.nan,
            "first_half": float(d.iloc[:half].mean()) if half else np.nan,
            "second_half": float(d.iloc[half:].mean()) if n - half else np.nan, "n": n}


def forecast_quality(closes: pd.DataFrame, halflife: float) -> float:
    """Correlation of log predicted vol with log realised vol over the next 24 hours (BTC)."""
    pred = VolManaged(assets=[BTC], halflife_hours=halflife).predicted_vol(closes[[BTC]])  # at known time T
    r = np.log(closes[BTC]).diff()          # r[t]: return between known times t and t + 1h (hourly bars)
    realised = r.rolling(24).std().shift(-23) * np.sqrt(HOURS_PER_YEAR)  # at T: std of r[T .. T+23]
    df = pd.concat([np.log(pred), np.log(realised)], axis=1, join="inner").replace(
        [np.inf, -np.inf], np.nan).dropna()
    return float(df.iloc[:, 0].corr(df.iloc[:, 1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-vol", type=float, default=None,
                    help="annual vol target; default = 0.5 x median predicted vol of the first 180 days")
    ap.add_argument("--window-days", type=float, default=14)
    ap.add_argument("--rebalance-hours", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.005)
    ap.add_argument("--spread-bps", type=float, default=2.0)
    ap.add_argument("--min-win-rate", type=float, default=0.60)
    ap.add_argument("--min-t", type=float, default=2.0)
    args = ap.parse_args()

    cfg = load_config()
    universe = {p: s for p, s in load_universe(cfg.data.universe_path).items() if p in MAJORS}
    closes = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", universe, start=cfg.data.start).ffill(limit=3)
    if closes.empty or BTC not in closes:
        raise SystemExit("no hourly data: run  quantbot download")
    first_window = closes.index.min() + pd.Timedelta(days=45)
    print(f"{closes.shape[1]} majors, hourly, windows of {args.window_days:g} days from {first_window:%Y-%m-%d} "
          f"to {closes.index.max():%Y-%m-%d}; rebalance every {args.rebalance_hours}h\n")

    print("Is volatility predictable? Correlation of predicted vs next-24h realised BTC volatility (log scale):")
    for hl in HALFLIVES:
        print(f"  half-life {hl:>3}h: {forecast_quality(closes, hl):+.2f}")
    print("  (0 = no information, 1 = perfect; above ~0.5 means volatility clusters strongly)\n")

    bar = closes.index.to_series().diff().median()
    bucket = pd.Series(closes.index + bar).dt.floor(f"{args.rebalance_hours}h")
    trade_on = (bucket != bucket.shift(1)).to_numpy()
    bt = dict(initial_capital=cfg.backtest.initial_capital, fees=cfg.fees, long_fee=cfg.backtest.long_fee,
              spread_bps=args.spread_bps, delay_bars=0, min_trade_usd=cfg.execution.min_trade_usd,
              min_close_usd=cfg.execution.min_close_usd, threshold=args.threshold,
              cash_buffer=cfg.execution.cash_buffer)

    rows = []
    for uname, assets in (("BTC", [BTC]), ("15 majors", [a for a in MAJORS if a in closes.columns])):
        target = args.target_vol
        if target is None:  # scale chosen from the first 180 days only (before almost all windows)
            pv = VolManaged(assets=assets, halflife_hours=72).predicted_vol(closes)
            target = 0.5 * float(pv.loc[pv.index < closes.index.min() + pd.Timedelta(days=180)].median())
        print(f"{uname}: target volatility {target:.0%}")
        for hl in HALFLIVES:
            vm = VolManaged(assets=assets, target_vol=target, halflife_hours=hl).generate_weights(closes)
            gross = vm.abs().sum(axis=1)
            capped = float((gross.loc[gross.index >= first_window] >= 0.95 - 1e-9).mean())
            avg_exposure = float(gross.loc[gross.index >= first_window].mean())
            const = vm.div(gross.where(gross > 0), axis=0).fillna(0.0) * avg_exposure
            r_vm = evaluate_windows(closes, vm, window_days=args.window_days, start=first_window,
                                    trade_on=trade_on, **bt)
            r_c = evaluate_windows(closes, const, window_days=args.window_days, start=first_window,
                                   trade_on=trade_on, **bt)
            s_vm, s_c = summarize(r_vm), summarize(r_c)
            ps, pso, pc = (paired(r_vm[k], r_c[k]) for k in ("sharpe", "sortino", "calmar"))
            rows.append({
                "universe": uname, "half-life": f"{hl}h", "avg exposure": avg_exposure, "capped": capped,
                "sharpe": (s_vm["median_sharpe"], s_c["median_sharpe"]),
                "sortino": (s_vm["median_sortino"], s_c["median_sortino"]),
                "calmar": (s_vm["median_calmar"], s_c["median_calmar"]),
                "worst dd": (s_vm["worst_max_dd"], s_c["worst_max_dd"]),
                "median return": (s_vm["median_return"], s_c["median_return"]),
                "fees": (s_vm["avg_fees"], s_c["avg_fees"]),
                "sharpe_wins": ps["wins"], "sharpe_t": ps["t"], "sharpe_halves": (ps["first_half"], ps["second_half"]),
                "sortino_wins": pso["wins"], "calmar_wins": pc["wins"], "windows": ps["n"],
            })

    pd.set_option("display.width", 220)
    tbl = pd.DataFrame({
        "universe": [r["universe"] for r in rows],
        "half-life": [r["half-life"] for r in rows],
        "avg exp": [f"{r['avg exposure']:.2f}" for r in rows],
        "at cap": [f"{r['capped']:.0%}" for r in rows],
        "Sharpe vm/const": [f"{a:.2f} / {b:.2f}" for a, b in (r["sharpe"] for r in rows)],
        "Sortino vm/const": [f"{a:.2f} / {b:.2f}" for a, b in (r["sortino"] for r in rows)],
        "Calmar vm/const": [f"{a:.1f} / {b:.1f}" for a, b in (r["calmar"] for r in rows)],
        "worst DD vm/const": [f"{a:.1%} / {b:.1%}" for a, b in (r["worst dd"] for r in rows)],
        "med return vm/const": [f"{a:+.1%} / {b:+.1%}" for a, b in (r["median return"] for r in rows)],
        "wins S/So/C": [f"{r['sharpe_wins']:.0%} {r['sortino_wins']:.0%} {r['calmar_wins']:.0%}" for r in rows],
        "Sharpe diff t": [f"{r['sharpe_t']:+.1f}" for r in rows],
    })
    print()
    print(f"Medians across {rows[0]['windows']} windows (vm = volatility-managed, const = same average exposure held constant):")
    print(tbl.to_string(index=False))

    line = "=" * 100
    print(f"\n{line}\nDOES VOLATILITY MANAGEMENT HELP?  Rules: beats constant exposure on Sharpe in >= "
          f"{args.min_win_rate:.0%} of windows, t >= {args.min_t:g} on the window-by-window Sharpe difference, "
          "and better in both halves of the history.")
    if any(r["capped"] > 0.5 for r in rows):
        print("  NOTE: some variants sit at the 95% cap most of the time, so they barely differ from constant "
              "exposure. Use a lower --target-vol for a meaningful test.")
    winners = [r for r in rows if r["sharpe_wins"] >= args.min_win_rate and r["sharpe_t"] >= args.min_t
               and r["sharpe_halves"][0] > 0 and r["sharpe_halves"][1] > 0]
    for r in sorted(winners, key=lambda r: -r["sharpe_t"]):
        print(f"  YES  {r['universe']}, half-life {r['half-life']}: higher Sharpe in {r['sharpe_wins']:.0%} of windows "
              f"(t {r['sharpe_t']:.1f}), Sortino in {r['sortino_wins']:.0%}, Calmar in {r['calmar_wins']:.0%}; "
              f"worst drawdown {r['worst dd'][0]:.1%} vs {r['worst dd'][1]:.1%}")
    if not winners:
        print("  NO variant passes all three rules.")
        for r in sorted(rows, key=lambda r: -(r["sharpe_t"] if np.isfinite(r["sharpe_t"]) else -99))[:3]:
            print(f"  closest: {r['universe']}, half-life {r['half-life']}: Sharpe wins {r['sharpe_wins']:.0%}, "
                  f"t {r['sharpe_t']:+.1f}, halves {r['sharpe_halves'][0]:+.2f} / {r['sharpe_halves'][1]:+.2f}")
    print(line)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "vol_managed_summary.csv", index=False)
    print(f"saved {OUT}/vol_managed_summary.csv")


if __name__ == "__main__":
    main()
