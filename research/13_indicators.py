"""Do classic technical indicators predict which coins outperform? (plus a combined signal)

    python research/13_indicators.py                  # 4-hour bars, holds of 4h / 12h / 1d
    python research/13_indicators.py --bar 1h --horizons 4,12,24
    python research/13_indicators.py --bar 1d --horizons 1,3

For each indicator (RSI, Williams %R, volatility, trends, ATR, MACD, TRIX, CCI, CMO, Ichimoku, KST,
ultimate oscillator, Aroon, stochastic, OBV, ease of movement, Chaikin money flow, accumulation/
distribution, relative volume, taker-buy ratio) and holding period: every period, rank the eligible
coins by the indicator, long one end and short the other (20% each), beta-neutral via BTC, and
measure the next period's return after costs. The data decides whether high values are bought or
sold. Same rules as research/06; the last --holdout-days stay untouched until a rule passes.
A combined signal (average rank of the indicators that look informative) is tested the same way.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.analysis.indicators import compute_all
from quantbot.analysis.xsection import XSRules, eligibility, evaluate, residual_returns, signal_periods
from quantbot.config import load_config
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe

BTC = "BTC/USD"
OUT = Path("reports/research")


def resample_ohlcv(fields: dict[str, pd.DataFrame], rule: str) -> dict[str, pd.DataFrame]:
    r = {k: v.resample(rule) for k, v in fields.items()}
    out = {"open": r["open"].first(), "high": r["high"].max(), "low": r["low"].min(), "close": r["close"].last()}
    for k in ("volume", "quote_volume", "taker_buy_quote"):
        out[k] = r[k].sum(min_count=1)
    return {k: v.iloc[:-1] for k, v in out.items()}  # drop the last, possibly incomplete, bar


def ic_stats(p: pd.DataFrame, direction: int) -> tuple[float, float]:
    ic = direction * p["ic"].dropna()
    return (float(ic.mean()), float(ic.mean() / (ic.std() / np.sqrt(len(ic))))) if len(ic) > 2 and ic.std() > 0 \
        else (np.nan, np.nan)


def redundancy(sigs: dict[str, pd.DataFrame], times: pd.DatetimeIndex) -> pd.DataFrame:
    """Correlation between indicators' cross-sectional ranks (do two indicators rank coins alike?)."""
    flat = {}
    for name, s in sigs.items():
        ranks = s.loc[s.index.isin(times)].rank(axis=1, pct=True) - 0.5
        flat[name] = ranks.to_numpy().ravel()
    return pd.DataFrame(flat).corr()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar", default="4h", help="bar size: 1h, 4h, 1d ...")
    ap.add_argument("--horizons", default="1,3,6", help="holding periods in bars")
    ap.add_argument("--holdout-days", type=float, default=365)
    ap.add_argument("--quantile", type=float, default=0.2)
    ap.add_argument("--cost-bps", type=float, default=25.0, help="round-trip cost per $ traded")
    ap.add_argument("--min-volume", type=float, default=200_000, help="min median USD volume per hour")
    ap.add_argument("--min-net-bps", type=float, default=5.0)
    args = ap.parse_args()
    rules = XSRules(min_net_bps=args.min_net_bps)
    horizons = [int(x) for x in args.horizons.split(",")]
    bar = pd.Timedelta(args.bar)
    bpd = pd.Timedelta(days=1) / bar
    hours_per_bar = bar / pd.Timedelta(hours=1)

    def hold_txt(h: int) -> str:
        d = bar * h
        return f"{d / pd.Timedelta(hours=1):g}h" if d < pd.Timedelta(days=1) else f"{d / pd.Timedelta(days=1):g}d"

    cfg = load_config()
    universe = load_universe(cfg.data.universe_path)
    names = ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote")
    raw = {k: load_panel(cfg.data.processed_dir, cfg.data.market, "1h", universe, start=cfg.data.start, field=k)
           for k in names}
    if raw["close"].empty or BTC not in raw["close"]:
        raise SystemExit("no hourly data: run  quantbot download")
    f = resample_ohlcv(raw, args.bar) if bar != pd.Timedelta(hours=1) else {k: v for k, v in raw.items()}
    c = f["close"]
    end = c.index.max()
    holdout_start = end - pd.Timedelta(days=args.holdout_days)
    research_start = c.index.min() + pd.Timedelta(days=60)

    resid, beta = residual_returns(c, BTC, beta_hours=int(30 * bpd))
    eligible = eligibility(c, f["quote_volume"], min_history_hours=int(30 * bpd),
                           min_usd_per_hour=args.min_volume * hours_per_bar)
    others = list(resid.columns)
    sigs = {k: v[others] for k, v in compute_all(f["open"], f["high"], f["low"], c, f["volume"],
                                                   f["quote_volume"], f["taker_buy_quote"]).items()}
    print(f"{len(others)} coins on {args.bar} bars; research {research_start:%Y-%m-%d} -> {holdout_start:%Y-%m-%d}, "
          f"holdout {holdout_start:%Y-%m-%d} -> {end:%Y-%m-%d} (untouched); {len(sigs)} indicators x "
          f"{len(horizons)} holding periods; cost {args.cost_bps:g} bps per round trip\n")

    def test(name: str, signal: pd.DataFrame, H: int, start, stop, direction: int | None = None) -> dict:
        p = signal_periods(signal, resid, beta, eligible, horizon=H, quantile=args.quantile,
                           cost_one_way_bps=args.cost_bps / 2, start=start, end=stop)
        if p.empty:
            return {}
        d = direction if direction is not None else (1 if p["spread_bps"].mean() > 0 else -1)
        st = evaluate(p, d, start + (stop - start) / 2)
        ic, ic_t = ic_stats(p, d)
        st.update(indicator=name, hold=H, direction=d, ic=ic, ic_t=ic_t, coins=float(p["coins"].median()))
        st["est_14d"] = st["net_bps"] / 1e4 * (14 * bpd / H) * 0.475
        st["pass_profit"] = st["net_bps"] >= rules.min_net_bps
        st["pass_confidence"] = st["t"] >= rules.min_t
        st["pass_sample"] = (st["periods"] >= rules.min_periods and st["periods_first"] >= rules.min_periods_per_half
                             and st["periods_second"] >= rules.min_periods_per_half)
        st["pass_stable"] = st["net_first_half"] > 0 and st["net_second_half"] > 0
        st["tradeable"] = all(st[k] for k in ("pass_profit", "pass_confidence", "pass_sample", "pass_stable"))
        return st

    rows = []
    for H in horizons:
        for name, s in sigs.items():
            st = test(name, s, H, research_start, holdout_start)
            if st:
                rows.append(st)
        # combined: average signed cross-sectional rank of indicators whose rank IC looks informative here
        here = [r for r in rows if r["hold"] == H and np.isfinite(r["ic_t"]) and r["ic_t"] >= 2]
        if len(here) >= 2:
            combo = sum(r["direction"] * (sigs[r["indicator"]].rank(axis=1, pct=True) - 0.5) for r in here) / len(here)
            st = test(f"COMBINED ({len(here)})", combo, H, research_start, holdout_start, direction=1)
            if st:
                st["members"] = ", ".join(r["indicator"] for r in here)
                rows.append(st)
        print(f"  tested holding period {hold_txt(H)}")
    grid = pd.DataFrame(rows).sort_values("t", ascending=False)

    top = grid.head(20)
    show = pd.DataFrame({
        "indicator": top["indicator"], "hold": top["hold"].map(hold_txt),
        "trade": np.where(top["direction"] > 0, "buy high", "buy low"),
        "IC": top["ic"].map("{:+.3f}".format), "IC t": top["ic_t"].map("{:+.1f}".format),
        "gross bps": top["gross_bps"].map("{:+.1f}".format), "cost bps": top["cost_bps"].map("{:.1f}".format),
        "net bps": top["net_bps"].map("{:+.1f}".format), "t": top["t"].map("{:+.1f}".format),
        "halves": [f"{a:+.0f} / {b:+.0f}" for a, b in zip(top["net_first_half"], top["net_second_half"], strict=True)],
        "~14d on $100k": top["est_14d"].map("{:+.1%}".format)})
    pd.set_option("display.width", 220)
    print("\nTop 20 of all indicator x holding-period combinations on the research period (sorted by t):")
    print(show.to_string(index=False))
    print("IC = rank correlation between the indicator and the next period's relative return (0.02-0.05 is "
          "typical of a real but weak signal). 'buy high' = long the coins with the highest values.")

    real = grid[(grid["ic_t"] >= 3) & ~grid["indicator"].str.startswith("COMBINED")].sort_values("ic_t", ascending=False)
    print("\nPatterns that are statistically real (rank IC t >= 3), whether or not they pay after costs:")
    if len(real):
        for _, r in real.iterrows():
            trade = "buy high" if r["direction"] > 0 else "buy low"
            verdict = "survives costs" if r["net_bps"] > 0 else "eaten by costs"
            print(f"  {r['indicator']:<26} {trade}, hold {hold_txt(int(r['hold'])):>4}: IC {r['ic']:+.3f} (t {r['ic_t']:.1f}), "
                  f"gross {r['gross_bps']:+.1f} bps vs cost {r['cost_bps']:.1f} bps -> {verdict}")
    else:
        print("  none")

    sample = c.index[(c.index >= research_start) & (c.index < holdout_start)][::12]
    corr = redundancy(sigs, sample)
    seen, families = set(), []
    for name in corr.index:
        if name in seen:
            continue
        fam = [n for n in corr.index if n not in seen and abs(corr.loc[name, n]) >= 0.7]
        seen.update(fam)
        if len(fam) > 1:
            families.append(fam)
    if families:
        print("\nIndicators that rank coins almost identically (|rank correlation| >= 0.7), i.e. the same signal:")
        for fam in families:
            print("  - " + ", ".join(fam))

    line = "=" * 100
    print(f"\n{line}\nTRADEABLE?  Tested {len(grid)} combinations. Each must pass all rules on the research period:")
    steps = [("pass_profit", f"net profit >= {rules.min_net_bps:g} bps per holding period after costs"),
             ("pass_confidence", f"...and statistically confident (t >= {rules.min_t:g})"),
             ("pass_sample", f"...and enough periods (>= {rules.min_periods}, >= {rules.min_periods_per_half} per half)"),
             ("pass_stable", "...and profitable in BOTH halves of the research period")]
    mask = pd.Series(True, index=grid.index)
    for col, text in steps:
        mask &= grid[col].astype(bool)
        print(f"  {text:<78} {int(mask.sum()):>5}")
    winners = grid[grid["tradeable"]]
    if len(winners):
        print(f"\n{len(winners)} rule(s) pass. Checking each ONCE on the untouched holdout:")
        for _, w in winners.iterrows():
            sig = sigs.get(w["indicator"])
            if sig is None:  # combined signal: rebuild from its members and their research-period directions
                members = w["members"].split(", ")
                dirs = {r["indicator"]: r["direction"] for r in rows if r["hold"] == w["hold"] and r["indicator"] in members}
                sig = sum(dirs[m] * (sigs[m].rank(axis=1, pct=True) - 0.5) for m in members) / len(members)
            h = test(w["indicator"], sig, int(w["hold"]), holdout_start, end + bar, direction=int(w["direction"]))
            ok = bool(h) and h["net_bps"] > 0 and h["t"] >= rules.holdout_min_t
            trade = "buy high" if w["direction"] > 0 else "buy low"
            print(f"  {w['indicator']} ({trade}, hold {hold_txt(int(w['hold']))}): research net {w['net_bps']:+.1f} bps "
                  f"(t {w['t']:.1f}, ~{w['est_14d']:+.1%}/14d) | holdout net "
                  f"{h.get('net_bps', np.nan):+.1f} bps (t {h.get('t', np.nan):.1f}, ~{h.get('est_14d', np.nan):+.1%}/14d)"
                  f"  ->  {'CONFIRMED' if ok else 'NOT CONFIRMED'}")
            if "members" in w and isinstance(w["members"], str):
                print(f"      combined from: {w['members']}")
        print("\nOnly CONFIRMED rules are candidates for the bot. Don't re-tune them on the holdout.")
    else:
        print("\nNOTHING QUALIFIES on the research period. The holdout stays untouched.")
        names_ = {"pass_profit": "profit", "pass_confidence": "confidence", "pass_sample": "periods",
                  "pass_stable": "both halves"}
        print("Closest candidates and the rules they fail:")
        for _, w in grid.sort_values("net_bps", ascending=False).head(6).iterrows():
            fails = ", ".join(n for k, n in names_.items() if not w[k])
            trade = "buy high" if w["direction"] > 0 else "buy low"
            print(f"  {w['indicator']} ({trade}, hold {hold_txt(int(w['hold']))}): net {w['net_bps']:+.1f} bps, "
                  f"t {w['t']:.1f}, IC {w['ic']:+.3f}  (fails: {fails})")
    print(line)
    OUT.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT / f"indicators_{args.bar}.csv", index=False)
    corr.round(2).to_csv(OUT / f"indicator_correlations_{args.bar}.csv")
    print(f"saved {OUT}/indicators_{args.bar}.csv and indicator_correlations_{args.bar}.csv")


if __name__ == "__main__":
    main()
