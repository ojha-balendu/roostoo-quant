"""Volatility, correlation and BTC-beta overview of the crypto universe.

Run from the repo root:   python research/01_correlations.py            (last 365 days)
                          python research/01_correlations.py --days 180
Tables go to reports/research/; heatmaps too if matplotlib is installed
(pip install -e ".[research]").
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.config import load_config
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe

BTC = "BTC/USD"
OUT = Path("reports/research")


def avg_offdiag(c: pd.DataFrame) -> float:
    a = c.to_numpy()
    return float(np.nanmean(a[~np.eye(len(a), dtype=bool)]))


def avg_corr_per_coin(c: pd.DataFrame) -> pd.Series:
    a = c.to_numpy(copy=True)
    np.fill_diagonal(a, np.nan)
    return pd.Series(np.nanmean(a, axis=0), index=c.columns)


def heatmap(c: pd.DataFrame, title: str, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(c.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    labels = [p.split("/")[0] for p in c.columns]
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--min-coverage", type=float, default=0.9, help="keep coins with >= this share of hours")
    args = ap.parse_args()

    cfg = load_config()
    closes = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", load_universe(cfg.data.universe_path))
    end = closes.index.max()
    closes = closes.loc[closes.index > end - pd.Timedelta(days=args.days)]
    coverage = closes.notna().mean()
    dropped = sorted(coverage[coverage < args.min_coverage].index)
    closes = closes.loc[:, coverage >= args.min_coverage]
    print(f"window {closes.index.min():%Y-%m-%d} -> {end:%Y-%m-%d}, {closes.shape[1]} coins")
    if dropped:
        print(f"dropped for short history in this window: {[p.split('/')[0] for p in dropped]}")

    r_h = np.log(closes).diff()                           # hourly log returns
    r_d = np.log(closes.resample("1D").last()).diff()     # daily log returns
    r_d = r_d.iloc[1:]
    btc = r_d[BTC]

    # Volatility and the fee hurdle -------------------------------------------
    ann_vol = r_h.std() * np.sqrt(24 * 365)
    med_abs_day = r_d.abs().median()

    # Correlations (ordered by correlation with BTC for readable heatmaps) ------
    pearson_d = r_d.corr()
    order = pearson_d[BTC].sort_values(ascending=False).index
    pearson_d = pearson_d.loc[order, order]
    spearman_d = r_d.corr(method="spearman").loc[order, order]
    pearson_h = r_h.corr().loc[order, order]
    crash_days = btc <= btc.quantile(0.2)
    pearson_crash = r_d[crash_days].corr().loc[order, order]

    # BTC beta and correlation after removing the BTC factor ---------------------
    beta = r_d.apply(lambda s: s.cov(btc) / btc.var())
    resid = r_d - np.outer(btc.to_numpy(), beta.to_numpy())
    resid = pd.DataFrame(resid, index=r_d.index, columns=r_d.columns).drop(columns=BTC)
    resid_corr = resid.corr()

    per_coin = pd.DataFrame({
        "ann_vol_%": ann_vol * 100,
        "median_daily_move_%": med_abs_day * 100,
        "beta_btc": beta,
        "corr_btc": pearson_d[BTC],
        "avg_corr_pearson": avg_corr_per_coin(pearson_d),
        "avg_corr_spearman": avg_corr_per_coin(spearman_d),
        "avg_corr_ex_btc": avg_corr_per_coin(resid_corr),
    }).round(3).sort_values("ann_vol_%")

    pd.set_option("display.width", 160)
    print("\nPer coin (sorted by volatility):")
    print(per_coin.to_string())

    print("\nAverage pairwise correlation across all coins:")
    print(f"  hourly returns, Pearson        {avg_offdiag(pearson_h):.2f}")
    print(f"  daily returns,  Pearson        {avg_offdiag(pearson_d):.2f}")
    print(f"  daily returns,  Spearman       {avg_offdiag(spearman_d):.2f}")
    print(f"  BTC's worst 20% of days        {avg_offdiag(pearson_crash):.2f}")
    print(f"  after removing BTC factor      {avg_offdiag(resid_corr):.2f}")

    pairs = pearson_d.where(np.triu(np.ones(pearson_d.shape, dtype=bool), 1)).stack().dropna().sort_values()
    fmt = lambda s: "\n".join(f"  {a.split('/')[0]:>10} ~ {b.split('/')[0]:<10} {v:+.2f}" for (a, b), v in s.items())  # noqa: E731
    print("\nMost correlated pairs (daily, Pearson):\n" + fmt(pairs.tail(10)[::-1]))
    print("\nLeast correlated pairs:\n" + fmt(pairs.head(10)))

    print(f"\nFee check: a round trip costs ~0.2%. Median daily move ranges "
          f"{per_coin['median_daily_move_%'].min():.2f}% to {per_coin['median_daily_move_%'].max():.2f}%.")

    OUT.mkdir(parents=True, exist_ok=True)
    per_coin.to_csv(OUT / "per_coin.csv")
    pearson_d.round(3).to_csv(OUT / "corr_pearson_daily.csv")
    spearman_d.round(3).to_csv(OUT / "corr_spearman_daily.csv")
    resid_corr.round(3).to_csv(OUT / "corr_ex_btc_daily.csv")
    heatmap(pearson_d, "Daily return correlation (Pearson)", OUT / "corr_pearson_daily.png")
    heatmap(resid_corr, "Correlation after removing BTC factor", OUT / "corr_ex_btc_daily.png")
    print(f"\nsaved tables{' and heatmaps' if (OUT / 'corr_pearson_daily.png').exists() else ''} to {OUT}/")


if __name__ == "__main__":
    main()
