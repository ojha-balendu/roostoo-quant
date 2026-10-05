"""Evaluate a trained agent on the held-out test period, against benchmarks.

    python research/11_rl_evaluate.py                         # latest run in models/rl/
    python research/11_rl_evaluate.py --run models/rl/quick --model final

Every strategy goes through the same accounting (fees, spread, gross cap, trade band):
  agent (deterministic), never trade, hold BTC, equal-weight 15 majors, and the competition
  trend portfolio (config/competition.yaml, rebalanced daily at 00:00 UTC).
Scores: every 14-day window starting every 2 days in the test period (overlapping), plus
back-to-back non-overlapping windows for the chart. Also shows the agent's in-sample score
on training windows, to expose overfitting.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from quantbot.config import load_config
from quantbot.rl.data import Split, window_starts
from quantbot.rl.env import RewardConfig
from quantbot.rl.features import LOOKBACKS
from quantbot.rl.pipeline import make_env, prepare_market, run_policy, summarize_runs
from quantbot.rl.sim import SimConfig, run_weights
from quantbot.strategy.registry import get_strategy

BTC = "BTC/USD"
COLORS = {"agent": "tab:blue", "never trade": "tab:gray", "hold BTC": "tab:orange", "equal-weight 15": "tab:red",
          "trend portfolio": "tab:green"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="models/rl/<name> (default: most recent)")
    ap.add_argument("--model", default="best", choices=["best", "final"])
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    runs_dir = Path("models/rl")
    run = Path(args.run) if args.run else max((p for p in runs_dir.glob("*") if (p / "meta.json").exists()),
                                              key=lambda p: p.stat().st_mtime, default=None)
    if run is None:
        raise SystemExit("no trained runs in models/rl/: run research/10_rl_train.py first")
    meta = json.loads((run / "meta.json").read_text())
    cfg = load_config(meta["config"])
    split = Split.from_dict(meta["split"])
    sim_cfg = SimConfig(**meta["sim"])
    rw = dict(meta["reward"])
    rw["score_weights"] = tuple(rw["score_weights"])
    reward_cfg = RewardConfig(**rw)
    model = PPO.load(run / f"{args.model}_model", device="cpu")
    m = prepare_market(cfg, meta["assets"], split, meta.get("feature_lookbacks", LOOKBACKS))
    ep = reward_cfg.episode_bars
    cols = list(m.closes.columns)
    print(f"run {run.name} ({args.model} model), test period {split.test_start:%Y-%m-%d} -> {split.end:%Y-%m-%d}")

    test_env = make_env(m, sim_cfg, reward_cfg, split.test_start, split.end + pd.Timedelta(minutes=15),
                        random_start=False)
    lo, end_idx = m.idx(split.test_start), test_env.start_hi + ep
    overlapping = window_starts(lo, end_idx, ep, step=96 * 2)
    back_to_back = window_starts(lo, end_idx, ep, step=ep)
    if not overlapping:
        raise SystemExit("test period shorter than one 14-day window")

    # benchmark weight schedules, aligned to the 15m bars
    n = len(m.prices)
    flat = np.zeros_like(m.prices)
    btc = np.zeros_like(m.prices)
    btc[:, cols.index(BTC)] = sim_cfg.max_gross
    avail = np.isfinite(m.prices).astype(float)
    equal = avail / np.maximum(avail.sum(axis=1, keepdims=True), 1) * sim_cfg.max_gross
    trend = get_strategy(cfg.strategy.name, cfg.strategy.params).generate_weights(m.closes)
    trend = trend.reindex(columns=cols).fillna(0.0).to_numpy()
    closes_at = m.closes.index + pd.Timedelta(minutes=15)
    daily = np.asarray((closes_at.hour == 0) & (closes_at.minute == 0))
    never = np.zeros(n, dtype=bool)
    schedules = {"never trade": (flat, never), "hold BTC": (btc, never), "equal-weight 15": (equal, never),
                 "trend portfolio": (trend, daily)}

    def run_all(starts):
        out = {"agent": [run_policy(model, test_env, s) for s in starts]}
        for name, (w, rb) in schedules.items():
            out[name] = [run_weights(m.prices, w, s, ep, sim_cfg, rb) for s in starts]
        return out

    results = run_all(overlapping)
    table = pd.DataFrame({k: summarize_runs(v, reward_cfg.score_weights) for k, v in results.items()}).T
    show = pd.DataFrame({
        "score": table["score"].map("{:+.2f}".format), "worst score": table["score_worst"].map("{:+.2f}".format),
        "return %": table["return_pct"].map("{:+.2f}".format), "Sortino": table["sortino"].map("{:+.2f}".format),
        "Sharpe": table["sharpe"].map("{:+.2f}".format), "Calmar": table["calmar"].map("{:+.2f}".format),
        "worst DD": table["max_dd"].map("{:.1%}".format), "% positive": table["pct_positive"].map("{:.0%}".format),
        "fees $": table["fees"].map("{:,.0f}".format), "avg exposure": table["exposure"].map("{:.2f}".format)})
    pd.set_option("display.width", 200)
    print(f"\nTEST: {len(overlapping)} overlapping 14-day windows (start every 2 days), ${sim_cfg.initial_capital:,.0f} "
          f"each, means across windows (score = 0.5*return% + 0.2*Sortino + 0.15*Sharpe + 0.15*Calmar):")
    print(show.to_string())

    train_env = make_env(m, sim_cfg, reward_cfg, split.train_start, split.val_start, random_start=False)
    tr_starts = window_starts(train_env.start_lo, train_env.start_hi + ep, ep, step=96 * 7)
    in_sample = summarize_runs([run_policy(model, train_env, s) for s in tr_starts], reward_cfg.score_weights)
    print(f"\nagent in-sample (training windows): score {in_sample['score']:+.2f}, return {in_sample['return_pct']:+.2f}%"
          f"  vs test: score {table.loc['agent', 'score']:+.2f}  -> "
          + ("large gap: the agent memorised its training period" if in_sample["score"] - table.loc["agent", "score"] > 3
             else "no large overfitting gap"))

    a, tr, b = table.loc["agent"], table.loc["trend portfolio"], table.loc["hold BTC"]
    beats = [k for k, r in (("trend portfolio", tr), ("hold BTC", b), ("equal-weight 15", table.loc["equal-weight 15"]))
             if a["score"] > r["score"]]
    if a["score"] <= 0.0:
        verdict = (f"the agent does NOT beat simply not trading (score {a['score']:+.2f} vs 0.00)"
                   + (f"; it scores above {', '.join(beats)} only because those lost money in this period"
                      if beats else ""))
    else:
        verdict = (f"the agent beats not trading (score {a['score']:+.2f})"
                   + (f" and {', '.join(beats)}" if beats else ", but none of the invested benchmarks")
                   + f" (trend {tr['score']:+.2f}, BTC {b['score']:+.2f})")
    print(f"\nVERDICT: {verdict}. Average exposure: agent {a['exposure']:.2f} vs trend {tr['exposure']:.2f}. "
          f"Only {len(back_to_back)} non-overlapping test windows exist, so treat gaps under ~2 points as noise.")

    # chart: back-to-back windows chained (each window restarts flat, equity carried over)
    chained = run_all(back_to_back)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    for name, lst in chained.items():
        eq, level = [], 1.0
        for r in lst:
            e = r["equity"] / r["equity"][0] * level
            eq.extend(e[:-1])
            level = e[-1]
        eq.append(level)
        x = m.closes.index[back_to_back[0]: back_to_back[0] + len(eq)]
        style = {"agent": dict(lw=2.4), "trend portfolio": dict(lw=1.8), "hold BTC": dict(lw=1.6)}.get(name, dict(lw=1, alpha=0.7))
        axes[0].plot(x, (np.array(eq) - 1) * 100, label=f"{name} ({(level - 1):+.1%})", color=COLORS[name], **style)
        if name in ("agent", "trend portfolio", "hold BTC"):
            arr = np.array(eq)
            axes[1].plot(x, (arr / np.maximum.accumulate(arr) - 1) * 100, label=name, lw=1.3, color=COLORS[name])
    g = np.concatenate([r["gross"] for r in chained["agent"]])
    nt = np.concatenate([r["net"] for r in chained["agent"]])
    xx = m.closes.index[back_to_back[0]: back_to_back[0] + len(g)]
    axes[2].plot(xx, g, label="agent gross exposure", color="tab:blue")
    axes[2].plot(xx, nt, label="agent net exposure (long - short)", color="tab:purple")
    for s in back_to_back:
        for ax in axes:
            ax.axvline(m.closes.index[s], color="grey", lw=0.6, ls=":")
    axes[0].set_title(f"RL agent vs benchmarks on the unseen test period ({len(back_to_back)} back-to-back 14-day "
                      f"competitions, ${sim_cfg.initial_capital:,.0f} each, fees included)")
    axes[0].set_ylabel("return %")
    axes[1].set_ylabel("drawdown %")
    axes[2].set_ylabel("exposure")
    axes[2].axhline(0, color="k", lw=0.5)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    png = run / f"test_{args.model}.png"
    fig.savefig(png, dpi=130)
    table.to_csv(run / f"test_{args.model}.csv")
    print(f"\nchart: {png}")
    if sys.platform == "darwin" and not args.no_open:
        subprocess.run(["open", str(png)], check=False)


if __name__ == "__main__":
    main()
