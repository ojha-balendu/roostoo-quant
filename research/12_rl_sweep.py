"""Compare a few RL configurations fairly: several seeds each, ranked on validation only.

    python research/12_rl_sweep.py                       # 7 configs x 3 seeds x 500k steps (~20 min)
    python research/12_rl_sweep.py --timesteps 300000 --seeds 2
    python research/12_rl_sweep.py --configs baseline,slow_small

Why seeds: RL results vary a lot between random seeds, so one lucky run proves nothing.
Ranking uses the mean (across seeds) of the best validation score among TRAINED checkpoints.
The test period is never touched here; evaluate only the winner, once, with research/11.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from quantbot.config import load_config
from quantbot.rl.data import MAJORS
from quantbot.rl.pipeline import prepare_market
from quantbot.rl.train import TrainParams, resolve_split, train_agent

SLOW = [16, 96, 384]  # 4h, 1d, 4d trends only
CONFIGS = {
    "baseline": {},
    "slow": {"lookbacks": SLOW},
    "slow_small": {"lookbacks": SLOW, "net": [64, 64], "n_epochs": 5},
    "slow_small_cautious": {"lookbacks": SLOW, "net": [64, 64], "n_epochs": 5, "risk_aversion": 10.0},
    "slow_small_bold": {"lookbacks": SLOW, "net": [64, 64], "n_epochs": 5, "risk_aversion": 2.0},
    "slow_small_wideband": {"lookbacks": SLOW, "net": [64, 64], "n_epochs": 5, "trade_band": 0.10},
    "slow_small_lowlr": {"lookbacks": SLOW, "net": [64, 64], "n_epochs": 5, "lr": 1e-4},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--configs", default=",".join(CONFIGS), help=f"comma-separated subset of {list(CONFIGS)}")
    ap.add_argument("--config", default="config/competition.yaml")
    args = ap.parse_args()
    chosen = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = set(chosen) - set(CONFIGS)
    if unknown:
        raise SystemExit(f"unknown configs {sorted(unknown)}; choose from {list(CONFIGS)}")

    tag = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    base = TrainParams(config=args.config, timesteps=args.timesteps, eval_every=max(args.timesteps // 10, 10_000))
    split = resolve_split(base)
    market = prepare_market(load_config(args.config), MAJORS, split)
    n_runs = len(chosen) * args.seeds
    print(f"sweep {tag}: {len(chosen)} configs x {args.seeds} seeds x {args.timesteps:,} steps = {n_runs} runs "
          f"(~{n_runs * args.timesteps / 9000 / 60:.0f} min at 9k steps/s)")
    print(f"train {split.train_start:%Y-%m-%d} -> {split.val_start:%Y-%m-%d} | validation -> "
          f"{split.test_start:%Y-%m-%d} | test (untouched) -> {split.end:%Y-%m-%d}\n")

    rows, t0 = [], time.time()
    for cname in chosen:
        for seed in range(args.seeds):
            p = replace(base, name=f"sweep_{tag}/{cname}_s{seed}", seed=seed, **CONFIGS[cname])
            s = train_agent(p, split=split, market=market, verbose=False)
            rows.append({"config": cname, "seed": seed, **s})
            print(f"  {cname:<22} seed {seed}: best trained val score {s['best_trained_score']:+6.2f} | final: "
                  f"val {s['final_score']:+6.2f}, val return {s['val_return_pct']:+6.2f}%, train return "
                  f"{s['train_return_pct']:+6.2f}%, exposure {s['final_exposure']:.2f}, fees ${s['final_fees']:,.0f}"
                  f"   [{(time.time() - t0) / 60:.0f} min]", flush=True)

    df = pd.DataFrame(rows)
    out = Path(base.out_root) / f"sweep_{tag}"
    df.to_csv(out / "sweep.csv", index=False)
    g = df.groupby("config").agg(
        score=("best_trained_score", "mean"), score_sd=("best_trained_score", "std"),
        seeds_above_zero=("best_trained_score", lambda x: int((x > 0).sum())),
        final_val=("final_score", "mean"), val_return=("val_return_pct", "mean"),
        train_return=("train_return_pct", "mean"), exposure=("final_exposure", "mean"),
        fees=("final_fees", "mean")).sort_values("score", ascending=False)
    g["gap"] = g["train_return"] - g["val_return"]
    show = pd.DataFrame({
        "mean best val score": g["score"].map("{:+.2f}".format), "+/- sd": g["score_sd"].map("{:.2f}".format),
        f"seeds > 0 (of {args.seeds})": g["seeds_above_zero"],
        "final val score": g["final_val"].map("{:+.2f}".format),
        "final val ret %": g["val_return"].map("{:+.2f}".format),
        "final train ret %": g["train_return"].map("{:+.2f}".format),
        "train-val gap": g["gap"].map("{:+.2f}".format),
        "exposure": g["exposure"].map("{:.2f}".format), "fees $": g["fees"].map("{:,.0f}".format)})
    pd.set_option("display.width", 220)
    print(f"\nRanking (validation only; never trading scores 0.00, hold BTC {df['hold_btc_score'].iloc[0]:+.2f}):")
    print(show.to_string())
    print("train-val gap = training-episode return minus validation return; large and positive = overfitting.")

    best = g.index[0]
    if g.loc[best, "score"] <= 0 or g.loc[best, "seeds_above_zero"] < max(2, args.seeds - 1):
        print(f"\nVERDICT: no configuration reliably beats not trading on validation (best: {best}, mean "
              f"{g.loc[best, 'score']:+.2f}, {g.loc[best, 'seeds_above_zero']}/{args.seeds} seeds above zero). "
              "The agent is not finding an edge that survives fees. The test period stays untouched.")
    else:
        run = df[df["config"] == best].sort_values("best_trained_score", ascending=False).iloc[0]
        print(f"\nVERDICT: '{best}' beats not trading on validation in {g.loc[best, 'seeds_above_zero']}/{args.seeds} "
              f"seeds (mean {g.loc[best, 'score']:+.2f}). Check its best seed ONCE on the untouched test period:")
        print(f"  python research/11_rl_evaluate.py --run {run['dir']}")
    print(f"\nsaved {out}/sweep.csv")


if __name__ == "__main__":
    main()
