"""Train a PPO agent to trade the 15 majors on 15-minute bars (runs locally on CPU).

    pip install -e ".[rl]"                       # once: gymnasium, stable-baselines3, torch
    python research/10_rl_train.py               # 1M steps on all history before validation
    python research/10_rl_train.py --lookbacks 16,96,384 --net 64,64 --name slow_small

Data split, from the end of your 1m data backwards: test (60 days, never touched here),
validation (30 days, only to pick the best checkpoint), training (all earlier history by default).
The log shows training-episode returns next to validation, so overfitting is visible: training
improving while validation gets worse. Outputs: models/rl/<name>/. Then:
    python research/11_rl_evaluate.py --run models/rl/<name>
To compare several settings fairly, use research/12_rl_sweep.py.
"""

from __future__ import annotations

import argparse

from quantbot.rl.train import TrainParams, train_agent


def ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    d = TrainParams()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=d.config)
    ap.add_argument("--name", default=None)
    ap.add_argument("--timesteps", type=int, default=d.timesteps)
    ap.add_argument("--train-days", type=float, default=None, help="default: all history before validation")
    ap.add_argument("--val-days", type=float, default=d.val_days)
    ap.add_argument("--test-days", type=float, default=d.test_days)
    ap.add_argument("--capital", type=float, default=d.capital)
    ap.add_argument("--n-envs", type=int, default=d.n_envs)
    ap.add_argument("--eval-every", type=int, default=d.eval_every)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--gamma", type=float, default=d.gamma)
    ap.add_argument("--n-epochs", type=int, default=d.n_epochs)
    ap.add_argument("--net", type=ints, default=d.net, help="hidden layer sizes, e.g. 64,64")
    ap.add_argument("--ent-coef", type=float, default=d.ent_coef)
    ap.add_argument("--log-std-init", type=float, default=d.log_std_init,
                    help="initial exploration noise (log std); large noise = random trades whose fees drown learning")
    ap.add_argument("--reward", default=d.reward, choices=["semivariance", "loss_aversion"],
                    help="per-step reward; semivariance penalises squared losses (see rl/env.py)")
    ap.add_argument("--risk-aversion", type=float, default=d.risk_aversion,
                    help="semivariance penalty; best exposure ~ edge / (risk_aversion * variance)")
    ap.add_argument("--loss-aversion", type=float, default=d.loss_aversion, help="only for --reward loss_aversion")
    ap.add_argument("--terminal-weight", type=float, default=d.terminal_weight,
                    help="weight of the episode composite score added at the end (default 0: it is mostly noise)")
    ap.add_argument("--trade-band", type=float, default=d.trade_band)
    ap.add_argument("--lookbacks", type=ints, default=d.lookbacks,
                    help="trend lookbacks in 15m bars (1=15m, 4=1h, 16=4h, 96=1d, 384=4d)")
    a = ap.parse_args()
    p = TrainParams(name=a.name, config=a.config, timesteps=a.timesteps, train_days=a.train_days,
                    val_days=a.val_days, test_days=a.test_days, capital=a.capital, n_envs=a.n_envs,
                    eval_every=a.eval_every, seed=a.seed, lr=a.lr, gamma=a.gamma, n_epochs=a.n_epochs, net=a.net,
                    ent_coef=a.ent_coef, log_std_init=a.log_std_init, reward=a.reward,
                    risk_aversion=a.risk_aversion, loss_aversion=a.loss_aversion,
                    terminal_weight=a.terminal_weight, trade_band=a.trade_band, lookbacks=a.lookbacks)
    s = train_agent(p)
    print(f"\nbest validation score {s['best_score']:+.2f} at step {s['best_step']:,} "
          f"(hold BTC {s['hold_btc_score']:+.2f}). Saved to {s['dir']}/")
    if s["best_step"] == 0:
        print("NOTE: no trained checkpoint beat the untrained policy (which barely trades): nothing learned "
              "survives fees on the validation period.")
    print(f"next:  python research/11_rl_evaluate.py --run {s['dir']}")


if __name__ == "__main__":
    main()
