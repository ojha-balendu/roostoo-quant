"""PPO training with checkpoint selection on a validation period. Used by research/10 and 12."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.config import load_config
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe
from quantbot.rl.data import BAR, MAJORS, Split, make_split, window_starts
from quantbot.rl.env import RewardConfig
from quantbot.rl.features import LOOKBACKS, WARMUP_BARS
from quantbot.rl.pipeline import Market, make_env, prepare_market, run_policy, summarize_runs, with_features
from quantbot.rl.sim import SimConfig, run_weights


@dataclass
class TrainParams:
    name: str | None = None
    config: str = "config/competition.yaml"
    timesteps: int = 1_000_000
    train_days: float | None = None        # None = all history available before validation
    val_days: float = 30
    test_days: float = 60
    capital: float = 100_000
    n_envs: int = 4
    eval_every: int = 50_000
    seed: int = 0
    lr: float = 3e-4
    gamma: float = 0.999
    n_epochs: int = 10
    net: list[int] = field(default_factory=lambda: [128, 128])
    ent_coef: float = 0.0
    log_std_init: float = -3.0
    reward: str = "semivariance"
    risk_aversion: float = 5.0
    loss_aversion: float = 0.5
    terminal_weight: float = 0.0
    trade_band: float = 0.05
    lookbacks: list[int] = field(default_factory=lambda: list(LOOKBACKS))
    out_root: str = "models/rl"


def resolve_split(p: TrainParams) -> Split:
    cfg = load_config(p.config)
    universe = load_universe(cfg.data.universe_path)
    probe = load_panel(cfg.data.processed_dir, cfg.data.market, "1m", {"BTC/USD": universe["BTC/USD"]})
    if probe.empty:
        raise SystemExit("no 1m data: run  quantbot download --interval 1m --start <date>")
    end = probe.index.max().floor("15min")
    train_days = p.train_days
    if train_days is None:  # everything between the warm-up and the validation period
        val_start = end - pd.Timedelta(days=p.test_days + p.val_days)
        earliest = probe.index.min() + BAR * (WARMUP_BARS + 2 * 96)
        train_days = max(0.0, (val_start - earliest).total_seconds() / 86400 - 1)
        if train_days < 30:
            raise SystemExit(f"only {train_days:.0f} days of 1m history before validation: download more")
    return make_split(end, train_days, p.val_days, p.test_days)


def _make_callback(env, starts, every, out_dir: Path, score_weights, total, verbose: bool):
    # imported here so this module loads without stable-baselines3 installed
    from stable_baselines3.common.callbacks import BaseCallback

    class ValidationCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval, self.best, self.best_step, self.rows, self.t0 = 0, -np.inf, 0, [], time.time()
            self.train_returns, self.train_fees = [], []

        def evaluate(self) -> None:
            runs = [run_policy(self.model, env, s) for s in starts]
            s = summarize_runs(runs, score_weights)
            tr = float(np.mean(self.train_returns[-16:])) if self.train_returns else float("nan")
            tf = float(np.mean(self.train_fees[-16:])) if self.train_fees else float("nan")
            row = {"timesteps": self.num_timesteps, **s, "train_return_pct": tr, "train_fees": tf}
            self.rows.append(row)
            improved = s["score"] > self.best
            if improved:
                self.best, self.best_step = s["score"], self.num_timesteps
                self.model.save(out_dir / "best_model")
            pd.DataFrame(self.rows).to_csv(out_dir / "validation.csv", index=False)
            if verbose:
                elapsed = time.time() - self.t0
                speed = self.num_timesteps / elapsed if elapsed > 1 and self.num_timesteps else 0
                eta = max(0.0, (total - self.num_timesteps) / speed / 60) if speed else float("nan")
                train_txt = f"train ret {tr:+6.2f}% fees ${tf:>6,.0f} | " if np.isfinite(tr) else " " * 33 + "| "
                print(f"  step {self.num_timesteps:>9,} | {train_txt}val score {s['score']:+6.2f} ret "
                      f"{s['return_pct']:+6.2f}% Sortino {s['sortino']:+6.2f} maxDD {s['max_dd']:6.2%} fees "
                      f"${s['fees']:>6,.0f} exp {s['exposure']:.2f}{' <- best' if improved else ''}"
                      + (f"  [{speed:,.0f}/s, ~{eta:.0f} min]" if speed else ""), flush=True)

        def _on_training_start(self) -> None:
            self.evaluate()
            self.next_eval = every

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if "episode_metrics" in info:
                    self.train_returns.append(info["episode_metrics"]["return_pct"])
                    self.train_fees.append(info["fees"])
            if self.num_timesteps >= self.next_eval:
                self.evaluate()
                self.next_eval += every
            return True

    return ValidationCallback()


def hold_btc_score(m: Market, starts, sim_cfg: SimConfig, reward_cfg: RewardConfig) -> dict[str, float]:
    w = np.zeros_like(m.prices)
    w[:, list(m.closes.columns).index("BTC/USD")] = sim_cfg.max_gross
    never = np.zeros(len(m.prices), dtype=bool)
    return summarize_runs([run_weights(m.prices, w, s, reward_cfg.episode_bars, sim_cfg, never) for s in starts],
                          reward_cfg.score_weights)


def train_agent(p: TrainParams, split: Split | None = None, market: Market | None = None,
                verbose: bool = True) -> dict:
    """Train one agent; returns a summary. Pass `split`/`market` to reuse loaded data (sweeps)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    cfg = load_config(p.config)
    split = split or resolve_split(p)
    market = with_features(market, p.lookbacks) if market is not None else prepare_market(
        cfg, MAJORS, split, p.lookbacks)
    sim_cfg = SimConfig(initial_capital=p.capital, fee=cfg.fees.taker, spread_bps=cfg.backtest.spread_bps,
                        trade_band=p.trade_band)
    reward_cfg = RewardConfig(reward=p.reward, risk_aversion=p.risk_aversion, loss_aversion=p.loss_aversion,
                              terminal_weight=p.terminal_weight)
    ep = reward_cfg.episode_bars
    make_env(market, sim_cfg, reward_cfg, split.train_start, split.val_start)  # fails early if range too short
    val_env = make_env(market, sim_cfg, reward_cfg, split.val_start, split.test_start, random_start=False)
    val_starts = window_starts(market.idx(split.val_start), market.idx(split.test_start) - 1, ep, step=96 * 2)
    if not val_starts:
        raise SystemExit("validation period shorter than one 14-day episode: increase val_days")
    bench = hold_btc_score(market, val_starts, sim_cfg, reward_cfg)

    name = p.name or pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = Path(p.out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    meta = {**asdict(p), "name": name, "assets": MAJORS, "split": split.to_dict(), "sim": sim_cfg.to_dict(),
            "reward": reward_cfg.to_dict(), "feature_lookbacks": list(p.lookbacks)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        days = (split.val_start - split.train_start).days
        print(f"train {split.train_start:%Y-%m-%d} -> {split.val_start:%Y-%m-%d} ({days} days) | validation -> "
              f"{split.test_start:%Y-%m-%d} | test -> {split.end:%Y-%m-%d} (not used)")
        print(f"{len(val_starts)} validation windows; hold BTC scores {bench['score']:+.2f} there, never trading 0.00")
        print(f"PPO: {p.timesteps:,} steps, net {p.net}, lr {p.lr}, epochs {p.n_epochs}, lookbacks {p.lookbacks} "
              f"bars, seed {p.seed}; reward {p.reward}"
              + (f" (risk aversion {p.risk_aversion:g})" if p.reward == "semivariance" else "")
              + (f", terminal weight {p.terminal_weight:g}" if p.terminal_weight else "") + "\n")

    def factory(rank: int):
        def _make():
            env = make_env(market, sim_cfg, reward_cfg, split.train_start, split.val_start)
            env.reset(seed=p.seed * 100 + rank)
            return env
        return _make

    vec = DummyVecEnv([factory(i) for i in range(p.n_envs)])
    model = PPO("MlpPolicy", vec, learning_rate=p.lr, n_steps=2048, batch_size=512, n_epochs=p.n_epochs,
                gamma=p.gamma, gae_lambda=0.95, clip_range=0.2, ent_coef=p.ent_coef, seed=p.seed, device="cpu",
                policy_kwargs={"net_arch": list(p.net), "log_std_init": p.log_std_init}, verbose=0)
    cb = _make_callback(val_env, val_starts, p.eval_every, out, reward_cfg.score_weights, p.timesteps, verbose)
    model.learn(total_timesteps=p.timesteps, callback=cb)
    cb.evaluate()
    model.save(out / "final_model")
    _plot(out, bench)
    rows = pd.DataFrame(cb.rows)
    late = rows[rows["timesteps"] > 0]
    return {"name": name, "dir": str(out), "best_score": cb.best, "best_step": cb.best_step,
            "final_score": float(rows["score"].iloc[-1]),
            "best_trained_score": float(late["score"].max()) if len(late) else float("nan"),
            "train_return_pct": float(rows["train_return_pct"].iloc[-1]),
            "val_return_pct": float(rows["return_pct"].iloc[-1]),
            "final_exposure": float(rows["exposure"].iloc[-1]), "final_fees": float(rows["fees"].iloc[-1]),
            "hold_btc_score": bench["score"]}


def _plot(out: Path, bench: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    v = pd.read_csv(out / "validation.csv")
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(v["timesteps"], v["score"], marker="o", label="agent (validation)")
    ax[0].axhline(bench["score"], color="tab:orange", ls="--", label="hold BTC")
    ax[0].axhline(0, color="k", lw=0.6, label="never trade")
    ax[0].set_ylabel("composite score")
    ax[1].plot(v["timesteps"], v["train_return_pct"], marker="o", label="training episodes (exploring)")
    ax[1].plot(v["timesteps"], v["return_pct"], marker="o", label="validation")
    ax[1].set_ylabel("return per 14 days %")
    ax[2].plot(v["timesteps"], v["fees"], marker="o", color="tab:red", label="validation fees")
    ax[2].set_ylabel("fees per 14 days ($)")
    ax[2].set_xlabel("training steps")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "training.png", dpi=120)
    plt.close(fig)
