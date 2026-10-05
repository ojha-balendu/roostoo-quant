"""Gymnasium trading environment: one episode = one 14-day competition starting flat.

Every 15 minutes the agent sees each coin's features, its current weights, how far through
the episode it is and its current drawdown, and outputs a target weight per coin in [-1, 1]
(scaled by max_weight, gross-capped). Reward per step, with r = the step's portfolio return:

  "semivariance" (default):  100 * (r - risk_aversion * min(r, 0)^2)
      Mean-semivariance utility. The penalty is on SQUARED losses, so it is small for small
      positions and grows fast for large ones; the best exposure is interior (about
      edge / (risk_aversion * variance)) and the same at any time scale. Downside-only, like Sortino.
  "loss_aversion" (old):     100 * (log r - loss_aversion * max(0, -log r))
      A LINEAR loss penalty grows exactly like expected profit, so holding anything only pays
      with an implausibly large edge: it pushes the agent to hold nothing.

`terminal_weight` (default 0) adds the episode's composite score at the last step. It is off by
default because Sharpe/Sortino/Calmar ignore position size: a 1% position swings the score as much
as a 95% one, so as a training reward it is mostly noise. It is still used to pick checkpoints
and to judge results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from quantbot.rl.scoring import DEFAULT_WEIGHTS, composite_score, episode_metrics
from quantbot.rl.sim import PortfolioSim, SimConfig, simple_returns, weights_from_action


@dataclass
class RewardConfig:
    episode_days: float = 14.0
    bars_per_day: int = 96
    reward: str = "semivariance"   # "semivariance" or "loss_aversion"
    risk_aversion: float = 5.0     # semivariance penalty
    loss_aversion: float = 0.5     # only for reward="loss_aversion"
    terminal_weight: float = 0.0
    score_weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS
    bust_fraction: float = 0.5     # episode ends early if equity falls below this share of the start

    def __post_init__(self) -> None:
        if self.reward not in ("semivariance", "loss_aversion"):
            raise ValueError("reward must be 'semivariance' or 'loss_aversion'")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score_weights"] = list(self.score_weights)
        return d

    @property
    def episode_bars(self) -> int:
        return int(round(self.episode_days * self.bars_per_day))


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, prices: np.ndarray, features: np.ndarray, valid: np.ndarray, sim_cfg: SimConfig,
                 reward_cfg: RewardConfig, start_lo: int, start_hi: int, random_start: bool = True):
        """Episodes start at an index in [start_lo, start_hi] and run episode_bars steps."""
        super().__init__()
        self.prices = np.asarray(prices, dtype=float)
        self.features = features
        self.valid = valid & np.isfinite(self.prices)
        self.T, self.n = self.prices.shape
        self.sim_cfg, self.cfg = sim_cfg, reward_cfg
        self.ep_len = reward_cfg.episode_bars
        if start_hi + self.ep_len > self.T - 1:
            raise ValueError("episode range runs past the end of the data")
        if start_hi < start_lo:
            raise ValueError("empty range of episode starts")
        self.start_lo, self.start_hi, self.random_start = start_lo, start_hi, random_start
        self.rets = simple_returns(self.prices)
        self.sim = PortfolioSim(self.n, sim_cfg)
        dim = self.n * features.shape[-1] + self.n + 2
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n,), dtype=np.float32)
        self.t = self.t0 = start_lo
        self.curve: list[float] = []

    def _obs(self) -> np.ndarray:
        eq = np.asarray(self.curve)
        drawdown = 1.0 - eq[-1] / eq.max() if len(eq) else 0.0
        progress = (self.t - self.t0) / self.ep_len
        obs = np.concatenate([self.features[self.t].ravel(), self.sim.weights, [progress, drawdown]])
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options and "start" in options:
            t0 = int(options["start"])
            if not self.start_lo <= t0 <= self.start_hi:
                raise ValueError(f"start {t0} outside [{self.start_lo}, {self.start_hi}]")
        elif self.random_start:
            t0 = int(self.np_random.integers(self.start_lo, self.start_hi + 1))
        else:
            t0 = self.start_lo
        self.t = self.t0 = t0
        self.sim.reset()
        self.curve = [self.sim.equity]
        return self._obs(), {}

    def step(self, action):
        t = self.t
        tradable = self.valid[t]
        before = self.sim.equity
        cost = self.sim.rebalance(weights_from_action(action, tradable, self.sim_cfg), tradable)
        self.sim.advance(self.rets[t])
        after = self.sim.equity
        self.t += 1
        self.curve.append(after)

        if self.cfg.reward == "semivariance":
            r = after / before - 1.0 if before > 0 else -1.0
            reward = 100.0 * (r - self.cfg.risk_aversion * min(r, 0.0) ** 2)
        else:
            lr = float(np.log(after / before)) if after > 0 and before > 0 else -10.0
            reward = 100.0 * (lr - self.cfg.loss_aversion * max(0.0, -lr))
        terminated = after <= self.cfg.bust_fraction * self.sim_cfg.initial_capital
        truncated = (self.t - self.t0) >= self.ep_len
        info = {"equity": after, "cost": cost, "gross": float(np.abs(self.sim.weights).sum())}
        if terminated or truncated:
            m = episode_metrics(np.asarray(self.curve))
            score = composite_score(m, self.cfg.score_weights)
            reward += self.cfg.terminal_weight * score
            info.update(episode_metrics=m, score=score, fees=self.sim.fees)
        return self._obs(), float(reward), bool(terminated), bool(truncated and not terminated), info
