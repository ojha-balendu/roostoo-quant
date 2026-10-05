"""Glue shared by the training and evaluation scripts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantbot.config import Config
from quantbot.rl.data import Split, check_coverage, data_start_for, index_of, load_15m
from quantbot.rl.env import RewardConfig, TradingEnv
from quantbot.rl.features import LOOKBACKS, build_features
from quantbot.rl.scoring import composite_score, episode_metrics
from quantbot.rl.sim import SimConfig


@dataclass
class Market:
    closes: pd.DataFrame
    prices: np.ndarray
    features: np.ndarray
    valid: np.ndarray

    def idx(self, ts: pd.Timestamp) -> int:
        return index_of(self.closes.index, ts)


def prepare_market(cfg: Config, assets: list[str], split: Split, lookbacks=LOOKBACKS) -> Market:
    closes = load_15m(cfg, assets, data_start_for(split))
    closes = closes.loc[closes.index < split.end + pd.Timedelta(minutes=15)]
    check_coverage(closes, split)
    return with_features(Market(closes, closes.to_numpy(dtype=float), None, None), lookbacks)


def with_features(m: Market, lookbacks) -> Market:
    """Same prices, features built with the given trend lookbacks (in 15-minute bars)."""
    features, valid = build_features(m.closes, tuple(int(x) for x in lookbacks))
    return Market(m.closes, m.prices, features, valid)


def make_env(m: Market, sim_cfg: SimConfig, reward_cfg: RewardConfig, range_start: pd.Timestamp,
             range_end: pd.Timestamp, random_start: bool = True) -> TradingEnv:
    """Environment whose episodes start and finish inside [range_start, range_end)."""
    lo = m.idx(range_start)
    hi = m.idx(range_end) - reward_cfg.episode_bars - 1
    return TradingEnv(m.prices, m.features, m.valid, sim_cfg, reward_cfg, lo, hi, random_start)


def run_policy(model, env: TradingEnv, start: int, deterministic: bool = True) -> dict:
    obs, _ = env.reset(options={"start": start})
    gross, net, done, info = [], [], False, {}
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        gross.append(float(np.abs(env.sim.weights).sum()))
        net.append(float(env.sim.weights.sum()))
        done = terminated or truncated
    equity = np.asarray(env.curve)
    return {"equity": equity, "fees": env.sim.fees, "gross": np.array(gross), "net": np.array(net),
            "metrics": info.get("episode_metrics", episode_metrics(equity)), "score": info.get("score")}


def summarize_runs(runs: list[dict], score_weights) -> dict[str, float]:
    ms = [r.get("metrics") or episode_metrics(r["equity"]) for r in runs]
    scores = [composite_score(m, score_weights) for m in ms]
    return {
        "score": float(np.mean(scores)),
        "score_worst": float(np.min(scores)),
        "return_pct": float(np.mean([m["return_pct"] for m in ms])),
        "sortino": float(np.mean([m["sortino"] for m in ms])),
        "sharpe": float(np.mean([m["sharpe"] for m in ms])),
        "calmar": float(np.mean([m["calmar"] for m in ms])),
        "max_dd": float(np.max([m["max_dd"] for m in ms])),
        "pct_positive": float(np.mean([m["return_pct"] > 0 for m in ms])),
        "fees": float(np.mean([r["fees"] for r in runs])),
        "exposure": float(np.mean([np.mean(r["gross"]) if len(r.get("gross", [])) else 0.0 for r in runs])),
    }
