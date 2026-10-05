import numpy as np
import pandas as pd
import pytest

from quantbot.rl.features import build_features
from quantbot.rl.scoring import composite_score, episode_metrics
from quantbot.rl.sim import SimConfig, run_weights, weights_from_action

NO_COST = SimConfig(fee=0.0, spread_bps=0.0, trade_band=0.0, max_gross=1.0, max_weight=1.0)


def prices_15m(days=40, n=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=days * 96, freq="15min", tz="UTC")
    r = rng.normal(0, 0.002, (len(idx), n))
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx,
                        columns=["BTC/USD", "ETH/USD", "SOL/USD"][:n])


gym = pytest.importorskip("gymnasium")
sb3 = pytest.importorskip("stable_baselines3")

from quantbot.rl.env import RewardConfig, TradingEnv  # noqa: E402


def make_test_env(p, cfg=None, rcfg=None, random_start=True):
    cfg = cfg or SimConfig()
    f, v = build_features(p)
    rcfg = rcfg or RewardConfig(episode_days=2)
    lo = 96 * 8
    return TradingEnv(p.to_numpy(), f, v, cfg, rcfg, lo, len(p) - rcfg.episode_bars - 2, random_start)


def test_env_passes_official_checkers():
    from gymnasium.utils.env_checker import check_env as gym_check
    from stable_baselines3.common.env_checker import check_env as sb3_check

    gym_check(make_test_env(prices_15m()), skip_render_check=True)
    sb3_check(make_test_env(prices_15m()), warn=True)


def test_episode_length_rewards_and_terminal_score():
    p = prices_15m()
    rcfg = RewardConfig(episode_days=2, reward="loss_aversion", loss_aversion=0.0, terminal_weight=0.0)
    env = make_test_env(p, NO_COST, rcfg)
    obs, _ = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape and np.isfinite(obs).all()
    total, steps, done = 0.0, 0, False
    action = np.array([1.0, -0.5, 0.0], dtype=np.float32)
    while not done:
        obs, r, term, trunc, info = env.step(action)
        total += r
        steps += 1
        done = term or trunc
    assert steps == rcfg.episode_bars and trunc and not term
    assert abs(total - 100 * np.log(env.curve[-1] / env.curve[0])) < 1e-6   # dense reward = log growth in %
    assert abs(info["score"] - composite_score(episode_metrics(np.asarray(env.curve)))) < 1e-12


def test_never_trading_scores_zero():
    env = make_test_env(prices_15m())
    env.reset(seed=0)
    done = False
    while not done:
        _, _, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        done = term or trunc
    assert info["score"] == 0.0 and env.curve[-1] == env.curve[0]


def test_agent_and_benchmarks_share_identical_accounting():
    p = prices_15m()
    env = make_test_env(p, random_start=False)
    env.reset(options={"start": env.start_lo})
    action = np.array([1.0, -0.6, 0.2], dtype=np.float32)                 # constant action
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(action)
        done = term or trunc
    w = np.tile(weights_from_action(action, np.ones(3, bool), env.sim_cfg), (len(p), 1))
    bench = run_weights(p.to_numpy(), w, env.start_lo, env.ep_len, env.sim_cfg)
    assert np.allclose(np.asarray(env.curve), bench["equity"], rtol=0, atol=1e-6)


def test_ppo_trains_saves_loads_and_runs(tmp_path):
    from stable_baselines3 import PPO

    from quantbot.rl.pipeline import run_policy, summarize_runs

    env = make_test_env(prices_15m())
    model = PPO("MlpPolicy", env, n_steps=128, batch_size=64, n_epochs=2, seed=0, device="cpu", verbose=0,
                policy_kwargs={"net_arch": [32, 32], "log_std_init": -1.0})
    model.learn(total_timesteps=512)
    model.save(tmp_path / "m")
    loaded = PPO.load(tmp_path / "m", device="cpu")
    obs, _ = env.reset(seed=3)
    assert np.allclose(model.predict(obs, deterministic=True)[0], loaded.predict(obs, deterministic=True)[0])
    eval_env = make_test_env(prices_15m(), random_start=False)
    run = run_policy(loaded, eval_env, eval_env.start_lo)
    s = summarize_runs([run], (0.5, 0.2, 0.15, 0.15))
    assert len(run["equity"]) == eval_env.ep_len + 1 and np.isfinite(list(s.values())).all()


def test_observation_size_follows_feature_count():
    p = prices_15m()
    f, v = build_features(p, (16, 96))
    env = TradingEnv(p.to_numpy(), f, v, SimConfig(), RewardConfig(episode_days=2), 96 * 8, len(p) - 300)
    assert env.observation_space.shape == (3 * 3 + 3 + 2,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape


def test_train_agent_end_to_end_on_disk_data(tmp_path, monkeypatch):
    """The exact path used by research/10 and research/12, on a small synthetic 1m dataset."""
    from quantbot.rl.data import MAJORS
    from quantbot.rl.train import TrainParams, resolve_split, train_agent

    rng = np.random.default_rng(1)
    idx = pd.date_range("2026-01-01", periods=1440 * 80, freq="1min", tz="UTC", name="open_time")
    f = rng.normal(0, 4e-4, len(idx))
    data = tmp_path / "data/processed/spot/1m"
    data.mkdir(parents=True)
    lines = ["pairs:"]
    for i, pair in enumerate(MAJORS):
        sym = pair.split("/")[0] + "USDT"
        r = (1 + 0.05 * i) * f + rng.normal(0, 3e-4, len(idx))
        pd.DataFrame({"close": 100 * np.exp(np.cumsum(r))}, index=idx).to_parquet(data / f"{sym}.parquet")
        lines.append(f'  "{pair}": {sym}')
    (tmp_path / "config").mkdir()
    (tmp_path / "config/universe.yaml").write_text("\n".join(lines) + "\n")
    (tmp_path / "config/test.yaml").write_text("data:\n  universe_path: config/universe.yaml\n")
    monkeypatch.chdir(tmp_path)

    p = TrainParams(config="config/test.yaml", timesteps=4096, n_envs=2, eval_every=2048, val_days=15,
                    test_days=15, net=[16, 16], lookbacks=[16, 96], name="t")
    split = resolve_split(p)                       # train_days=None -> all history before validation
    assert split.val_start - split.train_start > pd.Timedelta(days=30)
    s = train_agent(p, verbose=False)
    out = tmp_path / "models/rl/t"
    assert {"best_model.zip", "final_model.zip", "meta.json", "validation.csv"} <= {x.name for x in out.iterdir()}
    v = pd.read_csv(out / "validation.csv")
    assert v["timesteps"].iloc[0] == 0 and len(v) >= 3 and np.isfinite(s["final_score"])
    assert s["best_score"] >= v["score"].max() - 1e-9


def run_episode(env, action):
    env.reset(options={"start": env.start_lo})
    rewards, done, info = [], False, {}
    while not done:
        _, r, term, trunc, info = env.step(action)
        rewards.append(r)
        done = term or trunc
    return np.array(rewards), np.asarray(env.curve), info


def test_semivariance_reward_matches_formula_and_terminal_is_off_by_default():
    p = prices_15m()
    assert RewardConfig().reward == "semivariance" and RewardConfig().terminal_weight == 0.0
    env = make_test_env(p, NO_COST, RewardConfig(episode_days=2, risk_aversion=5.0), random_start=False)
    rewards, curve, info = run_episode(env, np.array([1.0, -1.0, 0.5], dtype=np.float32))
    r = curve[1:] / curve[:-1] - 1
    assert np.allclose(rewards, 100 * (r - 5.0 * np.minimum(r, 0) ** 2))   # no terminal bonus added
    assert "score" in info                                                    # but the score is still reported


def test_semivariance_penalty_is_quadratic_so_small_positions_are_cheap():
    p = prices_15m()
    out = {}
    for e in (0.2, 1.0):
        env = make_test_env(p, NO_COST, RewardConfig(episode_days=2, risk_aversion=5.0), random_start=False)
        rewards, curve, _ = run_episode(env, np.array([e, 0, 0], dtype=np.float32))
        r = curve[1:] / curve[:-1] - 1
        out[e] = (100 * r - rewards).sum()                                    # total risk penalty paid
    assert out[1.0] > 15 * out[0.2] > 0                                       # ~25x for 5x the position


def test_terminal_weight_adds_the_score_at_the_end():
    p = prices_15m()
    env = make_test_env(p, NO_COST, RewardConfig(episode_days=2, risk_aversion=0.0, terminal_weight=1.0),
                        random_start=False)
    rewards, curve, info = run_episode(env, np.array([0.5, 0, 0], dtype=np.float32))
    r = curve[1:] / curve[:-1] - 1
    assert np.allclose(rewards[:-1], 100 * r[:-1]) and abs(rewards[-1] - (100 * r[-1] + info["score"])) < 1e-9


def test_unknown_reward_mode_rejected():
    with pytest.raises(ValueError):
        RewardConfig(reward="sharpe")
