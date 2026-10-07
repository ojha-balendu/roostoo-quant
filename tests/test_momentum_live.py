import time

import numpy as np
import pandas as pd

from quantbot.analysis.xsection import eligibility, residual_returns
from quantbot.backtest.composite_bt import simulate
from quantbot.backtest.momentum import MomentumConfig, momentum_inputs, sleeve_weights
from quantbot.rl.sim import SimConfig, simple_returns
from quantbot.strategy.momentum_sleeve import MomentumSleeve


def market(days=70, n=16, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-07-01", periods=24 * days, freq="1h", tz="UTC")
    btc = rng.normal(0, 0.006, len(idx))
    cols = {"BTC/USD": btc} | {f"C{i:02d}/USD": (0.8 + 0.05 * i) * btc + rng.normal(rng.normal(0, 2e-4), 0.008, len(idx))
                               for i in range(n)}
    close = pd.DataFrame({k: 100 * np.exp(np.cumsum(v)) for k, v in cols.items()}, index=idx)
    qv = pd.DataFrame(5e6, index=idx, columns=close.columns)
    return close, qv


def test_live_decisions_equal_the_backtest_at_every_rebalance():
    close, qv = market()
    s = MomentumSleeve()
    s.set_context(quote_volume=qv)
    replay = s.generate_weights(close)
    # the research path (research/18 variant C through the simulator)
    resid, beta = residual_returns(close, "BTC/USD")
    coins = list(resid.columns)
    elig = eligibility(close, qv)[coins]
    mc = MomentumConfig(residual_riskadj=True, buffer_k=10)
    x = {k: v.reindex(columns=coins).to_numpy(dtype=float) for k, v in momentum_inputs(close, resid, elig, mc).items()}
    P = close[coins + ["BTC/USD"]].to_numpy()
    R, B, E = simple_returns(P), beta[coins].to_numpy(), elig.to_numpy(dtype=bool)
    daily = np.asarray((close.index + pd.Timedelta(hours=1)).hour == 0)
    targets = {}

    def decide(t, cur):
        if daily[t]:
            targets[t] = sleeve_weights(t, cur, x, B, E, R, mc)
            return targets[t]
        return cur
    cfg = SimConfig(initial_capital=1e6, fee=0.001, spread_bps=2, max_gross=1, max_weight=1, trade_band=0.005)
    simulate(P, close.index, decide, 0, len(close) - 1, cfg)
    traded = [t for t, w in targets.items() if np.abs(w).sum() > 0]
    assert len(traded) >= 30
    for t in traded:
        assert np.allclose(replay.iloc[t][coins + ["BTC/USD"]].to_numpy(), targets[t]), close.index[t]
    # and live target_weights with holdings = what the backtest held reproduces the next decision
    t_last = traded[-1]
    held = dict(zip(coins + ["BTC/USD"], targets[traded[-2]], strict=True))
    s.set_context(quote_volume=qv, holdings=held)
    live = s.target_weights(close.iloc[: t_last + 1])
    assert np.allclose([live.get(p, 0.0) for p in coins + ["BTC/USD"]], targets[t_last])


def test_illiquid_and_untradable_coins_are_never_picked():
    close, qv = market()
    s = MomentumSleeve()
    qv_low = qv.copy()
    qv_low["C00/USD"] = 1_000.0                                           # far below $200k per hour
    every = [c for c in close.columns if c != "BTC/USD"]
    tradable = set(close.columns) - {"C01/USD"}
    s.set_context(quote_volume=qv_low, holdings={}, tradable=tradable)
    w = s.target_weights(close)
    assert "C00/USD" not in w and "C01/USD" not in w
    assert sum(1 for p in w if p != "BTC/USD") == 10 and set(w) - {"BTC/USD"} <= set(every)
    assert abs(sum(abs(v) for v in w.values()) - 0.95) < 1e-9


def test_buffer_uses_exchange_holdings():
    close, qv = market()
    s = MomentumSleeve()
    s.set_context(quote_volume=qv, holdings={})
    fresh = s.target_weights(close)
    longs = sorted((p for p, v in fresh.items() if v > 0 and p != "BTC/USD"))
    s2 = MomentumSleeve(buffer_k=None)
    s2.set_context(quote_volume=qv, holdings={})
    assert set(p for p, v in s2.target_weights(close).items() if v > 0 and p != "BTC/USD") == set(longs)
    resid, _ = residual_returns(close, "BTC/USD")
    x = momentum_inputs(close, resid, eligibility(close, qv)[list(resid.columns)], s.mc)
    ranked = x["score"].iloc[-1].sort_values(ascending=False).index
    sixth = ranked[6]                                                       # ranked 7th: outside top 5, inside top 10
    s.set_context(quote_volume=qv, holdings={sixth: 0.09})
    assert s.target_weights(close).get(sixth, 0) > 0                         # held -> kept by the buffer
    s.set_context(quote_volume=qv, holdings={sixth: 1e-7})                   # exchange dust after a sale
    assert s.target_weights(close).get(sixth, 0) == 0                        # dust is not a holding


def test_warmup_fetch_returns_close_and_usd_volume(monkeypatch):
    from quantbot.live import warmup

    now = int(time.time() * 1000)
    opens = [now - 2 * 3_600_000 + i * 3_600_000 for i in range(3)]          # the last bar is still open
    rows = [[o, "1", "1", "1", str(100 + i), "2", o + 3_600_000 - 1, str(1000.0 * (i + 1)), 5, "1", "1", "0"]
            for i, o in enumerate(opens)]

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return rows

    class FakeSession:
        def get(self, *_a, **_k):
            return FakeResponse()

    monkeypatch.setattr(warmup.requests, "Session", FakeSession)
    panels = warmup.warmup_panels({"BTC/USD": "BTCUSDT"}, "1h", 3)
    assert list(panels["close"]["BTC/USD"]) == [100.0, 101.0]               # the still-open bar is dropped
    assert list(panels["quote_volume"]["BTC/USD"]) == [1000.0, 2000.0]


def test_runner_hands_volumes_holdings_and_tradable_pairs_to_the_strategy():
    from quantbot.config import Config
    from quantbot.live.runner import LiveRunner

    close, qv = market(days=60)
    seen = {}

    class Spy(MomentumSleeve):
        def target_weights(self, closes):
            seen.update(self.context)
            return {}

    class Snap:
        equity = 1e6
        usd_total = 1e6

        def weights(self):
            return {"C03/USD": 0.09}

        def accounting_consistent(self):
            return True

    class Pass:
        def update_drawdown(self, equity):
            return True

        def clip_targets(self, t, tradable=None):
            return t

    class NoStore:
        def record_equity(self, snap):
            return None

    class NoExec:
        def rebalance(self, targets, snap, refresh=None):
            return []

    r = LiveRunner.__new__(LiveRunner)
    r.cfg, r.strategy, r.closes, r.quote_volume = Config(), Spy(), close, qv
    r.risk, r.store, r.executor, r._volume_refreshed = Pass(), NoStore(), NoExec(), time.time()
    r._bars_since_rebalance = 0
    r._snapshot = lambda tickers=None: Snap()
    type(r).tradable = property(lambda self: {"BTC/USD", "C03/USD"})
    try:
        r._rebalance({})
    finally:
        del type(r).tradable
    assert seen["holdings"] == {"C03/USD": 0.09} and seen["tradable"] == {"BTC/USD", "C03/USD"}
    assert seen["quote_volume"] is qv
