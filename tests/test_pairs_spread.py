import numpy as np
import pandas as pd

from quantbot.strategy.pairs_spread import PairsSpread
from quantbot.strategy.registry import get_strategy


def synthetic(minutes=60 * 24 * 10, beta=1.2, ou_phi=0.995, seed=1):
    """1m prices where log(ETH) = beta*log(BTC) + mean-reverting noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=minutes, freq="1min", tz="UTC")
    lb = np.cumsum(rng.normal(0, 0.0008, minutes))
    noise = np.zeros(minutes)
    for i in range(1, minutes):
        noise[i] = ou_phi * noise[i - 1] + rng.normal(0, 0.0006)
    return pd.DataFrame({"ETH/USD": np.exp(beta * lb + noise) * 2500, "BTC/USD": np.exp(lb) * 80000}, index=idx)


def test_entries_respect_fee_rule_and_legs_hedge():
    closes = synthetic()
    s = PairsSpread(lookback_hours=24, fee_multiple=2.0)
    sig = s.signals(closes)
    entries = sig[sig["event"] == "enter"]
    assert len(entries) > 0
    assert (entries["edge"] >= 2.0 * entries["cost"] - 1e-12).all()
    assert (entries["z"].abs() >= 2.0).all()
    w = s.generate_weights(closes)
    live = w[w["ETH/USD"] != 0]
    ratio = -live["BTC/USD"] / live["ETH/USD"]  # BTC leg = -beta x ETH leg
    assert (ratio > 0.5).all() and (ratio < 2.5).all()
    assert np.allclose(live.abs().sum(axis=1), 0.95)


def test_every_trade_exits_for_a_reason():
    sig = PairsSpread(lookback_hours=24).signals(synthetic())
    events = sig["event"][sig["event"] != ""].tolist()
    for a, b in zip(events, events[1:], strict=False):
        assert (a == "enter") != (b == "enter")  # strictly alternates enter / exit
    assert set(events) <= {"enter", "revert", "stop", "timeout"}


def test_higher_fee_multiple_means_fewer_trades():
    closes = synthetic()
    n = [int((PairsSpread(lookback_hours=24, fee_multiple=k).signals(closes)["event"] == "enter").sum())
         for k in (0.0, 2.0, 6.0)]
    assert n[0] >= n[1] >= n[2]


def test_no_lookahead_and_registered():
    closes = synthetic(minutes=60 * 24 * 5)
    s = get_strategy("pairs_spread", {"lookback_hours": 24})
    full = s.generate_weights(closes)
    cut = closes.index[-700]
    pd.testing.assert_frame_equal(full.loc[:cut], s.generate_weights(closes.loc[:cut]), check_freq=False)
