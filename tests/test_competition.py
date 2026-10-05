import numpy as np
import pandas as pd

from quantbot.config import load_config
from quantbot.live.runner import is_rebalance_time
from quantbot.strategy.ensemble import Ensemble
from quantbot.strategy.registry import get_strategy
from quantbot.strategy.trend_vol import TrendVolTarget


def market(hours=24 * 60, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=hours, freq="1h", tz="UTC")
    f = rng.normal(0.0003, 0.008, hours)
    cols = {"BTC/USD": f, "ETH/USD": 1.2 * f + rng.normal(0, 0.004, hours), "SOL/USD": 1.4 * f + rng.normal(0, 0.006, hours)}
    return pd.DataFrame({k: 100 * np.exp(np.cumsum(v)) for k, v in cols.items()}, index=idx)


def test_ensemble_is_weighted_average_of_members():
    closes = market()
    a = {"lookbacks_hours": [24, 72], "long_in_bear": 1.0, "short_in_bull": 1.0}
    b = {"lookbacks_hours": [24, 72]}
    ens = Ensemble(members=[{"name": "trend_vol", "weight": 3, "params": a},
                            {"name": "trend_vol", "weight": 1, "params": b}])
    expected = 0.75 * TrendVolTarget(**a).generate_weights(closes) + 0.25 * TrendVolTarget(**b).generate_weights(closes)
    pd.testing.assert_frame_equal(ens.generate_weights(closes), expected)
    assert ens.warmup_bars == TrendVolTarget(**a).warmup_bars


def test_competition_config_builds_the_intended_portfolio():
    cfg = load_config("config/competition.yaml")
    s = get_strategy(cfg.strategy.name, cfg.strategy.params)
    assert isinstance(s, Ensemble) and len(s.members) == 2
    (m1, w1), (m2, w2) = s.members
    assert w1 == w2 == 0.5 and len(m1.assets) == len(m2.assets) == 15
    assert m1.long_in_bear == 1.0 and m2.long_in_bear == 0.0 and m1.lookbacks == [72, 168, 336]
    assert cfg.live.rebalance_utc_hours == [0] and cfg.live.warmup_bars >= s.warmup_bars
    assert cfg.execution.pause_between_orders_seconds >= 60
    assert cfg.accounting.short_collateral_in_usd_balance is True
    assert cfg.backtest.initial_capital == 1_000_000


def test_rebalance_schedule():
    assert is_rebalance_time(pd.Timestamp("2026-10-10 00:00", tz="UTC"), [0])
    assert not is_rebalance_time(pd.Timestamp("2026-10-10 01:00", tz="UTC"), [0])
    assert not is_rebalance_time(pd.Timestamp("2026-10-10 00:15", tz="UTC"), [0])
    assert is_rebalance_time(pd.Timestamp("2026-10-10 12:00", tz="UTC"), [0, 12])


def test_accounting_check_catches_wrong_collateral_setting():
    from quantbot.exchange.models import ShortPosition
    from quantbot.portfolio import PortfolioSnapshot

    short = {"ETH/USD": ShortPosition(1, "ETH/USD", 2500, 40, 100_000, 2500, 0.0)}
    # Collateral really is inside the USD balance (Roostoo, verified): consistent.
    ok = PortfolioSnapshot(0, usd_free=800_000, usd_lock=100_000, shorts=short, collateral_in_usd_balance=True)
    assert ok.accounting_consistent()
    # USD smaller than collateral: impossible if the setting were right -> flagged.
    bad = PortfolioSnapshot(0, usd_free=50_000, usd_lock=0, shorts=short, collateral_in_usd_balance=True)
    assert not bad.accounting_consistent()
    assert PortfolioSnapshot(0, 50_000, 0, shorts=short, collateral_in_usd_balance=False).accounting_consistent()
