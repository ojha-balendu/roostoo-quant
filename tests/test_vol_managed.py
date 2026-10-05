import numpy as np
import pandas as pd

from quantbot.strategy.registry import get_strategy
from quantbot.strategy.vol_managed import VolManaged


def regimes(hours=24 * 120, seed=3):
    """BTC with calm and turbulent stretches (vol x4), plus a low-vol second asset."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=hours, freq="1h", tz="UTC")
    vol = np.where((np.arange(hours) // (24 * 15)) % 2 == 0, 0.004, 0.016)
    btc = rng.normal(0, 1, hours) * vol
    gold = rng.normal(0, 0.002, hours)
    return pd.DataFrame({"BTC/USD": 100 * np.exp(np.cumsum(btc)), "PAXG/USD": 100 * np.exp(np.cumsum(gold))},
                        index=idx)


def test_exposure_drops_in_turbulent_regimes_and_is_capped():
    closes = regimes()
    w = VolManaged(assets=["BTC/USD"], target_vol=0.4, halflife_hours=24).generate_weights(closes)["BTC/USD"]
    calm = w[(np.arange(len(w)) // (24 * 15)) % 2 == 0].iloc[24 * 20:]
    wild = w[(np.arange(len(w)) // (24 * 15)) % 2 == 1]
    assert calm.median() > 2 * wild.median()
    assert (w <= 0.95 + 1e-12).all() and (w >= 0).all()


def test_inverse_vol_base_favours_calm_asset():
    closes = regimes()
    s = VolManaged(assets=["BTC/USD", "PAXG/USD"], base="inverse_vol", target_vol=0.3)
    w = s.generate_weights(closes).iloc[-1]
    assert w["PAXG/USD"] > w["BTC/USD"] > 0


def test_no_lookahead_and_registered():
    closes = regimes(hours=24 * 40)
    s = get_strategy("vol_managed", {"assets": ["BTC/USD"], "halflife_hours": 24, "min_hours": 48})
    full = s.generate_weights(closes)
    cut = closes.index[-300]
    pd.testing.assert_frame_equal(full.loc[:cut], s.generate_weights(closes.loc[:cut]), check_freq=False)


def test_managed_portfolio_has_steadier_risk():
    closes = regimes(hours=24 * 240)
    w = VolManaged(assets=["BTC/USD"], target_vol=0.2, halflife_hours=24).generate_weights(closes)["BTC/USD"]
    r = np.log(closes["BTC/USD"]).diff()
    managed = (w.shift(1) * r).dropna()
    constant = (w.mean() * r).dropna()
    roll = lambda x: x.rolling(24 * 7).std().dropna()  # noqa: E731
    assert roll(managed).std() / roll(managed).mean() < 0.5 * roll(constant).std() / roll(constant).mean()
