import pandas as pd

from quantbot.config import load_config
from quantbot.live.bars import BarBuilder, parse_interval
from quantbot.live.store import StateStore


def test_bar_builder_emits_on_boundary():
    bb = BarBuilder("1m")
    t0 = 1_700_000_040_000 - 1_700_000_040_000 % 60_000
    assert bb.update(t0 + 1_000, {"BTC/USD": 10}) is None
    assert bb.update(t0 + 30_000, {"BTC/USD": 12}) is None
    assert bb.update(t0 + 50_000, {"BTC/USD": 9}) is None
    ts, df = bb.update(t0 + 61_000, {"BTC/USD": 11})
    assert ts == pd.Timestamp(t0, unit="ms", tz="UTC")
    assert df.loc["BTC/USD"].tolist() == [10, 12, 9, 9]


def test_parse_interval():
    assert parse_interval("15m") == pd.Timedelta(minutes=15)
    assert parse_interval("4h") == pd.Timedelta(hours=4)


def test_store_roundtrip():
    s = StateStore(":memory:")
    s.set("peak_equity", "123.5")
    assert s.get_float("peak_equity", 0) == 123.5
    bars = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5]}, index=["BTC/USD"])
    s.upsert_bars(1_700_000_000_000, bars)
    panel = s.load_close_panel()
    assert panel["BTC/USD"].iloc[0] == 1.5


def test_repo_config_loads():
    cfg = load_config("config/config.yaml")
    assert cfg.live.dry_run is True
    assert cfg.fees.taker == 0.001 and cfg.fees.maker == 0.0005
    assert cfg.data.start == "2023-01-01"


def test_tradable_requires_live_price():
    from conftest import make_pair

    from quantbot.config import Config
    from quantbot.live.runner import LiveRunner
    from quantbot.strategy.flat import Flat

    r = LiveRunner(Config(), client=None, strategy=Flat(), universe={}, store=StateStore(":memory:"))
    r.pair_infos = {p: make_pair(p) for p in ["BTC/USD", "OMNI/USD", "TON/USD"]}
    assert r.tradable == {"BTC/USD", "OMNI/USD", "TON/USD"}  # before the first ticker
    r._live_pairs = {"BTC/USD"}
    assert r.tradable == {"BTC/USD"}
