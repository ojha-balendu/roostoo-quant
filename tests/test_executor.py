from conftest import FakeClient, snapshot_from

from quantbot.config import ExecutionConfig, FeeConfig
from quantbot.execution.executor import Executor


def make_executor(client, pair_infos, dry_run=False):
    cfg = ExecutionConfig(pause_between_orders_seconds=0)
    return Executor(client, pair_infos, cfg, FeeConfig(), dry_run=dry_run, sleep=lambda s: None)


def test_long_and_short_from_flat(fake_client, pair_infos):
    ex = make_executor(fake_client, pair_infos)
    results = ex.rebalance({"BTC/USD": 0.1, "ETH/USD": -0.1}, snapshot_from(fake_client),
                           refresh=lambda: snapshot_from(fake_client))
    assert [r.status for r in results] == ["filled", "open"]
    buy = next(c for c in fake_client.calls if c[0] == "place_order")
    assert buy[2] == "BUY" and buy[3] == "0.200000"  # 10k / 50k at the ask, 6 dp
    short = next(c for c in fake_client.calls if c[0] == "short_open")
    assert short == ("short_open", "ETH/USD", "10000.00")


def test_reversal_sells_before_shorting(fake_client, pair_infos):
    ex = make_executor(fake_client, pair_infos)
    ex.rebalance({"BTC/USD": 0.2}, snapshot_from(fake_client), refresh=lambda: snapshot_from(fake_client))
    fake_client.calls.clear()
    ex.rebalance({"BTC/USD": -0.2}, snapshot_from(fake_client), refresh=lambda: snapshot_from(fake_client))
    assert [c[0] for c in fake_client.calls] == ["place_order", "short_open"]
    assert fake_client.calls[0][2] == "SELL"
    assert fake_client.coins["BTC"] == 0


def test_missing_targets_are_flattened(fake_client, pair_infos):
    ex = make_executor(fake_client, pair_infos)
    ex.rebalance({"BTC/USD": 0.1, "ETH/USD": -0.1}, snapshot_from(fake_client),
                 refresh=lambda: snapshot_from(fake_client))
    ex.rebalance({}, snapshot_from(fake_client), refresh=lambda: snapshot_from(fake_client))
    snap = snapshot_from(fake_client)
    assert not snap.pairs_with_positions()


def test_increases_scaled_to_cash(pair_infos):
    client = FakeClient(usd=1_000)
    ex = make_executor(client, pair_infos)
    # Ask for 2x equity: must be scaled down to what free USD can pay for.
    ex.rebalance({"BTC/USD": 1.0, "ETH/USD": -1.0}, snapshot_from(client), refresh=lambda: snapshot_from(client))
    assert client.usd >= 0


def test_dry_run_sends_nothing(fake_client, pair_infos):
    ex = make_executor(fake_client, pair_infos, dry_run=True)
    results = ex.rebalance({"BTC/USD": 0.1}, snapshot_from(fake_client))
    assert [r.status for r in results] == ["dry_run"]
    assert fake_client.calls == []


def test_equity_accounting_modes(fake_client, pair_infos):
    ex = make_executor(fake_client, pair_infos)
    ex.rebalance({"ETH/USD": -0.5}, snapshot_from(fake_client), refresh=lambda: snapshot_from(fake_client))
    snap = snapshot_from(fake_client, collateral_in_usd_balance=False)
    # The fake removes collateral from USD, so equity ~= 100k minus the 0.1% open fee.
    assert abs(snap.equity - (100_000 - 50)) < 1
