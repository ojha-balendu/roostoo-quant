from quantbot.execution.decision import Action, decide

KW = dict(min_trade_usd=50, min_close_usd=2, threshold=0.02)


def actions(intents):
    return [i.action for i in intents]


def test_open_long_from_flat():
    (i,) = decide("BTC/USD", 0, 0, 100, 10_000, 0.2, **KW)
    assert i.action is Action.LONG and abs(i.notional - 2_000) < 1e-9


def test_flip_long_to_short_exits_first():
    out = decide("BTC/USD", 20, 0, 100, 10_000, -0.1, **KW)
    assert actions(out) == [Action.SELL, Action.SHORT]
    assert out[0].full and abs(out[1].notional - 1_000) < 1e-9


def test_flip_short_to_long():
    out = decide("BTC/USD", 0, 10, 100, 10_000, 0.1, **KW)
    assert actions(out) == [Action.CLOSE, Action.LONG]


def test_flat_target_closes_everything():
    assert actions(decide("BTC/USD", 5, 0, 100, 10_000, 0.0, **KW)) == [Action.SELL]
    assert actions(decide("BTC/USD", 0, 5, 100, 10_000, 0.0, **KW)) == [Action.CLOSE]


def test_within_threshold_no_trade():
    assert decide("BTC/USD", 20, 0, 100, 10_000, 0.21, **KW) == []


def test_partial_reductions():
    (i,) = decide("BTC/USD", 30, 0, 100, 10_000, 0.1, **KW)
    assert i.action is Action.SELL and not i.full and abs(i.quantity - 20) < 1e-9
    (i,) = decide("BTC/USD", 0, 30, 100, 10_000, -0.1, **KW)
    assert i.action is Action.CLOSE and not i.full and abs(i.quantity - 20) < 1e-9


def test_dust_is_ignored():
    assert decide("BTC/USD", 0.001, 0, 100, 10_000, 0.0, **KW) == []
