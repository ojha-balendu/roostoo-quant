import numpy as np
import pandas as pd

from quantbot.strategy.registry import get_strategy


def test_xs_momentum_shapes_and_live_parity():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024", periods=300, freq="1h", tz="UTC")
    closes = pd.DataFrame(100 * np.exp(rng.normal(0, 0.01, (300, 8)).cumsum(0)), index=idx,
                          columns=[f"C{i}/USD" for i in range(8)])
    s = get_strategy("xs_momentum", {"lookback_bars": 24, "top_n": 2, "bottom_n": 2, "gross": 0.8})
    w = s.generate_weights(closes)
    assert w.shape == closes.shape
    assert (w.iloc[:24] == 0).all().all()                       # no look-back yet
    assert np.allclose(w.iloc[30:].abs().sum(axis=1), 0.8)      # gross respected
    assert ((w.iloc[30:] > 0).sum(axis=1) == 2).all() and ((w.iloc[30:] < 0).sum(axis=1) == 2).all()
    # Live uses the last row of the same computation.
    assert s.target_weights(closes) == {k: v for k, v in w.iloc[-1].items() if v != 0}
