import numpy as np
import pandas as pd

from quantbot.backtest.engine import run_backtest
from quantbot.backtest.metrics import compute_metrics
from quantbot.config import FeeConfig


def panel(prices):
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="1h", tz="UTC")
    return pd.DataFrame({"BTC/USD": prices}, index=idx)


def test_flat_strategy_keeps_capital():
    closes = panel(np.linspace(100, 200, 50))
    res = run_backtest(closes, closes * 0.0, spread_bps=0)
    assert np.allclose(res.equity, 100_000) and res.metrics["n_trades"] == 0


def test_long_captures_rise_minus_fees():
    closes = panel(np.linspace(100, 200, 50))
    res = run_backtest(closes, closes * 0 + 1.0, spread_bps=0, delay_bars=0, cash_buffer=0.0, threshold=0.0)
    # Buys at 100 with 0.1% fee, then rides to 200.
    expected = 100_000 / 1.001 * 2
    assert abs(res.equity.iloc[-1] - expected) / expected < 1e-3


def test_short_profits_on_fall_and_pays_both_fees():
    closes = panel(np.linspace(100, 50, 50))
    w = closes * 0 - 0.5
    w.iloc[-1] = 0.0  # close at the end
    res = run_backtest(closes, w, spread_bps=0, delay_bars=0, threshold=0.0, fees=FeeConfig())
    actions = list(res.trades["action"])
    assert actions[0] == "SHORT" and actions[-1] == "CLOSE"
    assert res.equity.iloc[-1] > 100_000 * 1.2


def test_metrics_basic():
    eq = pd.Series(np.linspace(100, 110, 100), index=pd.date_range("2024", periods=100, freq="1D", tz="UTC"))
    m = compute_metrics(eq)
    assert m["max_drawdown"] == 0 and m["total_return"] > 0.099 and m["sharpe"] > 0
