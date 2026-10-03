from quantbot.config import RiskConfig
from quantbot.risk import RiskManager


def test_clip_per_asset_and_gross():
    rm = RiskManager(RiskConfig(max_weight_per_asset=0.2, max_gross_leverage=0.5))
    out = rm.clip_targets({"A": 0.9, "B": -0.9, "C": 0.1, "X": 0.1}, tradable={"A", "B", "C"})
    assert "X" not in out
    assert abs(sum(abs(v) for v in out.values()) - 0.5) < 1e-9
    assert out["A"] > 0 > out["B"]


def test_drawdown_kill_and_resume():
    rm = RiskManager(RiskConfig(max_drawdown_kill=0.2, kill_cooldown_hours=1))
    assert rm.update_drawdown(100_000, now=0)
    assert rm.update_drawdown(110_000, now=10)
    assert not rm.update_drawdown(85_000, now=20)      # 22.7% below peak -> halt
    assert not rm.update_drawdown(90_000, now=3_000)   # still cooling down
    assert rm.update_drawdown(90_000, now=3_700)       # resumes with a fresh peak
    assert rm.update_drawdown(80_000, now=3_800)       # 11% below new peak: fine
