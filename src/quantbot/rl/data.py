"""Loading 15-minute bars for the RL universe and splitting them in time.

  |-- warm-up --|------ train (fit) ------|-- validation --|------ test ------|
                                                                     ^ end of data
Training episodes never reach into validation; validation (used only to pick the best
checkpoint) never reaches into test; test is touched only by the evaluation script.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantbot.config import Config
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe
from quantbot.rl.features import WARMUP_BARS

MAJORS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "BNB/USD", "DOGE/USD", "ADA/USD", "LINK/USD",
          "AVAX/USD", "LTC/USD", "DOT/USD", "SUI/USD", "NEAR/USD", "UNI/USD", "AAVE/USD"]
BAR = pd.Timedelta(minutes=15)


@dataclass(frozen=True)
class Split:
    train_start: pd.Timestamp
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    end: pd.Timestamp

    def to_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> Split:
        return cls(**{k: pd.Timestamp(v) for k, v in d.items()})


def make_split(end: pd.Timestamp, train_days: float, val_days: float, test_days: float) -> Split:
    test_start = end - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)
    return Split(val_start - pd.Timedelta(days=train_days), val_start, test_start, end)


def load_15m(cfg: Config, assets: list[str], start: pd.Timestamp) -> pd.DataFrame:
    """15-minute closes (bar labelled by its open time) from the 1-minute data."""
    universe = {p: s for p, s in load_universe(cfg.data.universe_path).items() if p in assets}
    missing = sorted(set(assets) - set(universe))
    if missing:
        raise SystemExit(f"not in the universe file: {missing}")
    one_min = load_panel(cfg.data.processed_dir, cfg.data.market, "1m", universe, start=str(start.date()))
    if one_min.empty:
        raise SystemExit("no 1m data: run  quantbot download --interval 1m --start <date>")
    closes = one_min.resample("15min").last().ffill(limit=4)
    return closes.reindex(columns=[a for a in assets if a in closes.columns])


def data_start_for(split: Split) -> pd.Timestamp:
    return split.train_start - BAR * (WARMUP_BARS + 96)


def check_coverage(closes: pd.DataFrame, split: Split) -> None:
    need = data_start_for(split) + BAR * 96
    if closes.index.min() > need:
        days = (closes.index.max() - closes.index.min()).days
        raise SystemExit(f"1m data starts {closes.index.min():%Y-%m-%d} but this split needs data from "
                         f"{need:%Y-%m-%d} ({days} days available). Download more 1m history or use "
                         "smaller --train-days / --val-days / --test-days.")


def index_of(index: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    """Position of the first bar at or after ts."""
    return int(index.searchsorted(ts, side="left"))


def window_starts(lo: int, hi_end: int, ep_len: int, step: int) -> list[int]:
    """Episode start indices in [lo, ...] whose episodes finish by hi_end (exclusive)."""
    return list(range(lo, hi_end - ep_len + 1, step))
