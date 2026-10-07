"""In-sample / holdout windows for research scripts, with a one-shot holdout guard.

Protocol: the last ``holdout_days`` are held out.  Scripts evaluate on the
in-sample window by default; ``--evaluate-holdout`` runs once on the holdout
and leaves a marker file so a second look requires an explicit ``--force``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class StudyWindow:
    start: pd.Timestamp
    end: pd.Timestamp
    holdout_start: pd.Timestamp
    is_holdout: bool

    def describe(self) -> str:
        part = "HOLDOUT" if self.is_holdout else "in-sample"
        return f"{part} {self.start:%Y-%m-%d %H:%M} -> {self.end:%Y-%m-%d %H:%M} (holdout from {self.holdout_start:%Y-%m-%d})"


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def make_window(first: pd.Timestamp, last: pd.Timestamp, holdout_days: int, start: Optional[str] = None,
                evaluate_holdout: bool = False) -> StudyWindow:
    first, last = _utc(first), _utc(last)
    holdout_start = last - pd.Timedelta(days=holdout_days)
    if evaluate_holdout:
        return StudyWindow(holdout_start, last, holdout_start, True)
    s = max(first, _utc(start)) if start else first
    if s >= holdout_start:
        raise ValueError("in-sample window is empty: start is after the holdout start")
    return StudyWindow(s, holdout_start, holdout_start, False)


def claim_holdout(out_dir: Path, study: str, force: bool = False) -> Path:
    """Raise if the holdout of ``study`` was already used (unless ``force``); else mark it used."""
    marker = Path(out_dir) / f".holdout_used_{study}"
    if marker.exists() and not force:
        raise RuntimeError(f"holdout for {study} already evaluated on {marker.read_text().strip()} "
                           f"({marker}); pass --force if you really mean to look again")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(pd.Timestamp.now(tz="UTC").isoformat() + "\n")
    return marker
