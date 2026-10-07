"""Binance USD-M futures data from Binance Vision: funding rates and 5-minute open-interest metrics.

  fundingRate  monthly files  .../futures/um/monthly/fundingRate/SYM/SYM-fundingRate-YYYY-MM.zip
               columns: calc_time, funding_interval_hours, last_funding_rate
  metrics      daily files    .../futures/um/daily/metrics/SYM/SYM-metrics-YYYY-MM-DD.zip
               columns: create_time, symbol, sum_open_interest, sum_open_interest_value,
                        count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
                        count_long_short_ratio, sum_taker_long_short_vol_ratio
Readers find columns by name when a header is present and fall back to the documented order
otherwise; timestamps may be milliseconds, microseconds or text.
Perpetuals of low-priced coins are quoted per 1,000 coins (PEPE -> 1000PEPEUSDT); rates and
percentage OI changes are unaffected by that multiplier.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
METRICS_COLS = ["create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
                "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
                "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
METRICS_RENAME = {"sum_open_interest": "oi", "sum_open_interest_value": "oi_value",
                  "sum_toptrader_long_short_ratio": "top_trader_ls", "count_long_short_ratio": "account_ls",
                  "sum_taker_long_short_vol_ratio": "taker_ls"}


def _read_csv_in_zip(path: str | Path, expected: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        with zf.open(zf.namelist()[0]) as fh:
            df = pd.read_csv(fh, header=None, dtype=str)
    if df.empty:
        return pd.DataFrame(columns=expected)
    first = [str(x).strip().lower() for x in df.iloc[0]]
    if not first[0].replace(".", "").replace("-", "").isdigit():   # header row present
        df.columns = first
        df = df.iloc[1:]
    else:
        df.columns = expected[: df.shape[1]] + [f"extra_{i}" for i in range(max(0, df.shape[1] - len(expected)))]
    return df.reset_index(drop=True)


def _to_utc(col: pd.Series) -> pd.Series:
    num = pd.to_numeric(col, errors="coerce")
    if num.notna().mean() > 0.9:
        v = num.to_numpy(dtype="float64")
        ms = np.where(v > 1e14, v / 1000, v)                       # microseconds -> milliseconds
        return pd.Series(pd.to_datetime(ms, unit="ms", utc=True), index=col.index)
    return pd.to_datetime(col, utc=True, errors="coerce")


def _find(cols: list[str], *needles: str, avoid: tuple[str, ...] = ()) -> str | None:
    for c in cols:
        if all(n in c for n in needles) and not any(a in c for a in avoid):
            return c
    return None


def read_funding_zip(path: str | Path) -> pd.DataFrame:
    df = _read_csv_in_zip(path, FUNDING_COLS)
    cols = list(df.columns)
    tcol = _find(cols, "time") or cols[0]
    rcol = _find(cols, "rate") or cols[-1]
    out = pd.DataFrame({"time": _to_utc(df[tcol]), "rate": pd.to_numeric(df[rcol], errors="coerce")})
    return out.dropna()


def read_metrics_zip(path: str | Path) -> pd.DataFrame:
    df = _read_csv_in_zip(path, METRICS_COLS)
    cols = list(df.columns)
    tcol = _find(cols, "time") or cols[0]
    out = pd.DataFrame({"time": _to_utc(df[tcol])})
    picks = {"oi": _find(cols, "open_interest", avoid=("value",)), "oi_value": _find(cols, "open_interest_value"),
             "top_trader_ls": _find(cols, "toptrader", "ratio", avoid=("count",)),
             "account_ls": _find(cols, "count_long_short_ratio") or _find(cols, "long_short_ratio", avoid=("top", "taker")),
             "taker_ls": _find(cols, "taker")}
    for name, col in picks.items():
        out[name] = pd.to_numeric(df[col], errors="coerce") if col else np.nan
    return out.dropna(subset=["time", "oi"])


READERS = {"fundingRate": read_funding_zip, "metrics": read_metrics_zip}


def process_futures(raw_dir: str | Path, processed_dir: str | Path, data_type: str, symbol: str) -> Path | None:
    files = sorted((Path(raw_dir) / "futures/um").glob(f"*/{data_type}/{symbol}/*.zip"))
    frames = []
    for f in files:
        try:
            frames.append(READERS[data_type](f))
        except (zipfile.BadZipFile, ValueError, KeyError) as exc:
            log.error("cannot read %s (%s)", f, exc)
    frames = [f for f in frames if len(f)]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).dropna(subset=["time"]).drop_duplicates("time").sort_values("time")
    out = Path(processed_dir) / "futures" / data_type / f"{symbol}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.set_index("time").to_parquet(out)
    return out


def perp_candidates(spot_symbol: str) -> list[str]:
    """Spot BTCUSDT -> [BTCUSDT]; PEPEUSDT -> [PEPEUSDT, 1000PEPEUSDT]; 1000CHEEMSUSDT -> [1000CHEEMSUSDT]."""
    if spot_symbol.startswith("1000"):
        return [spot_symbol]
    return [spot_symbol, "1000" + spot_symbol]


def symbols_path(processed_dir: str | Path) -> Path:
    return Path(processed_dir) / "futures" / "symbols.json"


def save_symbol_map(processed_dir: str | Path, mapping: dict[str, str | None]) -> None:
    p = symbols_path(processed_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(mapping, indent=2, sort_keys=True))


def load_symbol_map(processed_dir: str | Path) -> dict[str, str]:
    p = symbols_path(processed_dir)
    if not p.exists():
        raise SystemExit("no futures data yet: run  quantbot futures")
    return {k: v for k, v in json.loads(p.read_text()).items() if v}


def load_futures_panel(processed_dir: str | Path, data_type: str, column: str,
                       pairs: list[str] | None = None) -> pd.DataFrame:
    """time x pair panel of one column (irregular timestamps; NaN where a coin has no value)."""
    mapping = load_symbol_map(processed_dir)
    series = {}
    for pair, perp in mapping.items():
        if pairs is not None and pair not in pairs:
            continue
        f = Path(processed_dir) / "futures" / data_type / f"{perp}.parquet"
        if f.exists():
            s = pd.read_parquet(f, columns=[column])[column]
            series[pair] = s[~s.index.duplicated(keep="last")]
    return pd.DataFrame(series).sort_index() if series else pd.DataFrame()


def asof(panel: pd.DataFrame, index: pd.DatetimeIndex, tolerance: pd.Timedelta | None = None) -> pd.DataFrame:
    """Latest value at or before each time in `index`, per column (no future values)."""
    out = {}
    for col in panel.columns:
        s = panel[col].dropna()
        if s.empty:
            out[col] = pd.Series(np.nan, index=index)
            continue
        pos = s.index.searchsorted(index, side="right") - 1
        vals = np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, None)], np.nan)
        if tolerance is not None:
            ages = index - s.index[np.clip(pos, 0, None)]
            vals = np.where((pos >= 0) & (ages <= tolerance), vals, np.nan)
        out[col] = pd.Series(vals, index=index)
    return pd.DataFrame(out, index=index)
