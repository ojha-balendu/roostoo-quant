"""Does Roostoo's price lag a real exchange? Analysis of a recorded price probe.

Input: a long table (one row per poll, source and pair) with columns
  poll (int), ts (receive time, seconds), source ("roostoo" | "binance"), pair, bid, ask, last
Roostoo is represented by its own bid/ask/last; the reference exchange by its mid price (bid+ask)/2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def panels(rec: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """poll x pair panels: roostoo bid/ask/last and reference mid, forward-filled."""
    r = rec[rec["source"] == "roostoo"]
    b = rec[rec["source"] == "binance"]
    out = {f"r_{c}": r.pivot_table(index="poll", columns="pair", values=c, aggfunc="last") for c in ("bid", "ask", "last")}
    mid = (b["bid"] + b["ask"]) / 2
    out["b_mid"] = b.assign(mid=mid).pivot_table(index="poll", columns="pair", values="mid", aggfunc="last")
    polls = sorted(set(rec["poll"]))
    pairs = sorted(set(out["r_last"].columns) & set(out["b_mid"].columns))
    out = {k: v.reindex(index=polls, columns=pairs).ffill() for k, v in out.items()}
    ts = rec.groupby("poll")["ts"].median().reindex(polls)
    out["seconds_per_poll"] = float(np.nanmedian(np.diff(ts.to_numpy()))) if len(ts) > 1 else np.nan
    return out


def update_stats(r_last: pd.DataFrame, seconds_per_poll: float) -> pd.DataFrame:
    """How often each Roostoo price changes between polls."""
    changed = r_last.diff().abs() > 0
    share = changed.iloc[1:].mean()
    gaps = {}
    for c in r_last.columns:
        idx = np.flatnonzero(changed[c].to_numpy())
        gaps[c] = float(np.median(np.diff(idx)) * seconds_per_poll) if len(idx) > 2 else np.nan
    return pd.DataFrame({"share_of_polls_changed": share, "median_seconds_between_changes": pd.Series(gaps)})


def lag_estimate(r_last: pd.DataFrame, b_mid: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    """For each pair, the delay k (in polls) at which Roostoo's price best matches the reference price
    k polls earlier (smallest mean absolute log difference), and how much better than k = 0 that is."""
    rows = {}
    lr, lb = np.log(r_last), np.log(b_mid)
    for c in r_last.columns:
        errs = []
        for k in range(max_lag + 1):
            d = (lr[c] - lb[c].shift(k)).abs().iloc[max_lag:]
            errs.append(float(d.mean()) if d.notna().any() else np.nan)
        errs = np.array(errs)
        if np.isfinite(errs).any():
            k = int(np.nanargmin(errs))
            rows[c] = {"lag_polls": k, "error_at_lag_bps": errs[k] * 1e4, "error_at_zero_bps": errs[0] * 1e4}
    return pd.DataFrame(rows).T


def opportunities(p: dict[str, pd.DataFrame], fee: float, margin_bps: float, horizon: int) -> pd.DataFrame:
    """Trades the lag would allow, net of a round trip: buy at Roostoo's ask when the reference mid is above
    it by more than 2 x fee + margin, sell at Roostoo's bid `horizon` polls later (and the mirror image for
    shorts). One trade at a time per pair."""
    thr = 2 * fee + margin_bps / 1e4
    rb, ra, bm = p["r_bid"].to_numpy(), p["r_ask"].to_numpy(), p["b_mid"].to_numpy()
    rows = []
    for j, pair in enumerate(p["r_last"].columns):
        busy_until = -1
        for t in range(len(bm) - horizon):
            if t <= busy_until or not (np.isfinite(ra[t, j]) and np.isfinite(rb[t, j]) and np.isfinite(bm[t, j])):
                continue
            side = 1 if bm[t, j] > ra[t, j] * (1 + thr) else (-1 if rb[t, j] > bm[t, j] * (1 + thr) else 0)
            if side == 0:
                continue
            if side == 1:
                pnl = rb[t + horizon, j] / ra[t, j] - 1 - 2 * fee
                gap = bm[t, j] / ra[t, j] - 1
            else:
                pnl = rb[t, j] / ra[t + horizon, j] - 1 - 2 * fee
                gap = rb[t, j] / bm[t, j] - 1
            if np.isfinite(pnl):
                rows.append({"pair": pair, "poll": t, "side": side, "gap_bps": gap * 1e4, "net_bps": pnl * 1e4})
                busy_until = t + horizon
    return pd.DataFrame(rows, columns=["pair", "poll", "side", "gap_bps", "net_bps"])
