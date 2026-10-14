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


def opportunities(p: dict[str, pd.DataFrame], fee: float, margin_bps: float, horizon: int,
                  delay: int = 0) -> pd.DataFrame:
    """Trades the lag would allow, net of a round trip: when the reference mid is above Roostoo's ask by more
    than 2 x fee + margin at poll t, buy at Roostoo's ask at poll t + delay and sell at its bid `horizon`
    polls after that (and the mirror image for shorts). One trade at a time per pair.
    delay = 0 enters at the quote recorded in the same poll, which was fetched just BEFORE the reference
    price that triggered the trade: slightly optimistic. delay >= 1 enters at a later quote (realistic:
    an order needs time to arrive)."""
    thr = 2 * fee + margin_bps / 1e4
    rb, ra, bm = p["r_bid"].to_numpy(), p["r_ask"].to_numpy(), p["b_mid"].to_numpy()
    n_t = len(bm)
    with np.errstate(invalid="ignore"):
        long_sig = bm > ra * (1 + thr)
        short_sig = rb > bm * (1 + thr)
    rows = []
    for j, pair in enumerate(p["r_last"].columns):
        busy_until = -1
        for t in np.flatnonzero(long_sig[:, j] | short_sig[:, j]):       # only seconds with a signal
            e = t + delay                                                  # entry poll
            if e + horizon >= n_t or t <= busy_until:
                continue
            side = 1 if long_sig[t, j] else -1
            if side == 1:
                pnl = rb[e + horizon, j] / ra[e, j] - 1 - 2 * fee
                gap = bm[t, j] / ra[t, j] - 1
            else:
                pnl = rb[e, j] / ra[e + horizon, j] - 1 - 2 * fee
                gap = rb[t, j] / bm[t, j] - 1
            if np.isfinite(pnl):
                rows.append({"pair": pair, "poll": int(t), "side": side, "gap_bps": gap * 1e4, "net_bps": pnl * 1e4})
                busy_until = e + horizon
    return pd.DataFrame(rows, columns=["pair", "poll", "side", "gap_bps", "net_bps"])


# ---------------------------------------------------------------------------------------- margin sweep
def kelly_stats(net: np.ndarray, trades_per_day: float, cap: float, kelly_scale: float = 0.5) -> dict[str, float]:
    """Kelly sizing for many small trades with returns `net` (decimal, per unit of position).
    f* = mu / sigma^2 (continuous approximation). The conservative version uses mu minus one standard
    error. The size used is kelly_scale x conservative f*, capped at `cap` (share of equity per trade).
    Growth per trade at size f: f mu - f^2 sigma^2 / 2; per day: x trades per day."""
    n = len(net)
    out = {"n": n, "mu": np.nan, "sigma": np.nan, "win": np.nan, "kelly": np.nan, "kelly_cons": np.nan,
           "f": 0.0, "profit_day": 0.0, "growth_day": 0.0}
    if n == 0:
        return out
    mu = float(np.mean(net))
    out.update(mu=mu, win=float(np.mean(net > 0)))
    if n < 2:
        return out
    sigma = float(np.std(net, ddof=1))
    out["sigma"] = sigma
    if sigma <= 0:
        return out
    se = sigma / np.sqrt(n)
    out["kelly"] = mu / sigma ** 2
    mu_c = max(mu - se, 0.0)
    out["kelly_cons"] = mu_c / sigma ** 2
    f = float(min(kelly_scale * out["kelly_cons"], cap))
    out["f"] = f
    out["profit_day"] = f * mu * trades_per_day
    out["growth_day"] = trades_per_day * (f * mu - f ** 2 * sigma ** 2 / 2)
    return out


def margin_sweep(p: dict[str, pd.DataFrame], fee: float, margins, horizon: int, hours: float, cap: float,
                 kelly_scale: float = 0.5, delay: int = 0) -> pd.DataFrame:
    rows = []
    for m in margins:
        ops = opportunities(p, fee, m, horizon, delay)
        net = ops["net_bps"].to_numpy() / 1e4
        tph = len(ops) / hours if hours > 0 else np.nan
        k = kelly_stats(net, tph * 24, cap, kelly_scale)
        rows.append({"margin_bps": m, "entry_gap_bps": 2 * fee * 1e4 + m, "trades": len(ops), "per_hour": tph,
                     "pairs": ops["pair"].nunique() if len(ops) else 0,
                     "mean_gap_bps": ops["gap_bps"].mean() if len(ops) else np.nan, **k})
    return pd.DataFrame(rows)


# ----------------------------------------------------------- millisecond lag: Binance stream vs Roostoo reads
def roostoo_changes(reads: pd.DataFrame) -> pd.DataFrame:
    """From a table of Roostoo reads (ts in seconds, pair, mid), keep the moments its mid actually CHANGED.
    Returns columns pair, ts, old_mid, new_mid (one row per change), sorted by ts within pair."""
    out = []
    for pair, g in reads.sort_values("ts").groupby("pair", sort=False):
        mid = g["mid"].to_numpy()
        ts = g["ts"].to_numpy()
        changed = np.flatnonzero(np.diff(mid) != 0) + 1
        for i in changed:
            out.append({"pair": pair, "ts": ts[i], "old_mid": mid[i - 1], "new_mid": mid[i]})
    return pd.DataFrame(out, columns=["pair", "ts", "old_mid", "new_mid"])


def match_binance_move(r_change: dict, binance: pd.DataFrame, window_s: float, tol_frac: float = 0.25):
    """A Roostoo change is 'explained by Binance' if, in the window_s seconds BEFORE the Roostoo change,
    Binance's mid moved in the same direction and reached close to Roostoo's new level. Returns the lag in
    seconds (Roostoo change ts minus the earliest Binance update that already showed the move), or None.
    tol_frac: how much of the Roostoo step Binance must cover to count as the same move."""
    b = binance[binance["pair"] == r_change["pair"]]
    if b.empty:
        return None
    step = r_change["new_mid"] - r_change["old_mid"]
    if step == 0:
        return None
    lo = r_change["ts"] - window_s
    pre = b[(b["ts"] >= lo) & (b["ts"] <= r_change["ts"])]
    if pre.empty:
        return None
    target = r_change["old_mid"] + tol_frac * step                       # Binance must reach at least here
    bm = pre["mid"].to_numpy()
    bt = pre["ts"].to_numpy()
    reached = (bm >= target) if step > 0 else (bm <= target)
    if not reached.any():
        return None
    first = bt[np.argmax(reached)]                                       # earliest Binance ts past the target
    return float(r_change["ts"] - first)


def lag_profile(reads: pd.DataFrame, binance: pd.DataFrame, window_s: float = 5.0, tol_frac: float = 0.25
                ) -> pd.DataFrame:
    """For every Roostoo price change, how many ms earlier Binance already showed the move.
    Returns one row per matched change: pair, ts, lag_s. Unmatched changes (Roostoo moved without a
    matching Binance move in the window) are dropped, with their count available via attrs."""
    changes = roostoo_changes(reads)
    rows, unmatched = [], 0
    for _, c in changes.iterrows():
        lag = match_binance_move(c.to_dict(), binance, window_s, tol_frac)
        if lag is None:
            unmatched += 1
        else:
            rows.append({"pair": c["pair"], "ts": c["ts"], "lag_s": lag})
    out = pd.DataFrame(rows, columns=["pair", "ts", "lag_s"])
    out.attrs["n_changes"] = len(changes)
    out.attrs["n_unmatched"] = unmatched
    return out
