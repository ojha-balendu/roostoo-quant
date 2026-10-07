"""Technical indicators on (time x coin) panels, using only past data.

Windows are in BARS (on 4-hour bars, 14 bars ~ 2.3 days). Indicators that depend on price level
or volume units are normalised so coins can be ranked against each other:
  MACD and ATR are divided by price; OBV, accumulation/distribution and ease of movement are
  turned into flows relative to recent volume; volume is relative to its own recent average.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


def ema(x: pd.DataFrame, span: int) -> pd.DataFrame:
    return x.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _per_column(x: pd.DataFrame, n: int, reducer) -> pd.DataFrame:
    """Apply reducer(windows[T-n+1, n]) -> values[T-n+1] to each column's trailing windows of length n."""
    a = x.to_numpy(dtype=float)
    out = np.full(a.shape, np.nan)
    if len(a) >= n:
        for j in range(a.shape[1]):
            out[n - 1:, j] = reducer(sliding_window_view(a[:, j], n))
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def rsi(c, n=14):
    d = c.diff()
    d = d.where(d.abs() > 1e-12 * c.abs(), 0.0).where(d.notna())  # floating-point noise is not a move
    up, dn = wilder(d.clip(lower=0), n), wilder((-d).clip(lower=0), n)
    return _clean(100 - 100 / (1 + up / dn)).where(up.notna())


def williams_r(h, l, c, n=14):
    hh, ll = h.rolling(n).max(), l.rolling(n).min()
    return _clean(-100 * (hh - c) / (hh - ll))


def volatility(c, n):
    return np.log(c).diff().rolling(n).std()


def trend(c, short, long, norm_span):
    r = np.log(c).diff()
    return _clean((r.ewm(span=short, adjust=False).mean() - r.ewm(span=long, adjust=False).mean())
                  / r.ewm(span=norm_span, adjust=False, min_periods=norm_span).std())


def atr_pct(h, l, c, n=14):
    pc = c.shift(1)
    tr = np.fmax(np.fmax(h - l, (h - pc).abs()), (l - pc).abs())
    return _clean(wilder(tr, n) / c)


def macd(c):
    return _clean(ema(ema(c, 12) - ema(c, 26), 9) / c)


def trix(c, n=15):
    return ema(ema(ema(c, n), n), n).pct_change(fill_method=None) * 100


def cci(h, l, c, n=20):
    tp = (h + l + c) / 3
    sma = tp.rolling(n).mean()
    mad = _per_column(tp, n, lambda w: np.abs(w - w.mean(axis=1, keepdims=True)).mean(axis=1))
    return _clean((tp - sma) / (0.015 * mad))


def cmo(c, n=14):
    d = c.diff()
    up, dn = d.clip(lower=0).rolling(n).sum(), (-d).clip(lower=0).rolling(n).sum()
    return _clean(100 * (up - dn) / (up + dn))


def ichimoku(h, l, c):
    conv = (h.rolling(9).max() + l.rolling(9).min()) / 2
    base = (h.rolling(26).max() + l.rolling(26).min()) / 2
    return _clean(c / ((conv + base) / 2) - 1)


def kst(c):
    def roc(n):
        return c.pct_change(n, fill_method=None) * 100
    return (roc(10).rolling(10).mean() + 2 * roc(15).rolling(10).mean() + 3 * roc(20).rolling(10).mean()
            + 4 * roc(30).rolling(15).mean())


def ultimate(h, l, c):
    pc = c.shift(1)
    low, high = np.fmin(l, pc), np.fmax(h, pc)
    bp, tr = c - low, high - low

    def avg(n):
        return bp.rolling(n).sum() / tr.rolling(n).sum()
    return _clean(100 * (4 * avg(7) + 2 * avg(14) + avg(28)) / 7)


def _arg_extreme(w: np.ndarray, fn) -> np.ndarray:
    ok = np.isfinite(w).all(axis=1)
    return np.where(ok, fn(np.where(np.isfinite(w), w, 0.0), axis=1).astype(float), np.nan)


def aroon_up_down(h, l, n=26) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aroon up / down, 0..100: 100 = the n-bar high (low) is the current bar."""
    up = 100 * _per_column(h, n + 1, lambda w: _arg_extreme(w, np.argmax)) / n
    dn = 100 * _per_column(l, n + 1, lambda w: _arg_extreme(w, np.argmin)) / n
    return up, dn


def aroon(h, l, n=26):
    """Aroon up minus Aroon down, -100..100 (positive = recent highs are more recent than lows)."""
    up, dn = aroon_up_down(h, l, n)
    return up - dn


def stochastic(h, l, c, n=14):
    hh, ll = h.rolling(n).max(), l.rolling(n).min()
    return _clean(100 * (c - ll) / (hh - ll))


def obv_flow(c, v, n=20):
    """On-balance volume change over n bars, as a share of volume traded: -1 (all down) .. +1 (all up)."""
    return _clean((np.sign(c.diff()) * v).rolling(n).sum() / v.rolling(n).sum())


def ease_of_movement(h, l, v, n=14):
    mid = (h + l) / 2
    rel_vol = v / v.rolling(n).mean()
    return _clean((mid.diff() / mid) * ((h - l) / mid) / rel_vol).rolling(n).mean()


def money_flow_volume(h, l, c, v):
    return _clean(((c - l) - (h - c)) / (h - l)) * v


def chaikin_money_flow(h, l, c, v, n=20):
    return _clean(money_flow_volume(h, l, c, v).rolling(n).sum() / v.rolling(n).sum())


def accumulation_distribution(h, l, c, v, n=60):
    """Change of the accumulation/distribution line over n bars, relative to volume traded."""
    return _clean(money_flow_volume(h, l, c, v).rolling(n).sum() / v.rolling(n).sum())


def relative_volume(qv, n=20):
    return _clean(np.log(qv / qv.rolling(n).mean()))


def taker_buy_ratio(tbq, qv, n=6):
    """Share of traded value where buyers crossed the spread, minus 0.5 (order-flow proxy)."""
    return _clean(tbq.rolling(n).sum() / qv.rolling(n).sum() - 0.5)


def compute_all(o, h, l, c, v, qv=None, tbq=None) -> dict[str, pd.DataFrame]:
    out = {
        "rsi": rsi(c), "williams_r": williams_r(h, l, c), "volatility_20": volatility(c, 20),
        "volatility_60": volatility(c, 60), "trend_1_3": trend(c, 1, 3, 3), "trend_5_20": trend(c, 5, 20, 20),
        "trend_20_60": trend(c, 20, 60, 20), "atr_pct": atr_pct(h, l, c), "macd": macd(c), "trix": trix(c),
        "cci": cci(h, l, c), "cmo": cmo(c), "ichimoku": ichimoku(h, l, c), "kst": kst(c),
        "ultimate": ultimate(h, l, c), "aroon": aroon(h, l), "stochastic": stochastic(h, l, c),
        "obv_flow": obv_flow(c, v), "ease_of_movement": ease_of_movement(h, l, v),
        "chaikin_money_flow": chaikin_money_flow(h, l, c, v),
        "accumulation_distribution": accumulation_distribution(h, l, c, v),
        "relative_volume": relative_volume(qv if qv is not None else v),
    }
    if tbq is not None and qv is not None:
        out["taker_buy_ratio"] = taker_buy_ratio(tbq, qv)
    return out


# --------------------------------------------------------------------------------------------------
# The indicator set used by the composite signal, keyed by the names in the coefficient dictionary.
# Price-level indicators are divided by price (MACD lines) or expressed relative to price (Ichimoku),
# cumulative ones become flows relative to volume (OBV, accumulation/distribution), and "volume"
# is log USD volume per bar, so that coins can be compared with each other.

def macd_parts(c):
    line = ema(c, 12) - ema(c, 26)
    signal = ema(line, 9)
    return _clean(line / c), _clean(signal / c), _clean((line - signal) / c)


def ichimoku_parts(h, l, c):
    conv = (h.rolling(9).max() + l.rolling(9).min()) / 2
    base = (h.rolling(26).max() + l.rolling(26).min()) / 2
    lead_a = (conv + base) / 2                                  # value computed now (not shifted forward)
    lead_b = (h.rolling(52).max() + l.rolling(52).min()) / 2
    return tuple(_clean(x / c - 1) for x in (conv, base, lead_a, lead_b))


def compute_named(o, h, l, c, v, qv) -> dict[str, pd.DataFrame]:
    """All indicators named as in the composite coefficient dictionary."""
    r = rsi(c)
    m_line, m_sig, m_hist = macd_parts(c)
    i_conv, i_base, i_a, i_b = ichimoku_parts(h, l, c)
    a_up, a_dn = aroon_up_down(h, l)
    k = stochastic(h, l, c)
    return {
        "relative_strength_index": r, "rsi": r, "williams_r": williams_r(h, l, c),
        "volatility_20": volatility(c, 20), "volatility_60": volatility(c, 60),
        "trend_1_3": trend(c, 1, 3, 3), "trend_5_20": trend(c, 5, 20, 20), "trend_20_60": trend(c, 20, 60, 20),
        "average_true_range": atr_pct(h, l, c), "macd": m_line, "macd_signal": m_sig, "macd_histogram": m_hist,
        "trix": trix(c), "commodity_channel_index": cci(h, l, c), "chande_momentum_oscillator": cmo(c),
        "ichimoku_conversion": i_conv, "ichimoku_base": i_base, "ichimoku_leading_a": i_a, "ichimoku_leading_b": i_b,
        "know_sure_thing": kst(c), "ultimate_oscillator": ultimate(h, l, c),
        "aroon_up": a_up, "aroon_down": a_dn, "aroon_oscillator": a_up - a_dn,
        "stochastic_k": k, "stochastic_d": k.rolling(3).mean(),
        "on_balance_volume": obv_flow(c, v), "ease_of_movement": ease_of_movement(h, l, v),
        "chaikin_money_flow": chaikin_money_flow(h, l, c, v),
        "accumulation_distribution_index": accumulation_distribution(h, l, c, v),
        "volume": _clean(np.log(qv)),
    }
