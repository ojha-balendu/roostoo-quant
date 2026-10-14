import numpy as np

from quantbot.backtest.threshold import ThresholdConfig, threshold_decider


def setup(scores, n=4, beta=1.0, risk=0.01):
    S = np.array(scores, dtype=float).reshape(-1, n)
    T = len(S)
    return S, np.full((T, n), risk), np.full((T, n), beta), np.ones((T, n), bool), np.ones(T, bool)


def test_entry_hysteresis_and_exit():
    S, R, B, E, C = setup([[0.9, 0.0, 0.0, 0.0],      # 0.9 < T: no entry
                           [1.2, 0.0, 0.0, 0.0],      # > T: enter long
                           [0.7, 0.0, 0.0, 0.0],      # between T/2 and T: keep (hysteresis)
                           [0.4, 0.0, 0.0, 0.0],      # < T/2: exit
                           [0.8, 0.0, 0.0, 0.0]])     # back between T/2 and T: no re-entry
    d = threshold_decider(ThresholdConfig(threshold=1.0), S, R, B, E, C, first=0)
    w = np.zeros(5)
    held = []
    for t in range(5):
        w = d(t, w)
        held.append(w[0])
    unit = 0.95 / 16
    assert held == [0.0, unit, unit, 0.0, 0.0]


def test_strongest_take_scarce_slots_and_hedge_cancels_beta():
    S, R, B, E, C = setup([[1.5, 3.0, 2.0, -2.5]])
    B[:] = [[1.2, 0.8, 1.0, 1.5]]
    d = threshold_decider(ThresholdConfig(threshold=1.0, max_per_side=2), S, R, B, E, C, first=0)
    w = d(0, np.zeros(5))
    assert w[1] > 0 and w[2] > 0 and w[0] == 0 and w[3] < 0                 # 3.0 and 2.0 beat 1.5 for 2 slots
    assert abs(np.dot(w[:4], B[0]) + w[4]) < 1e-12


def test_inverse_vol_size_and_gross_limit():
    S, R, B, E, C = setup([[2.0, 2.0, 2.0, 2.0]], beta=1.5)
    R[:] = [[0.005, 0.01, 0.02, 0.04]]
    tc = ThresholdConfig(threshold=1.0, sizing="inverse_vol", unit=0.2)
    w = threshold_decider(tc, S, R, B, E, C, first=0)(0, np.zeros(5))
    held = np.flatnonzero(w[:4])
    assert np.abs(w).sum() <= 0.95 + 1e-12                                   # hedge included
    assert w[0] == 0.25 and 0 in held                                         # calmest: 0.2 x 0.015/0.005 capped at 0.25
    assert len(held) < 4                                                      # entries skipped once gross is full


def test_no_resizing_while_held_and_nothing_between_checks():
    S, R, B, E, C = setup([[2.0, 0, 0, 0], [2.0, 0, 0, 0], [2.0, 0, 0, 0]])
    C[1] = False
    d = threshold_decider(ThresholdConfig(threshold=1.0), S, R, B, E, C, first=0)
    w0 = d(0, np.zeros(5))
    drifted = w0.copy()
    drifted[0] *= 1.07
    assert d(1, drifted) is drifted                                           # not a check bar
    w2 = d(2, drifted)
    assert w2[0] == drifted[0]                                                # still held at its drifted size
