from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.convergence.inverse_convergence_study import PAIR_WEIGHTS, events
from experiments.discovery.inverse_tracking_study import enter_weighted
from experiments.forecasting.strict_alpha_study import marked_trade


def tape(n, trend, morning):
    return SimpleNamespace(
        total=np.full(n, trend),
        from_reset=np.full((n, 390), morning),
        price=np.full((n, 390), 100.0),
        dv=np.full((n, 390), 1e8),
        adv=np.full(n, 1e9),
        close=np.full(n, 100.0),
        div=np.zeros(n),
    )


def test_contraction_buys_falling_prior_leader_and_rejects_crossings():
    a, b = tape(90, 0.002, -0.004), tape(90, -0.001, 0.005)
    f = events(a, b)
    assert f.convergence.iloc[-1] and f.side_a.iloc[-1] == 1
    reverse = events(b, a)
    assert reverse.convergence.iloc[-1] and reverse.side_a.iloc[-1] == -1
    a.from_reset[-1, 59] = -0.12
    assert not events(a, b).convergence.iloc[-1]
    a.from_reset[-1, 59] = -0.004
    b.from_reset[-1, 59] = -0.002
    assert not events(a, b).convergence.iloc[-1]
    a.total[-5:] += 1
    pd.testing.assert_series_equal(events(a, b).iloc[-5], f.iloc[-5])


@pytest.mark.parametrize(
    "weights,expected_gross", [(PAIR_WEIGHTS, 500.0), (np.array([1.0, 0.0]), 250.0)]
)
def test_opposite_leg_and_leader_control_accounting(weights, expected_gross):
    cal = pd.bdate_range("2024-02-01", periods=4)
    tapes = {"a": tape(4, 0.002, 0.0), "b": tape(4, -0.001, 0.0)}
    entry = enter_weighted(tapes, ("a", "b"), 0, 61, weights)
    tapes["a"].price[1, 61:] = 101.0
    tapes["b"].price[1, 61:] = 97.0
    result, _, _, fills = marked_trade(cal, tapes, ("a", "b"), 0, entry, 1, None, 1, 3)
    assert result["gross"] == pytest.approx(expected_gross)
    assert result["execution"] == pytest.approx(
        sum(abs(x["notional"]) for x in fills) * 0.0002
    )
    assert (result["borrow"] > 0) == (weights[1] < 0)
    assert fills[0]["units"] > 0 and fills[1]["units"] <= 0
