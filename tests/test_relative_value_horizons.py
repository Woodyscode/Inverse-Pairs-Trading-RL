import numpy as np
import pandas as pd

from data.minute_features import paired_moments
from relative_value_horizons import CAPITAL, FEE, basket_label, fit_predict, simulate


def test_label_and_costs_follow_actual_hold():
    index = pd.bdate_range("2021-01-01", periods=9)
    px = pd.DataFrame(
        {"A": [60, 80, 100, 104, 106, 110, 112, 114, 114], "B": [100.0] * 9},
        index=index,
    )
    beta = pd.Series(1.0, index=index)
    label = basket_label(px, beta, 5)
    # First execution is day 2, and the five-session exit is day 7.
    np.testing.assert_allclose(label.iloc[1], 0.07)
    warmup = pd.DataFrame(
        60.0, index=pd.bdate_range(end=index[0], periods=21)[:-1], columns=px.columns
    )
    extended = pd.concat([warmup, px])
    bars, trades = simulate(
        extended,
        extended * 1e6,
        pd.Series(True, index=extended.index),
        pd.Series(1.0, index=extended.index),
        pd.Series(1.0, index=extended.index),
        22,
        29,
        5,
        CAPITAL,
    )
    assert len(trades) == 1 and trades[0]["exit"] == index[7]
    np.testing.assert_allclose(trades[0]["gross_basket_return"], label.iloc[1])
    np.testing.assert_allclose(bars.gross_pnl.sum(), CAPITAL * 0.25 * label.iloc[1])
    np.testing.assert_allclose(
        bars.execution.sum(), FEE * (25000 + 12500 * 1.14 + 12500)
    )
    np.testing.assert_allclose(
        bars.equity.iloc[-1] - CAPITAL,
        (bars.gross_pnl - bars.execution - bars.carry).sum(),
    )


def test_fit_purges_every_label_that_crosses_training_boundary():
    n, boundary, horizon = 270, 200, 10
    x = pd.DataFrame({"x": np.sin(np.arange(n))})
    y = x.x.copy()
    eligible = pd.Series(True, index=x.index)
    expected = fit_predict(x, y, boundary, horizon, eligible)
    changed = y.copy()
    changed.iloc[boundary - horizon - 1 :] += 100
    pd.testing.assert_series_equal(
        expected, fit_predict(x, changed, boundary, horizon, eligible)
    )
    changed.iloc[boundary - horizon - 2] += 100
    assert not np.allclose(
        expected, fit_predict(x, changed, boundary, horizon, eligible)
    )


def test_minute_variance_excludes_gap_and_overnight_jumps():
    index = pd.to_datetime(
        [
            "2021-01-04 09:30",
            "2021-01-04 09:31",
            "2021-01-04 09:34",
            "2021-01-04 09:35",
            "2021-01-05 09:30",
            "2021-01-05 09:31",
        ]
    )
    prices = pd.DataFrame(
        {"a": [1.0, 2.0, 100.0, 110.0, 999.0, 999.0], "b": 1.0}, index=index
    )
    m = paired_moments(prices)
    assert list(m.synchronized_returns) == [2, 1]
    np.testing.assert_allclose(m.aa, [np.log(2) ** 2 + np.log(1.1) ** 2, 0.0])
