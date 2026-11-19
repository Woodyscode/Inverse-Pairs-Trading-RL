from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from env.spread_control import SpreadControlEnv
from inverse_tracking_study import enter
from strict_alpha_study import FEE, feature_frame, marked_trade, opportunities


def make_tapes(n):
    rng = np.random.default_rng(7)
    out = {}
    for symbol in ("A", "B"):
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.0001, (n, 390)), axis=1))
        close = prices[:, -1]
        out[symbol] = SimpleNamespace(
            price=prices,
            open=prices[:, 0],
            close=close,
            div=np.zeros(n),
            total=pd.Series(close).pct_change().to_numpy(copy=True),
            dv=np.full_like(prices, 2e7),
            adv=np.full(n, 1e9),
            overnight=np.zeros(n),
            from_reset=prices / 100 - 1,
        )
    return out


def test_opening_signal_cannot_see_future_fills_or_close():
    tapes = make_tapes(65)
    a, b = tapes.values()
    cols = feature_frame(a, b, -1.0, 0.0).columns
    before = feature_frame(a, b, -1.0, 0.0).loc[50, cols]
    a.price[50, 60:] *= 2
    a.total[50:] += 1
    a.dv[50, 60:] *= 4
    pd.testing.assert_series_equal(before, feature_frame(a, b, -1.0, 0.0).loc[50, cols])


def test_forecast_labels_use_next_observed_prices_and_distributions():
    tapes = make_tapes(65)
    cal = pd.bdate_range("2023-01-02", periods=65)
    a, b = tapes.values()
    a.div[51] = 1.0
    a.price[50, 61] = np.nan
    f = opportunities(
        tapes, [{"a": "A", "b": "B", "beta": -1.0, "intercept": 0.0}], cal, 1
    )
    row = f[f.day.eq(50)].iloc[0]
    expected = (
        (a.price[51, 61] + 1) / a.price[50, 62]
        - 1
        - row.hedge * (b.price[51, 61] / b.price[50, 62] - 1)
    )
    assert row.label == pytest.approx(expected / (1 - row.hedge))
    assert row.label_end == cal[51]


def test_partial_exits_reconcile_per_leg_fees_daily_marks_and_intraday_exposure():
    tapes = make_tapes(7)
    cal = pd.bdate_range("2024-01-05", periods=7)
    entry = enter(tapes, ("A", "B"), 0, 61, -1.0)
    tapes["A"].dv[1, 61] = 2000.0
    tapes["A"].div[1] = 0.18
    result, flat, marks, fills = marked_trade(
        cal, tapes, ("A", "B"), 0, entry, -1, -1.0, 1, 6
    )
    assert flat == 1 and len(fills) > 4
    assert sum(v["fee"] for v in fills) == pytest.approx(result["execution"])
    assert sum(abs(v["notional"]) for v in fills) * FEE == pytest.approx(
        result["execution"]
    )
    assert result["borrow"] > 0 and marks[0][5] > 0 and marks[1][5] == 0
    assert all(v[6] > 0 for v in marks.values())
    assert sum(v[0] - sum(v[1:4]) for v in marks.values()) == pytest.approx(
        result["net"]
    )


def test_control_hold_halve_and_blocked_short():
    n = 12
    data = {
        "features": np.zeros((n, 8)),
        "prices": np.ones((n, 2)) * 100,
        "alpha": np.zeros((n, 2)),
        "vol": np.zeros(n),
        "beta": -np.ones(n),
        "blocked": np.zeros(2, dtype=bool),
        "years": np.zeros(n),
        "forecast_uncertainty": np.zeros(n),
        "regime": np.zeros(n),
        "adv": np.ones((n, 2)) * 1e8,
    }
    settings = {
        "portfolio_control": True,
        "gross": 0.25,
        "cost_rate": 0.0,
        "funding": 0.0,
        "borrow": 0.0,
        "cash": 0.0,
        "adv_fraction": 0.01,
        "lambda_risk": 1.0,
        "lambda_turn": 0.0,
    }
    env = SpreadControlEnv(data, 1, n, settings, 100000, True)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    env.step(2)
    original = env.shares.copy()
    env.step(3)
    np.testing.assert_allclose(env.shares, original)
    env.step(4)
    np.testing.assert_allclose(env.shares, original / 2)
    data["blocked"][0] = True
    env.step(0)
    np.testing.assert_allclose(env.shares, 0)
