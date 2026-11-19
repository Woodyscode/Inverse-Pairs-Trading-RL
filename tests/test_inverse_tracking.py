from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.discovery.inverse_tracking_study import (
    carry_years,
    effective_beta,
    enter,
    exit_trade,
    expected_growth,
    price_units,
    thresholds,
)


def test_reset_compounds_daily_and_drifts_intraday():
    ref = SimpleNamespace(
        total=np.array([0.1, -1 / 11, 0]), from_reset=np.zeros((3, 1))
    )
    assert expected_growth(ref, -1, 0, 0, 2, 0) == pytest.approx(0.9 * (1 + 1 / 11))
    assert effective_beta(-1, 0.1, -0.1) == pytest.approx(-1.1 / 0.9)
    assert expected_growth(ref, -1, 1, 0, 1, 0) == pytest.approx(1)


def test_threshold_and_unit_selection_only_use_training_history():
    raw = np.ones(400) * 20
    split = np.ones(400)
    split[80:] = 0.2
    fit = np.arange(400) < 300
    assert price_units(raw, raw, split, fit)[0] == "split_adjusted"
    raw[80:] /= 0.2
    assert price_units(raw, np.ones(400) * 20, split, fit)[0] == "as_traded"
    r = np.sin(np.arange(400))
    before = thresholds(r, fit)
    r[~fit] += 1000
    assert thresholds(r, fit) == before


def test_matched_exit_uses_observed_fills_both_legs_and_distributions():
    calendar = pd.bdate_range("2024-01-05", periods=7)

    def tape():
        return SimpleNamespace(
            price=np.full((7, 390), 100.0),
            dv=np.full((7, 390), 1e8),
            adv=np.full(7, 1e9),
            div=np.zeros(7),
            close=np.full(7, 100.0),
            total=np.zeros(7),
            from_reset=np.zeros((7, 390)),
        )

    a, b = tape(), tape()
    a.price[0, 1] = np.nan
    tapes = {"A": a, "B": b}
    pair = ("A", "B")
    entry = enter(tapes, pair, 0, 1, -1)
    assert entry[0] == 2
    a.div[1] = 0.1
    a.price[1, 2:] = 101.0
    a.dv[1, 2] = 1000
    result, last = exit_trade(tapes, pair, calendar, 0, entry, -1, -1, 1, 2, 7)
    assert last == 1 and result["exit_minute"] == 3
    assert result["gross"] == pytest.approx(-137.5)
    assert result["execution"] == pytest.approx(0.0002 * (25000 + 25125))
    assert result["borrow"] > 0
    assert result["net"] == pytest.approx(
        result["gross"] - result["execution"] - result["funding"] - result["borrow"]
    )
    funding, _ = carry_years(
        calendar, 0, 2, 1, 2, np.array([12500, 12500]), np.array([[12500, 12500]])
    )
    assert funding == pytest.approx(25000 * 0.02 * 3 / 365)
