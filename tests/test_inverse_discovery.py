from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from data.fetch_inverse_etfs import COHORT
from inverse_pair_discovery import (
    evaluate,
    pair_universe,
    prepare_fold,
    signals,
    signed_weights,
)


def test_macro_classes_and_designated_inverse_pairs():
    book = list(pair_universe(yaml.safe_load(COHORT.read_text()), ["AAPL"]))
    routes = {frozenset((a, b)): c for c, a, b, _ in book}
    assert len(routes) == len(book)
    assert routes[frozenset(("TBT", "TLT"))] == "inverse_etf"
    assert routes[frozenset(("FXE", "UUP"))] == "within_macro_class"
    assert all(
        routes[frozenset(pair)] == "cross_asset"
        for pair in [("GLD", "IEF"), ("FXE", "GLD"), ("UUP", "LQD")]
    )
    assert routes[frozenset(("AAPL", "MUB"))] == "stock_sector_hedge"
    np.testing.assert_allclose(signed_weights(-2), [1 / 3, 2 / 3])
    with pytest.raises(ValueError):
        signed_weights(0.5)


def test_discovery_uses_training_prices_and_membership_only():
    rng = np.random.default_rng(12)
    n = 420
    index = pd.bdate_range("2015-01-02", periods=n)
    returns = rng.normal(0, 0.01, n)
    px = pd.DataFrame(
        {
            s: 100 * np.cumprod(1 + r)
            for s, r in {
                "SPY": returns,
                "SH": -returns + rng.normal(0, 0.0005, n),
                "OLD": returns,
                "NEW": returns,
            }.items()
        },
        index=index,
    )
    dv = pd.DataFrame(1e9, index=index, columns=px.columns)
    members = pd.DataFrame({"OLD": True, "NEW": index > index[299]}, index=index)
    fold = SimpleNamespace(train_start=index[0], train_end=index[299])
    cfg = {
        "families": {"us": {"SH": -1, "SPY": 1}},
        "asset_classes": {"equity": ["us"]},
        "sectors": [],
    }
    screen, diag = prepare_fold(px, dv, members, cfg, fold)
    assert "OLD" in set(screen.a) and "NEW" not in set(screen.a)
    assert (diag.rho <= -0.8).all()
    px.iloc[300:, 0] *= 2
    changed, changed_diag = prepare_fold(px, dv, members, cfg, fold)
    pd.testing.assert_frame_equal(changed, screen)
    pd.testing.assert_frame_equal(changed_diag, diag)


def test_signal_lag_signed_execution_and_volume_limited_exit():
    index = pd.bdate_range("2020-01-01", periods=70)
    px = pd.DataFrame({"A": 100.0, "B": 100.0}, index=index)
    px.iloc[41, 0] = 110
    dv = pd.DataFrame(1e8, index=index, columns=px.columns)
    active = pd.Series(True, index=index)
    score = pd.Series(0.0, index=index)
    score.iloc[39] = -2
    result, trades = evaluate(px, dv, active, score, -1, 40, 45, 1)
    assert trades[0]["entry"] == index[40] and trades[0]["exit"] == index[41]
    assert result["gross_pnl"] == pytest.approx(-1250.0)
    assert result["execution"] == pytest.approx(0.0002 * (25000 + 26250))
    assert result["carry"] == pytest.approx(
        25000 * 0.025 * (index[41] - index[40]).days / 365
    )
    dv.iloc[41, 0] = 100
    invalid, _ = evaluate(px, dv, active, score, -1, 40, 45, 1)
    assert (
        not invalid["valid"] and invalid["invalid_reason"] == "exit_exceeds_volume_cap"
    )
    rng = np.random.default_rng(8)
    px.iloc[:] = np.exp(rng.normal(0, 0.01, (70, 2)).cumsum(axis=0))
    for mode in ["tracking", "stationary_spread"]:
        before = signals(px, mode, -1, 0)
        changed = px.copy()
        changed.iloc[49:] *= 2
        pd.testing.assert_series_equal(
            before.iloc[:49], signals(changed, mode, -1, 0).iloc[:49]
        )
