from types import SimpleNamespace

import numpy as np
import pandas as pd

from short_leg_diagnostics import exposure, half_spread, opening_hour
from short_leg_execution_evidence import audit


def test_inverse_pair_reinforces_factor_and_signed_hedge_removes_it():
    x = np.random.default_rng(13).normal(size=(60, 6))
    y = np.column_stack([x[:, 0], -2 * x[:, 0]])
    betas, ratio, slope = exposure(x, y)
    np.testing.assert_allclose(ratio, 1.5)
    np.testing.assert_allclose(slope, -0.5)
    np.testing.assert_allclose(betas[0] - slope * betas[1], 0, atol=1e-12)


def test_opening_hour_excludes_overnight_and_later_bars():
    prices = np.ones((1, 390))
    prices[:, 59] = 102.0
    t = SimpleNamespace(open=np.array([100.0]), price=prices)
    np.testing.assert_allclose(opening_hour(t), 0.02)


def test_spread_uses_only_prior_sessions():
    prices = np.tile([100.0, 100.1] * 30, (25, 1))
    before = half_spread(prices)
    prices[20] = 110
    after = half_spread(prices)
    assert np.isfinite(before[20]) and before[20] == after[20]
    assert before[21] != after[21]


def test_future_quotes_and_locates_cannot_confirm_a_fill(tmp_path):
    fills = pd.DataFrame({
        "fold": 2025, "split": "test", "scope": "all", "book": "pair",
        "entry_date": pd.Timestamp("2025-03-10"), "date": pd.Timestamp("2025-03-10"),
        "minute": [61, 62], "symbol": "TEST", "units": [-2, 2], "price": 100.0,
    })
    quotes = tmp_path / "quotes.csv"
    pd.DataFrame({
        "symbol": "TEST", "ts": ["2025-03-10T14:31:57Z", "2025-03-10T14:32:00Z",
                                 "2025-03-10T14:32:59Z"],
        "bid": 49.5, "ask": 50.5, "bid_size": 4, "ask_size": 4, "price_multiplier": 2,
    }).to_csv(quotes, index=False)
    locates = tmp_path / "locates.csv"
    loan = pd.DataFrame({
        "symbol": ["TEST"], "available_at": ["2025-03-10T14:30:01Z"],
        "valid_until": ["2025-03-10T16:00:00Z"], "max_shares": [4], "price_multiplier": [2],
    })
    loan.to_csv(locates, index=False)
    result = audit(fills, quotes, locates)
    assert list(result.quote_state) == ["unknown", "covered"]
    assert list(result.locate_state) == ["unknown", "not_required"]
    assert result.quote_pnl_change.iloc[1] == -2.0
    loan["available_at"] = "2025-03-10T14:29:59Z"
    loan.to_csv(locates, index=False)
    assert audit(fills, quotes, locates).locate_state.iloc[0] == "documented"
    loan["max_shares"] = 3
    loan.to_csv(locates, index=False)
    assert audit(fills, quotes, locates).locate_state.iloc[0] == "insufficient"
