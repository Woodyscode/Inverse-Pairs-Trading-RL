import numpy as np
import pandas as pd

from backtest.backtester import Backtester
from backtest.baselines import RandomPolicy
from env.inverse_pairs_env import InversePairsTradingEnv
from utils.pairs_selection import find_strong_pairs


def _pair_config(use_cointegration=True):
    return {
        "pairs": {
            "correlation_threshold": 0.7,
            "allow_positive": False,
            "allow_negative": True,
            "use_cointegration": use_cointegration,
            "cointegration_p_value": 0.05,
            "min_observations": 120,
            "traditional_min_annual_vol": None,
            "traditional_max_annual_vol": None,
            "inverse_min_annual_vol": None,
            "inverse_max_annual_vol": None,
            "max_traditional_pairs": 0,
            "max_inverse_pairs": None,
        }
    }


def _ohlcv_from_close(close):
    return {
        "Open": close.copy(),
        "High": close.copy(),
        "Low": close.copy(),
        "Close": close.copy(),
        "Adj Close": close.copy(),
        "Volume": pd.DataFrame(1_000_000, index=close.index, columns=close.columns),
    }


def test_known_inverse_cointegrated_pair_passes_selection():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=260)
    a = 100 + np.cumsum(rng.normal(0, 0.5, size=len(idx)))
    b = 220 - a + rng.normal(0, 0.05, size=len(idx))
    prices = pd.DataFrame({"A": a, "B": b}, index=idx)
    returns = prices.pct_change().dropna()

    traditional, inverse = find_strong_pairs(returns, prices, _pair_config())

    assert traditional == []
    assert ("A", "B") in inverse


def test_non_cointegrated_pair_fails_selection():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-01", periods=260)
    prices = pd.DataFrame(
        {
            "A": 100 + np.cumsum(rng.normal(0, 1.0, size=len(idx))),
            "B": 80 + np.cumsum(rng.normal(0, 1.0, size=len(idx))),
        },
        index=idx,
    )
    returns = prices.pct_change().dropna()

    _, inverse = find_strong_pairs(returns, prices, _pair_config())

    assert ("A", "B") not in inverse


def test_random_policy_on_independent_assets_not_persistently_profitable():
    rng = np.random.default_rng(123)
    idx = pd.bdate_range("2020-01-01", periods=90)
    close = pd.DataFrame(
        {
            "A": 100 * np.cumprod(1 + rng.normal(0, 0.002, size=len(idx))),
            "B": 100 * np.cumprod(1 + rng.normal(0, 0.002, size=len(idx))),
        },
        index=idx,
    )
    pair = ("A", "B")
    features = {pair: np.ones((len(idx), 2), dtype=np.float32)}
    config = {
        "env": {
            "window_size": 5,
            "initial_capital": 100000.0,
            "max_position": 0.25,
            "max_total_exposure": 1.0,
            "transaction_cost": 0.002,
            "slippage_cost": 0.002,
            "cost_model": "turnover",
            "borrow_fee_annual": 0.01,
            "financing_rate_annual": 0.01,
            "cash_rate_annual": 0.0,
            "liquidity": {"enabled": False},
        },
        "reward": {"reward_clip": 5.0},
        "system": {"seed": 123},
        "plotting": {k: False for k in [
            "equity_curve", "compare_spy", "final_allocation", "return_distribution",
            "monthly_heatmap", "drawdown_curve", "ulcer_index_curve", "cdar_plot",
            "rolling_sharpe", "rolling_sortino", "rolling_vol",
            "trade_pnl_distribution", "gain_to_pain",
        ]},
    }
    env = InversePairsTradingEnv(
        ohlcv=_ohlcv_from_close(close),
        pairs=[pair],
        pair_types={pair: "inverse"},
        config=config,
        price_index=idx,
        features=features,
        metrics={pair: [{} for _ in range(len(idx))]},
        feature_dims=[2],
    )
    results = Backtester(
        agent=RandomPolicy(env.action_space.shape[0], seed=99),
        env=env,
        config=config,
    ).run()

    assert results["metrics"]["final_value"] < config["env"]["initial_capital"]


def test_manual_price_path_expected_reward_with_frictionless_trade():
    idx = pd.bdate_range("2020-01-01", periods=4)
    close = pd.DataFrame({"A": [100, 100, 110, 110], "B": [100, 100, 100, 100]}, index=idx)
    pair = ("A", "B")
    features = {pair: np.ones((len(idx), 1), dtype=np.float32)}
    config = {
        "env": {
            "window_size": 1,
            "initial_capital": 100000.0,
            "max_position": 0.25,
            "max_total_exposure": 1.0,
            "transaction_cost": 0.0,
            "slippage_cost": 0.0,
            "cost_model": "turnover",
            "borrow_fee_annual": 0.0,
            "financing_rate_annual": 0.0,
            "cash_rate_annual": 0.0,
            "liquidity": {"enabled": False},
        },
        "reward": {"reward_clip": 5.0},
        "system": {"seed": 1},
    }
    env = InversePairsTradingEnv(
        ohlcv=_ohlcv_from_close(close),
        pairs=[pair],
        pair_types={pair: "inverse"},
        config=config,
        price_index=idx,
        features=features,
        metrics={pair: [{} for _ in range(len(idx))]},
        feature_dims=[1],
    )
    env.reset()

    _, reward, _, _, info = env.step(np.array([1.0, 1.0], dtype=np.float32))

    expected_pnl = (25000.0 / (100 + 100)) * 10
    expected_reward = (expected_pnl / 100000.0) - (0.01 * 0.25)
    assert np.isclose(info["pnl"], expected_pnl)
    assert np.isclose(reward, expected_reward)
