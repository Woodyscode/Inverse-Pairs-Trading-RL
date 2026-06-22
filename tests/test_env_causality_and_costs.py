import numpy as np
import pandas as pd

from env.inverse_pairs_env import InversePairsTradingEnv


def _make_env(
    close,
    transaction_cost=0.0,
    slippage_cost=0.0,
    adj_close=None,
    price_type="Adj Close",
    signal_lag_bars=0,
):
    dates = pd.bdate_range("2020-01-01", periods=len(close["A"]))
    close_df = pd.DataFrame(close, index=dates, dtype=float)
    adj_close_df = pd.DataFrame(adj_close, index=dates, dtype=float) if adj_close is not None else close_df.copy()
    ohlcv = {
        "Open": close_df.copy(),
        "High": close_df.copy(),
        "Low": close_df.copy(),
        "Close": close_df.copy(),
        "Adj Close": adj_close_df.copy(),
        "Volume": pd.DataFrame(1_000_000, index=dates, columns=close_df.columns),
    }
    pair = ("A", "B")
    features = {pair: np.arange(len(dates), dtype=np.float32).reshape(-1, 1)}
    config = {
        "env": {
            "window_size": 1,
            "initial_capital": 100000.0,
            "max_position": 0.25,
            "max_total_exposure": 1.0,
            "transaction_cost": transaction_cost,
            "slippage_cost": slippage_cost,
            "cost_model": "turnover",
            "borrow_fee_annual": 0.0,
            "financing_rate_annual": 0.0,
            "cash_rate_annual": 0.0,
            "liquidity": {"enabled": False},
            "signal_lag_bars": signal_lag_bars,
        },
        "data": {"price_type": price_type},
        "reward": {"reward_clip": 5.0},
        "system": {"seed": 123},
    }
    return InversePairsTradingEnv(
        ohlcv=ohlcv,
        pairs=[pair],
        pair_types={pair: "inverse"},
        config=config,
        price_index=dates,
        features=features,
        metrics={pair: []},
        feature_dims=[1],
    )


def test_action_does_not_profit_from_already_observed_bar():
    env = _make_env({"A": [100, 200, 200], "B": [100, 100, 100]})
    obs, _ = env.reset()

    assert obs[0] == 1.0
    _, _, _, _, info = env.step(np.array([1.0, 1.0], dtype=np.float32))

    assert info["gross_pnl"] == 0.0
    assert info["pnl"] == 0.0


def test_signal_lag_uses_previous_feature_row_for_daily_close_signals():
    env = _make_env(
        {"A": [100, 100, 100], "B": [100, 100, 100]},
        signal_lag_bars=1,
    )
    obs, _ = env.reset()

    assert obs[0] == 0.0


def test_env_pnl_uses_configured_price_type():
    env = _make_env(
        close={"A": [100, 100, 100], "B": [100, 100, 100]},
        adj_close={"A": [100, 200, 220], "B": [100, 100, 100]},
        price_type="Adj Close",
    )
    env.reset()

    _, _, _, _, info = env.step(np.array([1.0, 1.0], dtype=np.float32))

    expected = 0.25 * 100000.0 * ((220 - 200) / 200)
    assert np.isclose(info["gross_pnl"], expected)


def test_action_reward_uses_future_interval_after_observation():
    env = _make_env({"A": [100, 200, 220], "B": [100, 100, 100]})
    env.reset()

    _, _, _, _, info = env.step(np.array([1.0, 1.0], dtype=np.float32))

    expected = 0.25 * 100000.0 * ((220 - 200) / 200)
    assert np.isclose(info["gross_pnl"], expected)
    assert np.isclose(info["pnl"], expected)


def test_costs_scale_with_turnover_not_profit():
    env = _make_env(
        {"A": [100, 100, 100, 100, 100], "B": [100, 100, 100, 100, 100]},
        transaction_cost=0.001,
        slippage_cost=0.001,
    )
    env.reset()

    _, _, _, _, entry = env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert np.isclose(entry["turnover"], 25000.0)
    assert np.isclose(entry["transaction_costs"], 50.0)
    assert np.isclose(entry["pnl"], -50.0)

    _, _, _, _, hold = env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert np.isclose(hold["turnover"], 0.0)
    assert np.isclose(hold["transaction_costs"], 0.0)

    _, _, _, _, exit_info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert exit_info["gross_pnl"] == 0.0
    assert exit_info["turnover"] > 0.0
    assert exit_info["transaction_costs"] > 0.0


def test_borrow_financing_and_liquidity_assumptions_apply():
    dates = pd.bdate_range("2020-01-01", periods=4)
    close_df = pd.DataFrame({"A": [100, 100, 100, 100], "B": [100, 100, 100, 100]}, index=dates)
    ohlcv = {
        "Open": close_df.copy(),
        "High": close_df.copy(),
        "Low": close_df.copy(),
        "Close": close_df.copy(),
        "Adj Close": close_df.copy(),
        "Volume": pd.DataFrame(1000, index=dates, columns=close_df.columns),
    }
    pair = ("A", "B")
    features = {pair: np.ones((len(dates), 1), dtype=np.float32)}
    config = {
        "env": {
            "window_size": 1,
            "initial_capital": 100000.0,
            "max_position": 0.25,
            "max_total_exposure": 1.0,
            "transaction_cost": 0.0,
            "slippage_cost": 0.0,
            "cost_model": "turnover",
            "borrow_fee_annual": 0.0252,
            "financing_rate_annual": 0.0252,
            "cash_rate_annual": 0.0,
            "liquidity": {
                "enabled": True,
                "adv_window": 2,
                "max_trade_pct_adv": 0.01,
            },
        },
        "reward": {"reward_clip": 5.0},
        "system": {"seed": 123},
    }

    env = InversePairsTradingEnv(
        ohlcv=ohlcv,
        pairs=[pair],
        pair_types={pair: "inverse"},
        config=config,
        price_index=dates,
        features=features,
        metrics={pair: []},
        feature_dims=[1],
    )
    env.reset()
    _, _, _, _, info = env.step(np.array([1.0, 1.0], dtype=np.float32))

    assert info["turnover"] < 25000.0
    assert info["borrow_cost"] > 0.0
    assert info["financing_cost"] > 0.0
    assert info["pnl"] < 0.0
