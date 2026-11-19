import numpy as np
import pytest

from env.features.residual import alpha_forecasts, residual_state
from env.spread_control import SpreadControlEnv
from experiments.portfolio_control.residual_control_study import synthetic


def test_log_hedge_features_and_forecasts_respect_the_information_cutoff():
    prices, volume, _, _ = synthetic(7)
    prices, volume = prices.iloc[:480].copy(), volume.iloc[:480].copy()
    train_end = 380
    state, target = residual_state(prices, volume, train_end)
    revised = prices.copy()
    revised.iloc[350:, 0] *= 1.2
    future, _ = residual_state(revised, volume, train_end)
    np.testing.assert_allclose(state.iloc[:350], future.iloc[:350], equal_nan=True)
    assert np.isclose(state.beta.iloc[-1], state.beta.iloc[train_end - 1])
    h = state.beta.iloc[280]
    ret = prices.iloc[282] / prices.iloc[281] - 1
    assert np.isclose(target[280], (ret.iloc[0] - h * ret.iloc[1]) / (1 + abs(h)))
    first = alpha_forecasts(state, target, train_end)
    # Forecast at the first block boundary can only use labels through boundary - 2.
    poisoned = target.copy()
    poisoned[255:] = 999
    second = alpha_forecasts(state, poisoned, train_end)
    np.testing.assert_allclose(first[256:319], second[256:319])


def small_env(prices):
    p = np.array(prices, dtype=float)
    n = len(p)
    data = {
        "prices": p,
        "features": np.zeros((n, 8)),
        "alpha": np.zeros((n, 2)),
        "beta": np.full(n, -2.0),
        "vol": np.full(n, 0.01),
        "years": np.zeros(n),
        "adv": np.full((n, 2), 1e9),
        "blocked": np.zeros(2, dtype=bool),
    }
    cfg = {
        "gross": 0.3,
        "cost_rate": 0.001,
        "funding": 0,
        "borrow": 0,
        "cash": 0,
        "lambda_risk": 2,
        "lambda_turn": 0.0001,
        "adv_fraction": 0.01,
    }
    return SpreadControlEnv(data, 1, n, cfg, 100000)


def test_signed_dollar_hedge_inventory_timing_and_reward_reconciliation():
    env = small_env([[100, 100], [100, 100], [110, 95], [110, 95]])
    env.reset()
    obs, reward, done, _, info = env.step(2)
    assert not done
    np.testing.assert_allclose(env.shares, [100, 200])
    assert info["gross_pnl"] == 0
    assert info["costs"] == 30
    assert env.equity == 99970
    assert reward == pytest.approx(
        100 * (-30 / 1e5 - 2 * (0.3 * 0.01) ** 2 - 0.0001 * 0.3)
    )
    # Position is marked at entry at signal time even though the next close is known to the simulator.
    assert obs[-2] == 0
    _, _, done, _, info = env.step(1)
    assert done and np.all(env.shares == 0)
    assert info["costs"] == 30


def test_partial_fills_keep_the_signed_dollar_hedge_and_zero_volume_blocks_entry():
    env = small_env([[100, 100]] * 4)
    env.data["adv"][0] = [1e6, 1e6]
    env.reset()
    _, _, _, _, info = env.step(0)
    np.testing.assert_allclose(env.shares, [-50, -100])
    assert info["fill"] == 0.5
    env.reset()
    env.data["adv"][0, 0] = 0
    _, _, _, _, info = env.step(0)
    assert info["gross"] == 0


def test_realized_volume_limits_fills_after_the_signal_and_wind_down_closes_inventory():
    env = small_env([[100, 100]] * 6)
    env.settings.update(strict_volume=True, unwind_sessions=1)
    env.data["dollars"] = np.full((6, 2), 1e8)
    env.data["dollars"][1, 0] = 0.0
    env.reset()
    _, _, _, _, info = env.step(2)
    assert info["filled_notional"] == 0
    env.step(2)
    np.testing.assert_allclose(env.shares, [100, 200])
    env.step(2)
    _, _, done, _, info = env.step(2)
    assert done and info["wind_down"]
    assert info["exit_notional"] == 0 and info["exit_adv_multiple"] == 0


def test_strict_inventory_remains_within_one_session_exit_budget_on_repeated_entries():
    env = small_env([[100, 100]] * 6)
    env.settings["strict_volume"] = True
    env.data["adv"][:] = [1e6, 1e6]
    env.data["dollars"] = env.data["adv"].copy()
    env.reset()
    env.step(2)
    env.step(2)
    np.testing.assert_allclose(env.shares, [50, 100])
