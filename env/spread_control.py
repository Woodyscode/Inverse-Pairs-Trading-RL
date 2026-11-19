"""Single-pair directional control with signed log-hedge dollar weights."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SpreadControlEnv(gym.Env):
    def __init__(
        self, data, start, stop, settings, capital, alpha=False, training=False
    ):
        self.data, self.start, self.stop = data, start, stop
        self.settings, self.capital = settings, capital
        self.alpha, self.training = alpha, training
        self.portfolio_control = settings.get("portfolio_control", False)
        self.action_space = spaces.Discrete(5 if self.portfolio_control else 3)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (11 + int(alpha) + 8 * self.portfolio_control,), np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.start
        self.end = self.stop - 1
        if self.training:
            length = min(126, self.end - self.t)
            self.t = int(self.np_random.integers(self.start, self.end - length + 1))
            self.end = self.t + length
        self.equity = self.capital
        self.shares = np.zeros(2)
        self.basis = np.zeros(2)
        self.age = 0
        self.inventory = [0.0, 0.0, 0.0]
        return self._obs(), {}

    def _obs(self):
        s = self.t - 1
        values = self.data["features"][s].tolist() + self.inventory
        if self.alpha:
            values.append(self.data["alpha"][s, 0] * 100)
        if self.portfolio_control:
            cfg = self.settings
            # The target for s-2 matures at close s. Forecast errors for s-1 are unavailable.
            values.extend(
                [
                    self.data["forecast_uncertainty"][s] * 100,
                    1e4 * cfg["cost_rate"],
                    cfg["funding"] * 100,
                    cfg["borrow"] * 100,
                    float(self.data["blocked"].any()),
                    np.log1p(
                        self.data["adv"][s].min() * cfg["adv_fraction"] / self.capital
                    ),
                    min(self.end - self.t, 252) / 252,
                    self.data["regime"][s],
                ]
            )
        return np.asarray(values, dtype=np.float32)

    def step(self, action):
        t, cfg = self.t, self.settings
        direction = int(action) - 1
        px, next_px = self.data["prices"][t : t + 2]
        equity = self.equity
        h = self.data["beta"][t - 1]
        weights = np.array([1.0, -h]) / (1 + abs(h))
        gross_target = cfg["gross"] * equity
        if cfg.get("strict_volume", False):
            # Size inventory for a one-session exit at the trailing participation limit.
            active = abs(weights) > 0
            inventory_budget = cfg["adv_fraction"] * self.data["adv"][t - 1, active]
            gross_target = min(
                gross_target, np.min(inventory_budget / abs(weights[active]))
            )
        target = direction * gross_target * weights / px
        if self.portfolio_control and action in (3, 4):
            multiplier = 1.0 if action == 3 else 0.5
            value = np.sum(np.abs(self.shares * px))
            target = self.shares * min(multiplier, gross_target / max(value, 1e-12))
            direction = int(np.sign(target[0]))
        unwind = self.end - t <= cfg.get("unwind_sessions", 0)
        if unwind:
            target.fill(0.0)
        if np.any((target < 0) & self.data["blocked"]):
            target.fill(0.0)
        order = (target - self.shares) * px
        caps = cfg["adv_fraction"] * self.data["adv"][t - 1]
        if cfg.get("strict_volume", False):
            # Realized volume limits the fill after order submission. It is absent from the signal.
            caps = np.minimum(caps, cfg["adv_fraction"] * self.data["dollars"][t])
        mask = abs(order) > 0
        fill = min(1.0, np.min(caps[mask] / abs(order[mask]))) if mask.any() else 1.0
        old = self.shares.copy()
        self.shares += fill * (target - old)
        opened = old * self.shares <= 0
        increased = (abs(self.shares) > abs(old)) & ~opened
        self.basis[opened] = px[opened]
        self.basis[increased] += (
            (px[increased] - self.basis[increased])
            * abs(self.shares[increased] - old[increased])
            / abs(self.shares[increased])
        )
        legs = self.shares * px
        gross = abs(legs).sum() / equity
        current_side = int(np.sign(self.shares[0]))
        self.age = (
            (1 if current_side != np.sign(old[0]) else self.age + 1)
            if current_side
            else 0
        )
        self.inventory = [
            current_side * gross / cfg["gross"],
            100 * np.dot(self.shares, px - self.basis) / equity,
            min(self.age, 252) / 63,
        ]
        turnover = fill * abs(order).sum()
        gross_pnl = np.dot(self.shares, next_px - px)
        carry = self.data["years"][t] * (
            abs(legs).sum() * cfg["funding"]
            - np.minimum(legs, 0).sum() * cfg["borrow"]
            - max(equity - abs(legs).sum(), 0) * cfg["cash"]
        )
        self.t += 1
        done = self.t == self.end
        exit_notional = 0.0
        exit_adv_multiple = 0.0
        if done:
            closing = abs(self.shares * next_px)
            exit_notional = closing.sum()
            exit_caps = cfg["adv_fraction"] * self.data["adv"][t]
            if cfg.get("strict_volume", False):
                exit_caps = np.minimum(
                    exit_caps, cfg["adv_fraction"] * self.data["dollars"][t + 1]
                )
                if np.any(closing > exit_caps + 1e-8):
                    raise ValueError(
                        "Fold cannot be liquidated under the specified participation limit"
                    )
            exit_adv_multiple = (closing / np.maximum(exit_caps, 1e-12)).max()
            turnover += exit_notional
            self.shares.fill(0.0)
        costs = turnover * cfg["cost_rate"] + carry
        self.equity += gross_pnl - costs
        if self.equity <= 0:
            raise ValueError("Portfolio insolvent")
        risk = cfg["lambda_risk"] * (gross * self.data["vol"][t - 1]) ** 2
        turn = cfg["lambda_turn"] * turnover / equity
        net = (gross_pnl - costs) / equity
        reward = 100 * (net - risk - turn)
        info = {
            "action_id": int(action),
            "wind_down": unwind,
            "exit_notional": exit_notional,
            "exit_adv_multiple": exit_adv_multiple,
            "equity": self.equity,
            "net_return": net,
            "gross_pnl": gross_pnl,
            "costs": costs,
            "turnover": turnover / equity,
            "gross": gross,
            "risk_penalty": risk,
            "turnover_penalty": turn,
            "action": direction,
            "position": current_side,
            "reward": reward,
            "fill": fill,
            "requested_notional": abs(order).sum(),
            "filled_notional": abs(order).sum() * fill,
            "capacity": min(
                caps[abs(weights) > 0] / (cfg["gross"] * abs(weights[abs(weights) > 0]))
            ),
        }
        return self._obs(), reward, done, False, info
