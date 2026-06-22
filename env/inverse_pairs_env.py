import numpy as np
import gymnasium as gym
from gymnasium import spaces

from utils.logger import get_logger
from utils.rewards import reward_pnl

logger = get_logger("InversePairsEnv")


class InversePairsTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        ohlcv,
        pairs,
        pair_types,
        config,
        price_index,
        features,
        metrics,
        feature_dims,
    ):
        super().__init__()

        self.ohlcv = ohlcv
        self.price_type = config.get("data", {}).get("price_type", "Adj Close")
        if self.price_type not in ohlcv:
            raise KeyError(f"Configured price_type '{self.price_type}' missing from OHLCV")
        self.close = ohlcv[self.price_type]
        self.signal_lag_bars = int(config.get("env", {}).get("signal_lag_bars", 0))

        self.pairs = pairs
        self.pair_types = pair_types

        self.price_index = price_index
        self.config = config

        self.features = features
        self.metrics = metrics
        self.feature_dims = feature_dims

        self.window = config["env"]["window_size"]
        self.initial_capital = config["env"]["initial_capital"]

        self.transaction_cost = config["env"]["transaction_cost"]
        self.slippage_cost = config["env"]["slippage_cost"]
        self.cost_model = config["env"].get("cost_model", "turnover")
        self.borrow_fee_annual = config["env"].get("borrow_fee_annual", 0.0)
        self.financing_rate_annual = config["env"].get("financing_rate_annual", 0.0)
        self.cash_rate_annual = config["env"].get("cash_rate_annual", 0.0)
        self.liquidity_cfg = config["env"].get("liquidity", {"enabled": False})

        self.max_pos = config["env"]["max_position"]
        self.max_total_exposure = config["env"]["max_total_exposure"]

        self.reward_cfg = config["reward"]

        first_pair = pairs[0]
        self.T = len(features[first_pair])

        self.obs_dim = sum(self.feature_dims) + len(self.pairs)
        self.num_pairs = len(self.pairs)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_pairs * 2,),
            dtype=np.float32,
        )

        seed = config.get("system", {}).get("seed", None)
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)

        self.current_step = 0
        self.portfolio_value = self.initial_capital

        self.positions = {pair: 0.0 for pair in pairs}

        self.pnl_history = []
        self.reward_history = []
        self.value_history = []
        self.per_pair_pnls = {pair: [] for pair in pairs}

    def _get_obs(self):
        """
        Observation at step t contains only features computed with data
        available through t plus a static pair-type flag.
        """
        obs = []
        feature_step = max(0, self.current_step - self.signal_lag_bars)

        for pair in self.pairs:
            feats = self.features[pair][feature_step]
            obs.append(feats)

            if self.pair_types[pair] == "traditional":
                obs.append(np.array([1.0], dtype=np.float32))
            else:
                obs.append(np.array([-1.0], dtype=np.float32))

        return np.concatenate(obs, dtype=np.float32)

    def _parse_action(self, action):
        action = np.clip(action, -1, 1)

        total_exposure = 0.0
        target_positions = {}

        for i, pair in enumerate(self.pairs):
            direction = action[i * 2]
            size = abs(action[i * 2 + 1])
            pos = direction * size * self.max_pos

            target_positions[pair] = pos
            total_exposure += abs(pos)

        if total_exposure > self.max_total_exposure:
            scale = self.max_total_exposure / total_exposure
            for pair in target_positions:
                target_positions[pair] *= scale
            total_exposure = self.max_total_exposure

        return target_positions, total_exposure

    def _dollar_volume(self, asset, step):
        volume = self.ohlcv.get("Volume")
        if volume is None or asset not in volume:
            return np.inf

        window = int(self.liquidity_cfg.get("adv_window", 20))
        start = max(0, step - window + 1)
        dollar_volume = volume[asset].iloc[start:step + 1] * self.close[asset].iloc[start:step + 1]
        value = float(dollar_volume.mean())
        return value if np.isfinite(value) and value > 0 else np.inf

    def _apply_liquidity_limits(self, target_positions):
        if not self.liquidity_cfg.get("enabled", False):
            return target_positions

        max_trade_pct_adv = float(self.liquidity_cfg.get("max_trade_pct_adv", 0.01))
        limited = target_positions.copy()

        for (A, B) in self.pairs:
            current = self.positions[(A, B)]
            target = limited[(A, B)]
            delta = target - current
            if delta == 0:
                continue

            pair_trade_notional = abs(delta) * self.portfolio_value
            leg_trade_notional = pair_trade_notional / 2.0
            cap = min(
                self._dollar_volume(A, self.current_step),
                self._dollar_volume(B, self.current_step),
            ) * max_trade_pct_adv

            if leg_trade_notional > cap and leg_trade_notional > 0:
                scale = cap / leg_trade_notional
                limited[(A, B)] = current + delta * scale

        return limited

    def _calculate_turnover(self, target_positions):
        turnover = 0.0
        for pair in self.pairs:
            turnover += abs(target_positions[pair] - self.positions[pair]) * self.portfolio_value
        return turnover

    def _calculate_carry_costs(self, target_positions):
        short_notional = 0.0
        gross_notional = 0.0

        for pos in target_positions.values():
            pair_notional = abs(pos) * self.portfolio_value
            gross_notional += pair_notional
            short_notional += pair_notional / 2.0

        borrow_cost = short_notional * self.borrow_fee_annual / 252.0
        financing_cost = gross_notional * self.financing_rate_annual / 252.0
        net_cash = max(
            self.portfolio_value * (1.0 - sum(abs(v) for v in target_positions.values())),
            0.0,
        )
        cash_credit = net_cash * self.cash_rate_annual / 252.0
        carry_costs = borrow_cost + financing_cost - cash_credit

        return carry_costs, borrow_cost, financing_cost, cash_credit

    def step(self, action):
        if self.current_step >= self.T - 1:
            return self._get_obs(), 0.0, True, False, {
                "portfolio_value": self.portfolio_value,
                "pnl": 0.0,
                "reward": 0.0,
                "gross_pnl": 0.0,
                "transaction_costs": 0.0,
                "carry_costs": 0.0,
                "borrow_cost": 0.0,
                "financing_cost": 0.0,
                "cash_credit": 0.0,
                "turnover": 0.0,
                "has_transition": False,
            }

        target_positions, total_exposure = self._parse_action(action)
        target_positions = self._apply_liquidity_limits(target_positions)
        total_exposure = sum(abs(pos) for pos in target_positions.values())

        if self.cost_model != "turnover":
            raise ValueError(f"Unsupported cost_model: {self.cost_model}")

        turnover = self._calculate_turnover(target_positions)
        transaction_costs = (self.transaction_cost + self.slippage_cost) * turnover
        carry_costs, borrow_cost, financing_cost, cash_credit = self._calculate_carry_costs(target_positions)

        gross_pnl = 0.0
        for (A, B) in self.pairs:
            pos = target_positions[(A, B)]

            A_entry = self.close[A].iloc[self.current_step]
            B_entry = self.close[B].iloc[self.current_step]
            A_exit = self.close[A].iloc[self.current_step + 1]
            B_exit = self.close[B].iloc[self.current_step + 1]

            ret_A = (A_exit - A_entry) / A_entry
            ret_B = (B_exit - B_entry) / B_entry
            spread_ret = ret_A - ret_B

            pnl = pos * self.portfolio_value * spread_ret
            gross_pnl += pnl
            self.per_pair_pnls[(A, B)].append(pnl)

        total_pnl = gross_pnl - transaction_costs - carry_costs

        prev_value = self.portfolio_value
        self.portfolio_value += total_pnl
        self.positions.update(target_positions)

        reward = reward_pnl(
            portfolio_value=self.portfolio_value,
            prev_portfolio_value=prev_value,
            position_size=total_exposure,
            reward_clip=self.reward_cfg["reward_clip"],
        )

        self.pnl_history.append(total_pnl)
        self.reward_history.append(reward)
        self.value_history.append(self.portfolio_value)

        self.current_step += 1
        obs = self._get_obs()

        info = {
            "portfolio_value": self.portfolio_value,
            "pnl": total_pnl,
            "reward": reward,
            "gross_pnl": gross_pnl,
            "transaction_costs": transaction_costs,
            "carry_costs": carry_costs,
            "borrow_cost": borrow_cost,
            "financing_cost": financing_cost,
            "cash_credit": cash_credit,
            "turnover": turnover,
            "has_transition": True,
        }

        return obs, reward, False, False, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = max(1, self.window)
        if self.current_step >= self.T:
            raise ValueError(
                f"window_size={self.window} leaves no tradable steps for T={self.T}"
            )

        self.portfolio_value = self.initial_capital

        for pair in self.pairs:
            self.positions[pair] = 0.0
            self.per_pair_pnls[pair].clear()

        self.pnl_history.clear()
        self.reward_history.clear()
        self.value_history.clear()

        return self._get_obs(), {}

    def finalize_positions(self):
        alloc = {}

        for (A, B), pos in self.positions.items():
            alloc[A] = alloc.get(A, 0) + pos / 2
            alloc[B] = alloc.get(B, 0) - pos / 2

        total = sum(abs(v) for v in alloc.values())

        if total > 0:
            for key in alloc:
                alloc[key] /= total

        return alloc

    def render(self):
        step = self.current_step
        date = self.price_index[step]

        print(f"[Step {step}] {date} - Portfolio ${self.portfolio_value:,.2f}")
        for pair in self.pairs:
            print(f"    {pair}: pos={self.positions[pair]:+.3f}")
