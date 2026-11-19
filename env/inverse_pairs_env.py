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
        pair_models=None,
        start_step=None,
    ):
        super().__init__()

        self.ohlcv = ohlcv
        self.price_type = config.get("data", {}).get("price_type", "Adj Close")
        if self.price_type not in ohlcv:
            raise KeyError(f"Configured price_type '{self.price_type}' missing from OHLCV")
        self.close = ohlcv[self.price_type]
        self.signal_lag_bars = int(config.get("env", {}).get("signal_lag_bars", 0))

        if self.signal_lag_bars < 0:
            raise ValueError("signal_lag_bars must be nonnegative")
        self.start_step = start_step
        self.betas = np.array([
            (pair_models or {}).get(pair, {}).get("hedge_ratio", 1.0) for pair in pairs
        ])
        self.leg_prices = self.close[[a for a, _ in pairs]].to_numpy(dtype=float)
        self.hedge_prices = self.close[[b for _, b in pairs]].to_numpy(dtype=float)
        for prices in (self.leg_prices, self.hedge_prices):
            if not np.isfinite(prices).all() or (prices <= 0).any():
                raise ValueError("Selected legs require a positive quote on every evaluation bar")
        self.units = np.zeros(len(pairs))
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

        if self.liquidity_cfg.get("enabled", False):
            # Orders at the close use only preceding sessions' raw dollar volume.
            self.adv = (ohlcv["Close"] * ohlcv["Volume"]).shift(1).rolling(
                int(self.liquidity_cfg.get("adv_window", 20)), min_periods=1
            ).mean()
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

    def _limit_trades(self, delta, a_prices, b_prices):
        # Each pair is a separately executable basket. Shared symbols consume
        # the same ADV budget, including when two baskets trade opposite ways.
        blocked = self.config["env"].get("unborrowable_symbols", [])
        for i, (a, b) in enumerate(self.pairs):
            desired = self.units[i] + delta[i]
            if (a in blocked and desired < 0) or (b in blocked and desired * self.betas[i] > 0):
                delta[i] = -self.units[i]

        if not self.liquidity_cfg.get("enabled", False):
            return delta
        orders = {}
        for i, (a, b) in enumerate(self.pairs):
            orders[a] = orders.get(a, 0.0) + abs(delta[i]) * a_prices[i]
            orders[b] = orders.get(b, 0.0) + abs(delta[i] * self.betas[i]) * b_prices[i]
        factors = {}
        for asset, order in orders.items():
            cap = self.adv[asset].iat[self.current_step] * self.liquidity_cfg["max_trade_pct_adv"]
            factors[asset] = min(1.0, cap / order) if order and np.isfinite(cap) else 0.0
        return delta * np.array([
            min(factors[a], factors[b]) if self.betas[i] else factors[a]
            for i, (a, b) in enumerate(self.pairs)
        ])

    def step(self, action):
        t = self.current_step
        if t >= self.T - 1:
            raise RuntimeError("Reset the finished episode before stepping")
        if self.cost_model != "turnover":
            raise ValueError(f"Unsupported cost_model: {self.cost_model}")

        prev_value = self.portfolio_value
        target, _ = self._parse_action(action)
        a, b = self.leg_prices[t], self.hedge_prices[t]
        next_a, next_b = self.leg_prices[t + 1], self.hedge_prices[t + 1]
        unit_gross = a + abs(self.betas) * b
        desired = np.array(list(target.values()), dtype=float) * prev_value / unit_gross
        delta = self._limit_trades(desired - self.units, a, b)
        self.units += delta

        leg_a = self.units * a
        leg_b = -self.units * self.betas * b
        gross = np.abs(leg_a).sum() + np.abs(leg_b).sum()
        short = -np.minimum(leg_a, 0).sum() - np.minimum(leg_b, 0).sum()
        total_exposure = gross / prev_value
        carry_days = (self.price_index[t + 1] - self.price_index[t]).days
        year_fraction = carry_days / 365.0
        borrow_cost = short * self.borrow_fee_annual * year_fraction
        financing_cost = gross * self.financing_rate_annual * year_fraction
        cash_credit = max(prev_value - gross, 0.0) * self.cash_rate_annual * year_fraction
        carry_costs = borrow_cost + financing_cost - cash_credit
        pair_pnl = self.units * ((next_a - a) - self.betas * (next_b - b))
        for pair, pnl in zip(self.pairs, pair_pnl):
            self.per_pair_pnls[pair].append(float(pnl))
        gross_pnl = pair_pnl.sum()
        turnover = np.dot(abs(delta), unit_gross)
        end_gross = self.units * (next_a + abs(self.betas) * next_b)
        terminal = t + 1 == self.T - 1
        costs_rate = self.transaction_cost + self.slippage_cost
        if prev_value + gross_pnl - turnover * costs_rate - carry_costs <= 0:
            terminal = True
        if terminal:
            # Fold boundaries settle both legs at the final mark, with exit costs.
            turnover += abs(end_gross).sum()
            self.units.fill(0)
        transaction_costs = turnover * costs_rate
        total_pnl = gross_pnl - transaction_costs - carry_costs
        self.portfolio_value = max(0.0, prev_value + total_pnl)
        total_pnl = self.portfolio_value - prev_value
        for i, pair in enumerate(self.pairs):
            self.positions[pair] = 0.0 if terminal else end_gross[i] / self.portfolio_value

        reward = reward_pnl(
            portfolio_value=self.portfolio_value,
            prev_portfolio_value=prev_value,
            position_size=total_exposure,
            reward_clip=self.reward_cfg["reward_clip"],
            penalty=self.reward_cfg.get("pos_penalty", 0.01),
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
            "gross_exposure": total_exposure,
            "active_positions": int(np.count_nonzero(end_gross)),
            "has_transition": True,
        }

        return obs, reward, terminal, False, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.units.fill(0)
        self.current_step = max(self.signal_lag_bars, self.window) if self.start_step is None else self.start_step
        if not self.signal_lag_bars <= self.current_step < self.T - 1:
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

        if self.portfolio_value > 0:
            for i, (a, b) in enumerate(self.pairs):
                weight = self.units[i] / self.portfolio_value
                alloc[a] = alloc.get(a, 0.0) + weight * self.leg_prices[self.current_step, i]
                alloc[b] = alloc.get(b, 0.0) - weight * self.betas[i] * self.hedge_prices[self.current_step, i]
        return alloc

    def render(self):
        step = self.current_step
        date = self.price_index[step]

        print(f"[Step {step}] {date} - Portfolio ${self.portfolio_value:,.2f}")
        for pair in self.pairs:
            print(f"    {pair}: pos={self.positions[pair]:+.3f}")
