import numpy as np
import gymnasium as gym
from gymnasium import spaces

from utils.logger import get_logger
from utils.rewards import reward_style_simple

logger = get_logger("InversePairsEnv")


# ======================================================================
#                      FINAL ENVIRONMENT (WITH EN INPUT)
# ======================================================================
class InversePairsTradingEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        ohlcv,
        pairs,
        config,
        price_index,
        features,
        metrics,
        feature_dims
    ):
        super().__init__()

        # ----------------------------------------------------------
        # Core Data Inputs
        # ----------------------------------------------------------
        self.ohlcv = ohlcv
        self.close = ohlcv["Close"]
        self.pairs = pairs
        self.price_index = price_index
        self.config = config

        # Feature matrices per pair (already EN-reduced)
        #   features[(A,B)] = np.array(T, K_reduced)
        self.features = features
        self.metrics  = metrics
        self.feature_dims = feature_dims

        self.window = config["env"]["window_size"]
        self.initial_capital = config["env"]["initial_capital"]

        self.transaction_cost = config["env"]["transaction_cost"]
        self.slippage_cost = config["env"]["slippage_cost"]

        self.max_pos = config["env"]["max_position"]
        self.max_total_exposure = config["env"]["max_total_exposure"]

        # Length of window
        first_pair = pairs[0]
        self.T = len(features[first_pair])

        # ----------------------------------------------------------
        # Observation / Action spaces
        # ----------------------------------------------------------
        self.obs_dim = sum(self.feature_dims)   # dynamic per window
        self.num_pairs = len(pairs)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # For each pair: (direction, size)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_pairs * 2,),
            dtype=np.float32
        )

        # ----------------------------------------------------------
        # State
        # ----------------------------------------------------------
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.positions = {pair: 0.0 for pair in pairs}

        # Tracking history
        self.pnl_history = []
        self.reward_history = []
        self.value_history = []


    # ==================================================================
    # OBSERVATION
    # ==================================================================
    def _get_obs(self):

        obs = []

        for idx, pair in enumerate(self.pairs):
            feats = self.features[pair][self.current_step]
            obs.append(feats)

        return np.concatenate(obs, dtype=np.float32)


    # ==================================================================
    # STEP
    # ==================================================================
    def step(self, action):

        # Stop if no more timesteps
        if self.current_step >= self.T - 2:
            info = {
                "portfolio_value": self.portfolio_value,
                "pnl": 0.0,
                "reward": 0.0,
            }
            return self._get_obs(), 0.0, True, False, info

        # --------------------------------------------------------------
        # Action → positions
        # --------------------------------------------------------------
        action = np.clip(action, -1, 1)
        total_exposure = 0

        for i, pair in enumerate(self.pairs):
            direction = action[i * 2]     # -1, +1
            size      = abs(action[i * 2 + 1]) * self.max_pos

            pos = direction * size
            self.positions[pair] = pos
            total_exposure += abs(pos)

        # Exposure cap
        if total_exposure > self.max_total_exposure:
            scale = self.max_total_exposure / total_exposure
            for pair in self.pairs:
                self.positions[pair] *= scale

        # --------------------------------------------------------------
        # LEG-BY-LEG PNL
        # --------------------------------------------------------------
        total_pnl = 0.0

        for (A, B) in self.pairs:

            pos = self.positions[(A, B)]

            A_prev = self.close[A][self.current_step - 1]
            B_prev = self.close[B][self.current_step - 1]
            A_now  = self.close[A][self.current_step]
            B_now  = self.close[B][self.current_step]

            ret_A = (A_now - A_prev) / A_prev
            ret_B = (B_now - B_prev) / B_prev

            notional = self.portfolio_value * 0.5 * abs(pos)

            pnl = notional * (pos * (ret_A - ret_B))
            total_pnl += pnl

        # --------------------------------------------------------------
        # TRANSACTION + SLIPPAGE COSTS
        # --------------------------------------------------------------
        tc = self.transaction_cost * abs(total_pnl)
        sc = self.slippage_cost * abs(total_pnl)

        total_pnl -= (tc + sc)

        # --------------------------------------------------------------
        # Portfolio update
        # --------------------------------------------------------------
        prev_value = self.portfolio_value
        self.portfolio_value += total_pnl

        reward = reward_style_simple(
            portfolio_value=self.portfolio_value,
            prev_portfolio_value=prev_value
        )

        # Logging histories
        self.pnl_history.append(total_pnl)
        self.reward_history.append(reward)
        self.value_history.append(self.portfolio_value)

        # Move forward
        self.current_step += 1
        obs = self._get_obs()

        info = {
            "portfolio_value": self.portfolio_value,
            "pnl": total_pnl,
            "reward": reward
        }

        return obs, reward, False, False, info


    # ==================================================================
    # RESET
    # ==================================================================
    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        self.current_step = self.window
        self.portfolio_value = self.initial_capital

        for p in self.pairs:
            self.positions[p] = 0.0

        self.pnl_history.clear()
        self.reward_history.clear()
        self.value_history.clear()

        return self._get_obs(), {}


    # ==================================================================
    # FINAL ALLOCATION (for Backtester pie chart)
    # ==================================================================
    def finalize_positions(self):

        alloc = {}

        for (A, B), pos in self.positions.items():

            alloc[A] = alloc.get(A, 0) + pos / 2
            alloc[B] = alloc.get(B, 0) - pos / 2

        total = sum(abs(v) for v in alloc.values())

        if total > 0:
            for k in alloc:
                alloc[k] /= total

        return alloc


    # ==================================================================
    def render(self):
        step = self.current_step
        date = self.price_index[step]

        print(f"[Step {step}] {date} — Portfolio ${self.portfolio_value:,.2f}")
        for p in self.pairs:
            print(f"    {p}: pos={self.positions[p]:+.3f}")
