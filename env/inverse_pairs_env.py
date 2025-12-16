import numpy as np
import gymnasium as gym
from gymnasium import spaces

from utils.logger import get_logger
from utils.rewards import reward_pnl

logger = get_logger("InversePairsEnv")


# ======================================================================
#                      FINAL MULTI-STRATEGY RL ENVIRONMENT
# ======================================================================
class InversePairsTradingEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        ohlcv,
        pairs,
        pair_types,          # NEW: {"A,B": "traditional" | "inverse"}
        config,
        price_index,
        features,
        metrics,
        feature_dims
    ):
        super().__init__()

        # ----------------------------------------------------------
        # Core Inputs
        # ----------------------------------------------------------
        self.ohlcv = ohlcv
        self.close = ohlcv["Close"]

        self.pairs = pairs
        self.pair_types = pair_types

        self.price_index = price_index
        self.config = config

        self.features = features
        self.metrics = metrics
        self.feature_dims = feature_dims

        # ----------------------------------------------------------
        # Environment parameters
        # ----------------------------------------------------------
        self.window = config["env"]["window_size"]
        self.initial_capital = config["env"]["initial_capital"]

        self.transaction_cost = config["env"]["transaction_cost"]
        self.slippage_cost = config["env"]["slippage_cost"]

        self.max_pos = config["env"]["max_position"]
        self.max_total_exposure = config["env"]["max_total_exposure"]

        self.reward_cfg = config["reward"]

        # Number of timesteps
        first_pair = pairs[0]
        self.T = len(features[first_pair])

        # ----------------------------------------------------------
        # Observation & Action Spaces
        # ----------------------------------------------------------
        # Add +1 context feature per pair for "pair_type"
        self.obs_dim = sum(self.feature_dims) + len(self.pairs)
        self.num_pairs = len(self.pairs)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Action = [direction_i, size_i] for all pairs
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_pairs * 2,),
            dtype=np.float32,
        )

        # ----------------------------------------------------------
        # State
        # ----------------------------------------------------------
        self.current_step = 0
        self.portfolio_value = self.initial_capital

        # signed exposure weights per pair
        self.positions = {pair: 0.0 for pair in pairs}

        # logs
        self.pnl_history = []
        self.reward_history = []
        self.value_history = []
        self.per_pair_pnls = {pair: [] for pair in pairs}


    # ==================================================================
    # OBSERVATION
    # ==================================================================
    def _get_obs(self):
        """
        Obs for each pair = [features..., pair_type_flag]
        traditional -> +1
        inverse     -> -1
        """
        obs = []

        for pair in self.pairs:
            feats = self.features[pair][self.current_step]
            obs.append(feats)

            # context flag: what strategy is this pair?
            if self.pair_types[pair] == "traditional":
                obs.append(np.array([1.0], dtype=np.float32))
            else:
                obs.append(np.array([-1.0], dtype=np.float32))

        return np.concatenate(obs, dtype=np.float32)


    # ==================================================================
    # STEP
    # ==================================================================
    def step(self, action):

        # Terminal condition
        if self.current_step >= self.T - 2:
            return self._get_obs(), 0.0, True, False, {
                "portfolio_value": self.portfolio_value,
                "pnl": 0.0,
                "reward": 0.0,
            }

        # --------------------------------------------------------------
        # Parse action → positions
        # --------------------------------------------------------------
        action = np.clip(action, -1, 1)

        total_exposure = 0.0
        pair_notionals = {}

        for i, pair in enumerate(self.pairs):

            direction = action[i * 2]         # -1 → +1
            size = abs(action[i * 2 + 1])     # 0 → 1
            pos = direction * size * self.max_pos

            self.positions[pair] = pos

            notional = abs(pos) * self.portfolio_value
            pair_notionals[pair] = notional
            total_exposure += abs(pos)

        # Exposure clamp
        if total_exposure > self.max_total_exposure:
            scale = self.max_total_exposure / total_exposure
            for p in self.pairs:
                self.positions[p] *= scale
                pair_notionals[p] *= scale

        # --------------------------------------------------------------
        # PnL Calculation (universal)
        # --------------------------------------------------------------
        total_pnl = 0.0

        for (A, B) in self.pairs:

            pos = self.positions[(A, B)]
            notional = pair_notionals[(A, B)]

            A_prev = self.close[A].iloc[self.current_step - 1]
            B_prev = self.close[B].iloc[self.current_step - 1]
            A_now  = self.close[A].iloc[self.current_step]
            B_now  = self.close[B].iloc[self.current_step]

            ret_A = (A_now - A_prev) / A_prev
            ret_B = (B_now - B_prev) / B_prev

            # spread-based return
            spread_ret = ret_A - ret_B
            pnl = pos * notional * spread_ret
            total_pnl += pnl

            self.per_pair_pnls[(A, B)].append(pnl)

        # --------------------------------------------------------------
        # Transaction + slippage costs
        # --------------------------------------------------------------
        tc = self.transaction_cost * abs(total_pnl)
        sc = self.slippage_cost * abs(total_pnl)
        total_pnl -= (tc + sc)

        # --------------------------------------------------------------
        # Portfolio update
        # --------------------------------------------------------------
        prev_value = self.portfolio_value
        self.portfolio_value += total_pnl

        # --------------------------------------------------------------
        # PURE PNL REWARD
        # --------------------------------------------------------------
        reward = reward_pnl(
            portfolio_value=self.portfolio_value,
            prev_portfolio_value=prev_value,
            position_size=total_exposure,
            reward_clip=self.reward_cfg["reward_clip"],
        )

        # Logging
        self.pnl_history.append(total_pnl)
        self.reward_history.append(reward)
        self.value_history.append(self.portfolio_value)

        # Next timestep
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

        self.current_step = max(1, self.window)
        self.portfolio_value = self.initial_capital

        for p in self.pairs:
            self.positions[p] = 0.0
            self.per_pair_pnls[p].clear()

        self.pnl_history.clear()
        self.reward_history.clear()
        self.value_history.clear()

        return self._get_obs(), {}


    # ==================================================================
    # FINAL PORTFOLIO ALLOCATION
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
    # RENDER
    # ==================================================================
    def render(self):
        step = self.current_step
        date = self.price_index[step]

        print(f"[Step {step}] {date} — Portfolio ${self.portfolio_value:,.2f}")
        for p in self.pairs:
            print(f"    {p}: pos={self.positions[p]:+.3f}")
