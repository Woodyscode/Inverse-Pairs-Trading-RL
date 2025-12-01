# live/live_env.py

import numpy as np
from collections import defaultdict

class LiveInversePairsEnv:
    """
    Live environment that mirrors the training environment EXACTLY,
    but without episode resets or reward tracking.
    Uses TRUE COMPOUNDING just like the backtester.
    """

    def __init__(self, pairs, window_size=30, kelly_fraction=0.5, initial_equity=100000):
        self.pairs = pairs
        self.window_size = window_size
        self.kelly_fraction = kelly_fraction

        # Rolling live history for each asset
        self.price_history = defaultdict(list)

        # Track open positions
        self.positions = {}   # (A,B): {...}

        # Portfolio tracking
        self.initial_equity = initial_equity
        self.portfolio_value = initial_equity

    # -------------------------------------------------
    # Update rolling price window
    # -------------------------------------------------
    def update_prices(self, price_dict):
        for asset, price in price_dict.items():
            self.price_history[asset].append(price)

            # maintain rolling window
            if len(self.price_history[asset]) > self.window_size:
                self.price_history[asset].pop(0)

    # -------------------------------------------------
    # Spread + Z-score
    # -------------------------------------------------
    def _get_spread_and_zscore(self, a, b):
        if len(self.price_history[a]) < self.window_size:
            return None, None

        a_hist = np.array(self.price_history[a])
        b_hist = np.array(self.price_history[b])

        spread_series = a_hist - b_hist
        spread = spread_series[-1]
        mean = spread_series.mean()
        std = spread_series.std() + 1e-6

        z = (spread - mean) / std
        return spread, z

    # -------------------------------------------------
    # Build RL observation vector
    # -------------------------------------------------
    def get_observation(self):
        obs = []

        for a, b in self.pairs:
            spread, z = self._get_spread_and_zscore(a, b)
            if spread is None:
                return None
            obs.extend([spread, z])

        return np.array(obs, dtype=np.float32)

    # -------------------------------------------------
    # Apply RL action → open/update positions
    # -------------------------------------------------
    def decode_and_apply_action(self, action, live_prices):
        """
        MultiDiscrete action per pair:
        0 = hold
        1 = long A / short B
        2 = short A / long B
        """
        for idx, (a, b) in enumerate(self.pairs):
            act = action[idx]

            if act == 0:
                continue

            direction = 1 if act == 1 else -1

            # Kelly sizing based on current equity
            capital = self.portfolio_value * self.kelly_fraction / len(self.pairs)
            if capital <= 0:
                continue

            a_price = live_prices[a]
            b_price = live_prices[b]

            size_a = capital / (2 * a_price)
            size_b = capital / (2 * b_price)

            self.positions[(a, b)] = {
                "direction": direction,
                "size_a": size_a,
                "size_b": size_b,
                "entry_prices": (a_price, b_price)
            }

    # -------------------------------------------------
    # Update portfolio based on open PnL
    # -------------------------------------------------
    def update_portfolio_value(self, live_prices):
        """
        True compounding:
            equity[t+1] = equity[t] + ΔPnL
        """
        total_pnl = 0

        for (a, b), pos in self.positions.items():
            a_price = live_prices[a]
            b_price = live_prices[b]

            entry_a, entry_b = pos["entry_prices"]
            d = pos["direction"]

            pnl = (
                d * (a_price - entry_a) * pos["size_a"] +
                -d * (b_price - entry_b) * pos["size_b"]
            )

            total_pnl += pnl

        # True compounding — SAME as backtester:
        self.portfolio_value = self.initial_equity + total_pnl

        return self.portfolio_value
