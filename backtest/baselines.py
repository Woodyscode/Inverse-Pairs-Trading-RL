import numpy as np


class CashPolicy:
    name = "cash"

    def __init__(self, action_dim):
        self.action_dim = action_dim

    def predict(self, obs):
        return np.zeros(self.action_dim, dtype=np.float32)


class RandomPolicy:
    name = "random"

    def __init__(self, action_dim, seed=42, max_abs_action=1.0):
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)
        self.max_abs_action = max_abs_action

    def predict(self, obs):
        return self.rng.uniform(
            -self.max_abs_action,
            self.max_abs_action,
            size=self.action_dim,
        ).astype(np.float32)


class ZScoreMeanReversionPolicy:
    name = "zscore_mean_reversion"

    def __init__(self, env, entry_z=1.0, size=1.0):
        self.env = env
        self.entry_z = entry_z
        self.size = size

    def predict(self, obs):
        action = np.zeros(self.env.action_space.shape, dtype=np.float32)

        for i, pair in enumerate(self.env.pairs):
            metrics = self.env.metrics.get(pair, [])
            step = min(self.env.current_step, len(metrics) - 1)
            zscore = metrics[step].get("zscore", 0.0) if metrics else 0.0

            if abs(zscore) < self.entry_z:
                direction = 0.0
                size = 0.0
            else:
                direction = -np.sign(zscore)
                size = self.size

            action[i * 2] = direction
            action[i * 2 + 1] = size

        return action
