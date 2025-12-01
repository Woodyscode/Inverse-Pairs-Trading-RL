from typing import Dict, Any
import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.agent_base import AgentBase
from utils.logger import get_logger

logger = get_logger("PPOAgent")


class PPOAgent(AgentBase):
    """
    Production-grade PPO wrapper.

    ✓ Gymnasium + SB3 compatible
    ✓ Vectorized training env ONLY
    ✓ Raw env used for inference/backtesting
    ✓ Safe unwrapping of SB3 predict()
    ✓ Device auto-selection (cpu / cuda)
    ✓ Config-driven hyperparameters
    ✓ Cleaner evaluation loop (Gymnasium API)
    """

    # ===============================================================
    # INIT
    # ===============================================================
    def __init__(self, config: Dict[str, Any], env, eval_env=None):
        super().__init__(config)

        hp = config["rl"]["hyperparameters"]
        device_cfg = config["system"].get("device", "cuda")

        # -----------------------------------------------------------
        # Device safety
        # -----------------------------------------------------------
        if device_cfg not in ["cpu", "cuda"]:
            logger.warning(f"[PPOAgent] Invalid device '{device_cfg}', using CPU")
            device_cfg = "cpu"

        logger.info("[PPOAgent] Initializing PPO agent...")

        # -----------------------------------------------------------
        # SB3 requires vectorized env **for training only**
        # -----------------------------------------------------------
        self.train_env = DummyVecEnv([lambda: env])
        self.raw_env = env

        if eval_env is not None:
            self.eval_env = DummyVecEnv([lambda: eval_env])
        else:
            self.eval_env = None

        # -----------------------------------------------------------
        # Create PPO model
        # -----------------------------------------------------------
        self.model = PPO(
            policy=hp.get("policy", "MlpPolicy"),
            env=self.train_env,
            learning_rate=hp.get("learning_rate", 3e-4),
            n_steps=hp.get("n_steps", 2048),
            batch_size=hp.get("batch_size", 64),
            n_epochs=hp.get("n_epochs", 10),
            gamma=hp.get("gamma", 0.99),
            gae_lambda=hp.get("gae_lambda", 0.95),
            clip_range=hp.get("clip_range", 0.2),
            ent_coef=hp.get("ent_coef", 0.0),
            vf_coef=hp.get("vf_coef", 0.5),
            max_grad_norm=hp.get("max_grad_norm", 0.5),
            device=device_cfg,
            verbose=1,
        )

    # ===============================================================
    # TRAINING
    # ===============================================================
    def train(self, timesteps: int, log_interval=10, save_dir="models/ppo"):
        """
        Train SB3 PPO with progress bar + vectorized env.
        """
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"[PPOAgent] Training for {timesteps:,} timesteps...")

        # SB3 learn loop
        self.model.learn(
            total_timesteps=timesteps,
            log_interval=log_interval,
            progress_bar=True,
        )

        # Save final model
        self.save(os.path.join(save_dir, "final_model"))
        logger.info("[PPOAgent] Training complete.")

    # ===============================================================
    # PREDICT (raw, non-vec env)
    # ===============================================================
    def predict(self, obs):
        """
        Returns:
            np.ndarray shape (action_dim,)
        """
        action, _ = self.model.predict(obs, deterministic=True)

        # SB3 sometimes returns shape (1, action_dim)
        if isinstance(action, np.ndarray) and action.ndim == 2:
            return action[0]

        return np.array(action, dtype=np.float32)

    # ===============================================================
    # SAVE & LOAD
    # ===============================================================
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"[PPOAgent] Saved model → {path}")

    def load(self, path):
        self.model = PPO.load(path, env=self.train_env)
        logger.info(f"[PPOAgent] Loaded model ← {path}")

    # ===============================================================
    # EVALUATE (Gymnasium API compliant)
    # ===============================================================
    def evaluate(self, env, steps=None):
        """
        Evaluates on **raw environment**, not vectorized.

        Returns dict:
            - final_value
            - rewards
            - portfolio_values
        """

        logger.info("[PPOAgent] Evaluating model...")

        obs, _ = env.reset()
        terminated = False
        truncated = False

        rewards, portfolio_vals = [], []

        while not (terminated or truncated):

            action = self.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            rewards.append(reward)
            portfolio_vals.append(info["portfolio_value"])

            if steps is not None and len(rewards) >= steps:
                break

        return {
            "final_value": portfolio_vals[-1],
            "portfolio_values": portfolio_vals,
            "rewards": rewards,
        }
