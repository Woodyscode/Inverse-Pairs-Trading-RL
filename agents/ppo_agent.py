import os
import numpy as np
from typing import Dict, Any

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

from agents.agent_base import AgentBase
from utils.logger import get_logger


logger = get_logger("PPOAgent")


# ======================================================================
#                   CUSTOM CALLBACK: LOG REWARD & VALUE
# ======================================================================
class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        if "episode" in self.locals.get("infos", [{}])[0]:
            ep_info = self.locals["infos"][0]["episode"]
            logger.info(
                f"[PPO] Episode: r={ep_info['r']:.4f}, l={ep_info['l']}"
            )
        return True


# ======================================================================
#                         PPO AGENT (FINAL)
# ======================================================================
class PPOAgent(AgentBase):

    def __init__(self, config: Dict[str, Any], env, eval_env=None):
        super().__init__(config)

        hp = config["rl"]["hyperparameters"]
        device_cfg = config["system"].get("device", "cpu")

        if device_cfg not in ["cpu", "cuda"]:
            logger.warning(f"[PPOAgent] Invalid device '{device_cfg}', using CPU")
            device_cfg = "cpu"

        logger.info("[PPOAgent] Initializing PPO Agent...")

        # ==================================================================
        # SB3 requires vectorized env for training
        # ==================================================================
        self.train_env = DummyVecEnv([lambda: Monitor(env)])
        self.raw_env = env

        if eval_env is not None:
            self.eval_env = DummyVecEnv([lambda: Monitor(eval_env)])
        else:
            self.eval_env = None

        # ==================================================================
        # PPO Model (with fully config-driven hyperparams)
        # ==================================================================
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
            ent_coef=hp.get("ent_coef", 0.00),
            vf_coef=hp.get("vf_coef", 0.50),
            max_grad_norm=hp.get("max_grad_norm", 0.5),
            device=device_cfg,
            verbose=1,
            tensorboard_log="logs/tensorboard/",
        )


    # ======================================================================
    #                              TRAINING
    # ======================================================================
    def train(self, timesteps: int, log_interval=10, save_dir="models/ppo"):
        """
        Train PPO with vectorized env + callbacks.
        """

        logger.info(f"[PPOAgent] Training for {timesteps:,} timesteps...")
        os.makedirs(save_dir, exist_ok=True)

        # --------------------------------------------------------------
        # Callbacks for monitoring, saving, and evaluation
        # --------------------------------------------------------------
        checkpoint_cb = CheckpointCallback(
            save_freq=10_000,
            save_path=os.path.join(save_dir, "checkpoints"),
            name_prefix="ppo_checkpoint",
        )

        callbacks = [checkpoint_cb, RewardLoggingCallback()]

        if self.eval_env is not None:
            eval_cb = EvalCallback(
                self.eval_env,
                best_model_save_path=os.path.join(save_dir, "best_model"),
                log_path=os.path.join(save_dir, "eval_logs"),
                eval_freq=20_000,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_cb)

        # --------------------------------------------------------------
        # SB3 Learn
        # --------------------------------------------------------------
        self.model.learn(
            total_timesteps=timesteps,
            log_interval=log_interval,
            callback=callbacks,
            progress_bar=True,
        )

        # Save final model
        self.save(os.path.join(save_dir, "final_model"))
        logger.info("[PPOAgent] Training complete.")


    # ======================================================================
    #                             ACTION SELECTION
    # ======================================================================
    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        if isinstance(action, np.ndarray) and action.ndim == 2:
            return action[0]
        return action


    # ======================================================================
    #                               SAVE / LOAD
    # ======================================================================
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"[PPOAgent] Saved model → {path}")

    def load(self, path):
        self.model = PPO.load(path, env=self.train_env)
        logger.info(f"[PPOAgent] Loaded model ← {path}")


    # ======================================================================
    #                               EVALUATE
    # ======================================================================
    def evaluate(self, env, steps=None):

        logger.info("[PPOAgent] Evaluating model (raw environment)...")

        obs, _ = env.reset()
        terminated = False
        truncated = False

        rewards = []
        values = []

        while not (terminated or truncated):

            action = self.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            rewards.append(reward)
            values.append(info["portfolio_value"])

            if steps is not None and len(rewards) >= steps:
                break

        return {
            "final_value": values[-1],
            "rewards": rewards,
            "portfolio_values": values,
        }
