from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import torch

from utils.logger import get_logger

logger = get_logger("AgentBase")


class AgentBase(ABC):
    """
    Abstract base class for all trading RL agents.

    Provides:
    - Unified constructor
    - Device handling
    - Logging setup
    - Seed helper
    - Enforced method signatures

    Subclasses must implement:
        train()
        predict()
        save()
        load()
        evaluate()
    """

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------
    def __init__(self, config: Dict[str, Any]):

        self.config = config

        # --------------------------------------------------------------
        # DEVICE HANDLING
        # --------------------------------------------------------------
        device = config.get("system", {}).get("device", "cpu")

        if isinstance(device, str):
            if device not in ["cpu", "cuda"]:
                logger.warning(f"[AgentBase] Invalid device '{device}', defaulting to cpu.")
                device = "cpu"
        else:
            logger.warning(f"[AgentBase] Device must be str. Using cpu.")
            device = "cpu"

        self.device = device
        self.torch_device = torch.device(device)

        # --------------------------------------------------------------
        # SEED
        # --------------------------------------------------------------
        seed = config.get("system", {}).get("seed", None)
        if seed is not None:
            self.set_seed(seed)
            logger.info(f"[AgentBase] Seed set to {seed}")

        logger.info(f"{self.__class__.__name__} initialized (device={self.device}).")

    # ------------------------------------------------------------------
    # SEEDING
    # ------------------------------------------------------------------
    def set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ==================================================================
    # ABSTRACT METHODS
    # ==================================================================

    @abstractmethod
    def train(self, *args, **kwargs):
        """
        Train the agent on a given environment.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, observation):
        """
        Produce an action given an observation from the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str):
        """
        Save the model to disk.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str):
        """
        Load the model from disk.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, env, steps: Optional[int] = None):
        """
        Evaluate the agent's performance in a given environment.
        Return a dict of evaluation metrics.
        """
        raise NotImplementedError


