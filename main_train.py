import yaml
import numpy as np
import random
import torch

from setup_paths import *

# === DATA PIPELINE ===
from data.preprocessing import prepare_data
from data.loaders import DataLoader

# === RL / ENV / BACKTEST ===
from env.inverse_pairs_env import InversePairsTradingEnv
from agents.ppo_agent import PPOAgent
from backtest.backtester import Backtester
from backtest.walkforward import WalkForwardEngine

from utils.logger import get_logger
logger = get_logger("MainTrain")


# ======================================================================
#                  ENV FACTORY (Unified API for WF + Training)
# ======================================================================
def build_env(config, ohlcv_slice, pairs, price_index):
    """
    Standard factory used by WalkForwardEngine and fixed split mode.
    """
    return InversePairsTradingEnv(
        ohlcv=ohlcv_slice,
        pairs=pairs,
        config=config,
        price_index=price_index,
    )


# ======================================================================
#                          BACKTEST WRAPPER
# ======================================================================
def backtest_runner(agent, env, config):
    tester = Backtester(agent=agent, env=env, config=config)
    return tester.run()


# ======================================================================
#                     SPY BENCHMARK LOADING HELPER
# ======================================================================
def load_benchmark(price_index):
    """
    Loads SPY as benchmark, aligned to master price_index.
    """
    logger.info("[MainTrain] Loading SPY benchmark...")

    start = price_index[0].strftime("%Y-%m-%d")
    end   = price_index[-1].strftime("%Y-%m-%d")

    spy = DataLoader.load_single_price(
        ticker="SPY",
        start=start,
        end=end,
        price_type="Adj Close",
    )

    if "SPY" not in spy.columns:
        raise RuntimeError("[MainTrain] SPY download failed: missing 'SPY' column.")

    spy_curve = spy["SPY"].reindex(price_index).ffill()
    spy_curve = spy_curve / spy_curve.iloc[0]
    spy_curve.name = "SPY"

    return spy_curve


# ======================================================================
#                     FIXED TRAIN/TEST MODE HELPERS
# ======================================================================
def fixed_train_test_split(ohlcv, returns, price_index, config):
    """
    Performs an 80/20 split when walkforward is OFF.
    Includes pair selection.
    """
    logger.info("[MainTrain] Walk-forward disabled — Using 80/20 split.")

    total_days = len(price_index)
    split = int(total_days * 0.8)

    # --- Slicing ---
    train_ohlcv = {k: df.iloc[:split] for k, df in ohlcv.items()}
    test_ohlcv  = {k: df.iloc[split:] for k, df in ohlcv.items()}

    train_ret = returns.iloc[:split]
    test_ret  = returns.iloc[split:]

    # --- Pair selection using WF engine’s selector ---
    wf_tmp = WalkForwardEngine(
        ohlcv=ohlcv,
        returns=returns,
        price_index=price_index,
        config=config,
        env_builder=build_env,
    )
    pairs = wf_tmp._select_pairs(train_ohlcv, train_ret)

    # --- Train env ---
    env_train = build_env(
        config=config,
        ohlcv_slice=train_ohlcv,
        pairs=pairs,
        price_index=price_index[:split]
    )
    agent = PPOAgent(config, env_train)
    agent.train(config["training"]["timesteps"])

    # --- Test env ---
    env_test = build_env(
        config=config,
        ohlcv_slice=test_ohlcv,
        pairs=pairs,
        price_index=price_index[split:]
    )

    return backtest_runner(agent, env_test, config)


# ======================================================================
#                               MAIN
# ======================================================================
def main(config_path="config/config.yaml"):

    # ----------------------------------------------------------
    # Global reproducibility
    # ----------------------------------------------------------
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    logger.info("[MainTrain] Loading configuration...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # =================================================================
    # (1) DATA PREPARATION
    # =================================================================
    ohlcv, returns, price_index = prepare_data(config)
    config["price_index"] = price_index

    logger.info("[MainTrain] Data loaded:")
    for key, df in ohlcv.items():
        logger.info(f"  {key}: {df.shape}")
    logger.info(f"  Returns: {returns.shape}")

    # =================================================================
    # (2) Load benchmark curve
    # =================================================================
    spy_curve = load_benchmark(price_index)
    config["benchmark_curve"] = spy_curve

    # =================================================================
    # (3) WALK-FORWARD MODE
    # =================================================================
    if config["walkforward"]["enabled"]:
        logger.info("[MainTrain] Running Walk-Forward Optimization...")

        wf = WalkForwardEngine(
            ohlcv=ohlcv,
            returns=returns,
            price_index=price_index,
            config=config,
            env_builder=build_env,   # same factory
        )
        return wf.run()

    # =================================================================
    # (4) FIXED 80/20 MODE
    # =================================================================
    return fixed_train_test_split(
        ohlcv=ohlcv,
        returns=returns,
        price_index=price_index,
        config=config,
    )


# ======================================================================
#                                RUN
# ======================================================================
if __name__ == "__main__":
    main()
