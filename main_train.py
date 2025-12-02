import yaml
import numpy as np
import random
import torch
import psutil
import os

from setup_paths import *

# === DATA PIPELINE ===
from data.preprocessing import prepare_data
from data.loaders import DataLoader

# === RL / ENV / BACKTEST ===
from env.inverse_pairs_env import InversePairsTradingEnv
from backtest.walkforward import WalkForwardEngine

from utils.logger import get_logger
logger = get_logger("MainTrain")


# ======================================================================
#                    ENV FACTORY (FULL SIGNATURE)
# ======================================================================
def build_env(config, ohlcv_slice, pairs, price_index, features, metrics, feature_dims):
    """
    Unified factory for WF environment construction.
    WalkForwardEngine guarantees the correct inputs.
    """
    logger.info(
        f"[MainTrain] Building environment: {len(pairs)} pairs, dims={sum(feature_dims)}"
    )

    return InversePairsTradingEnv(
        ohlcv=ohlcv_slice,
        pairs=pairs,
        config=config,
        price_index=price_index,
        features=features,
        metrics=metrics,
        feature_dims=feature_dims,
    )


# ======================================================================
#                    SPY BENCHMARK LOADING HELPER
# ======================================================================
def load_benchmark(price_index):
    """
    Loads SPY aligned to the master price index.
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
        raise RuntimeError("[MainTrain] SPY fetch failed.")

    curve = spy["SPY"].reindex(price_index).ffill()
    curve = curve / curve.iloc[0]
    curve.name = "SPY"

    logger.info("[MainTrain] SPY benchmark loaded successfully.")
    return curve


# ======================================================================
#                    RAM / CPU Diagnostics
# ======================================================================
def log_system_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)
    logger.info(f"[System] Process Memory Usage: {mem:.2f} MB")
    logger.info(f"[System] CPU Count: {psutil.cpu_count(logical=True)}")


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

    logger.info("====================================================")
    logger.info("================= MAIN TRAIN START =================")
    logger.info("====================================================")
    
    log_system_usage()

    # ----------------------------------------------------------
    # 1. Load configuration
    # ----------------------------------------------------------
    logger.info("[MainTrain] Loading configuration file...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # ----------------------------------------------------------
    # 2. Data Preparation
    # ----------------------------------------------------------
    logger.info("[MainTrain] Preparing full dataset...")

    ohlcv, returns, price_index = prepare_data(config)
    config["price_index"] = price_index

    logger.info("\n[MainTrain] ===== DATA SUMMARY =====")
    for key, df in ohlcv.items():
        logger.info(f"  {key:10s}: shape={df.shape}, start={df.index[0]}, end={df.index[-1]}")
    logger.info(f"  RETURNS: {returns.shape}")
    logger.info("====================================\n")

    # ----------------------------------------------------------
    # 3. Benchmark
    # ----------------------------------------------------------
    logger.info("[MainTrain] Preparing benchmark curve...")
    spy_curve = load_benchmark(price_index)
    config["benchmark_curve"] = spy_curve

    # ----------------------------------------------------------
    # 4. Walk Forward Optimization (EN + Feature Computation)
    # ----------------------------------------------------------
    if config["walkforward"]["enabled"]:
        logger.info("[MainTrain] Walk-Forward Optimization ENABLED.")
        logger.info("[MainTrain] Creating WalkForwardEngine...")

        wf = WalkForwardEngine(
            ohlcv=ohlcv,
            returns=returns,
            price_index=price_index,
            config=config,
            env_builder=build_env,   # Full signature factory
        )

        logger.info("[MainTrain] Running WF engine...")
        results = wf.run()

        logger.info("[MainTrain] Walk-Forward run complete.")
        logger.info(f"[MainTrain] Final stitched equity value: {results['equity_curve'].iloc[-1]:,.2f}")

        return results

    else:
        logger.error("[MainTrain] ERROR: Walk-forward disabled. The current architecture REQUIRES WF.")
        raise RuntimeError("Walk-forward must be enabled for EN + feature pipeline.")


# ======================================================================
#                              RUN
# ======================================================================
if __name__ == "__main__":
    main()
