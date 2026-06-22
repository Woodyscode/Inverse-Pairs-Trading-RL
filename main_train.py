import yaml
import numpy as np
import random
import torch
import psutil
import os
import json
import argparse
import pandas as pd
import math

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
#                    ENV FACTORY (UPDATED FINAL VERSION)
# ======================================================================
def build_env(
    *,
    ohlcv,
    pairs,
    pair_types,     # <--- NEW
    config,
    price_index,
    features,
    metrics,
    feature_dims
):
    """
    Unified factory for WalkForwardEngine (FULL SIGNATURE).
    Now includes pair_types to support classical + inverse logic.
    """

    logger.info(
        f"[MainTrain] Building environment: {len(pairs)} pairs, dims={sum(feature_dims)}"
    )

    return InversePairsTradingEnv(
        ohlcv=ohlcv,
        pairs=pairs,
        pair_types=pair_types,    # <--- NEW
        config=config,
        price_index=price_index,
        features=features,
        metrics=metrics,
        feature_dims=feature_dims,
    )



# ======================================================================
#                    SPY BENCHMARK LOADING HELPER
# ======================================================================
def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if hasattr(obj, "item"):
        value = obj.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def save_run_outputs(results, config):
    run_dir = config.get("output", {}).get("run_dir")
    if not run_dir:
        return

    os.makedirs(run_dir, exist_ok=True)

    results["equity_curve"].to_csv(os.path.join(run_dir, "equity_curve.csv"), header=True)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(_json_safe(results["metrics"]), f, indent=2, default=_json_default)

    with open(os.path.join(run_dir, "window_metrics.json"), "w") as f:
        json.dump(_json_safe(results.get("window_metrics", [])), f, indent=2, default=_json_default)

    with open(os.path.join(run_dir, "selected_pairs.json"), "w") as f:
        json.dump(_json_safe(results.get("selected_pairs", [])), f, indent=2, default=_json_default)

    serializable_config = {
        key: value for key, value in config.items()
        if key not in {"price_index", "benchmark_curve"}
    }
    with open(os.path.join(run_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(serializable_config, f, sort_keys=False)


def load_benchmark(price_index, config, ohlcv=None):
    if not config.get("evaluation", {}).get("compare_to_spy", True):
        curve = np.ones(len(price_index), dtype=float)
        return pd.Series(curve, index=price_index, name="Benchmark")

    logger.info("[MainTrain] Loading SPY benchmark...")
    price_type = config.get("data", {}).get("price_type", "Adj Close")

    if ohlcv is not None and price_type in ohlcv and "SPY" in ohlcv[price_type].columns:
        curve = ohlcv[price_type]["SPY"].reindex(price_index).ffill()
        curve = curve / curve.iloc[0]
        curve.name = "SPY"
        logger.info("[MainTrain] SPY benchmark loaded from prepared dataset.")
        return curve

    start = price_index[0].strftime("%Y-%m-%d")
    end   = price_index[-1].strftime("%Y-%m-%d")

    spy = DataLoader.load_single_price(
        ticker="SPY",
        start=start,
        end=end,
        price_type=price_type,
    )

    if "SPY" not in spy.columns:
        raise RuntimeError("[MainTrain] SPY fetch failed.")

    curve = spy["SPY"].reindex(price_index).ffill()
    curve = curve / curve.iloc[0]
    curve.name = "SPY"

    logger.info("[MainTrain] SPY benchmark loaded successfully.")
    return curve



# ======================================================================
#                    SYSTEM DIAGNOSTICS
# ======================================================================
def log_system_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)
    logger.info(f"[System] Process Memory Usage: {mem:.2f} MB")
    logger.info(f"[System] CPU Count: {psutil.cpu_count(logical=True)}")



# ======================================================================
#                               MAIN
# ======================================================================
def main(config_path="config/config.yaml", smoke=False):

    # ----------------------------------------------------------
    # Global reproducibility
    # ----------------------------------------------------------
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

    if smoke:
        config["smoke"] = True

    seed = config.get("system", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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
    spy_curve = load_benchmark(price_index, config, ohlcv=ohlcv)
    config["benchmark_curve"] = spy_curve

    # ----------------------------------------------------------
    # 4. Walk Forward Optimization
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
        save_run_outputs(results, config)

        return results

    else:
        logger.error("[MainTrain] ERROR: Walk-forward disabled. The current architecture REQUIRES WF.")
        raise RuntimeError("Walk-forward must be enabled for EN + feature pipeline.")



# ======================================================================
#                              RUN
# ======================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run inverse pairs training/backtest.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--smoke", action="store_true", help="Run local deterministic smoke pipeline.")
    parser.add_argument(
        "--daily-validation",
        action="store_true",
        help="Run the locked daily EOD validation benchmark.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.daily_validation:
        from daily_validation import run_daily_validation
        run_daily_validation()
    else:
        config_path = "config/smoke_config.yaml" if args.smoke else args.config
        main(config_path=config_path, smoke=args.smoke)
