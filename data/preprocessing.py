import pandas as pd
import numpy as np

from data.loaders import DataLoader
from data.returns import compute_log_returns, normalize_returns
from data.wavelet import denoise_dataframe

from utils.logger import get_logger
logger = get_logger("Preprocessing")


# ======================================================================
#                    STANDARDIZED PREPROCESSING PIPELINE
# ======================================================================
def prepare_data(config):
    """
    FINAL VERSION:
    - Loads OHLCV using DataLoader (NO preprocessing performed there)
    - Computes log returns
    - Normalizes returns
    - Wavelet-denoises returns
    - Aligns OHLCV + cleaned returns
    - Returns:
        ohlcv: dict of DataFrames
        cleaned_returns: fully denoised returns
        idx: aligned DatetimeIndex

    No pair selection.
    No feature computation.
    """

    logger.info("\n[DATA] ========= PREPARING FULL DATA PIPELINE =========")

    # ----------------------------------------------------------
    # 1) Load ALL raw OHLCV via DataLoader (clean & aligned already)
    # ----------------------------------------------------------
    ohlcv = DataLoader.load_prices(
        assets_path=config["data"]["assets_path"],
        start=config["data"]["start_date"],
        end=config["data"]["end_date"],
        min_history=config["data"]["min_history"],
    )

    close = ohlcv["Close"]

    logger.info("[DATA] OHLCV loaded. Computing returns...")

    # ----------------------------------------------------------
    # 2) Compute log returns
    # ----------------------------------------------------------
    log_ret = compute_log_returns(close)

    # ----------------------------------------------------------
    # 3) Normalize returns
    # ----------------------------------------------------------
    norm_ret = normalize_returns(log_ret)

    # ----------------------------------------------------------
    # 4) Wavelet denoise
    # ----------------------------------------------------------
    cleaned_returns = denoise_dataframe(norm_ret)

    # ----------------------------------------------------------
    # 5) Align OHLCV with cleaned returns
    # ----------------------------------------------------------
    idx = close.index.intersection(cleaned_returns.index).sort_values()

    for key in ohlcv:
        ohlcv[key] = ohlcv[key].loc[idx]

    cleaned_returns = cleaned_returns.loc[idx]

    logger.info(
        f"[DATA] Final aligned shapes: OHLCV={ohlcv['Close'].shape}, "
        f"RETURNS={cleaned_returns.shape}"
    )
    logger.info(f"[DATA] Total timesteps: {len(idx)}")
    logger.info("[DATA] ========= PREPROCESSING COMPLETE =========\n")

    return ohlcv, cleaned_returns, idx
