import pandas as pd
import numpy as np

from data.loaders import DataLoader
from data.returns import compute_log_returns, normalize_returns
from data.wavelet import denoise_dataframe

from utils.logger import get_logger
logger = get_logger("Preprocessing")


# ======================================================================
#                    FINAL PREPROCESSING PIPELINE
# ======================================================================
def prepare_data(config):
    """
    Prepares raw OHLCV data, log returns, wavelet-denoised returns,
    and aligned price index.

    IMPORTANT:
    - This function NO LONGER handles pairs.
    - Pair selection is done in WalkForwardEngine.
    - Per-pair advanced features are computed INSIDE the environment.
    """

    logger.info("\n[DATA] ========= PREPARING FULL DATA PIPELINE =========")

    # -------------------------------
    # Read config values
    # -------------------------------
    assets_path = config["data"]["assets_path"]
    start       = config["data"]["start_date"]
    end         = config["data"]["end_date"]
    min_hist    = config["data"]["min_history"]


    # ----------------------------------------------------------
    # 1. Load universe tickers
    # ----------------------------------------------------------
    universe = DataLoader.load_universe(assets_path)
    logger.info(f"[DATA] Universe contains {len(universe)} tickers.")


    # ----------------------------------------------------------
    # 2. Load OHLCV data for ALL assets
    # ----------------------------------------------------------
    ohlcv = DataLoader.load_prices(
        assets_path=assets_path,
        start=start,
        end=end,
        min_history=min_hist,
    )

    # ----------------------------------------------------------
    # 3. Filter OHLCV to tickers actually present
    # ----------------------------------------------------------
    for key, df in ohlcv.items():
        ohlcv[key] = df[[t for t in universe if t in df.columns]]

    if ohlcv["Close"].empty:
        raise RuntimeError("[DATA] OHLCV is empty after filtering universe.")

    logger.info("[DATA] Shapes after universe filtering:")
    for k, df in ohlcv.items():
        logger.info(f"   {k:10s}: {df.shape}")


    # ----------------------------------------------------------
    # 4. Compute log returns → normalize → wavelet denoise
    # ----------------------------------------------------------
    close = ohlcv["Close"]

    log_ret  = compute_log_returns(close)
    norm_ret = normalize_returns(log_ret)

    # wavelet smoothing for noise reduction
    cleaned_returns = denoise_dataframe(norm_ret)


    # ----------------------------------------------------------
    # 5. Align OHLCV with cleaned returns
    # ----------------------------------------------------------
    idx = close.index.intersection(cleaned_returns.index).sort_values()

    for key in ohlcv:
        ohlcv[key] = ohlcv[key].loc[idx]

    cleaned_returns = cleaned_returns.loc[idx]

    logger.info(
        f"[DATA] Aligned shapes: OHLCV={ohlcv['Close'].shape}, RETURNS={cleaned_returns.shape}"
    )
    logger.info(f"[DATA] Total timesteps: {len(idx)}")


    # ----------------------------------------------------------
    # 6. RETURN CORE DATA ONLY
    # ----------------------------------------------------------
    # No pairs, no feature vectors computed here.
    # The environment will compute pairs inside windows.
    # ----------------------------------------------------------
    logger.info("[DATA] ========= PREPROCESSING COMPLETE =========\n")

    return ohlcv, cleaned_returns, idx
