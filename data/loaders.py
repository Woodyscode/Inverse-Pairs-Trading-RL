import os
import yaml
import numpy as np
import pandas as pd
import yfinance as yf

from utils.logger import get_logger
logger = get_logger("DataLoader")


# ============================================================
#  HELPER: CLEAN ANY DATAFRAME (NO INPLACE)
# ============================================================
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized cleaning used by BOTH the DataLoader and walk-forward engine.
    Ensures:
    - No NaN
    - No +/- inf
    - Fully forward/back filled
    - Returns a REAL COPY (never a view)
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    return df.copy()      # ← MOST IMPORTANT (prevents view slicing issues)


# ============================================================
#  DATA LOADER CLASS
# ============================================================
class DataLoader:

    # --------------------------------------------------------
    # LOAD UNIVERSE YAML
    # --------------------------------------------------------
    @staticmethod
    def load_universe(path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"[DataLoader] Universe YAML is empty: {path}")

        tickers = []

        for category, groups in data.items():
            for group in groups:

                if isinstance(group, list):
                    tickers.extend([x.strip() for x in group])

                elif isinstance(group, str):
                    tickers.extend([x.strip() for x in group.split("-")])

                else:
                    raise TypeError(f"Invalid universe entry type: {type(group)}")

        tickers = sorted(list(set(tickers)))
        logger.info(f"[DataLoader] Loaded {len(tickers)} tickers.")
        return tickers

    # --------------------------------------------------------
    # LOAD FULL OHLCV FOR THE ENTIRE UNIVERSE
    # --------------------------------------------------------
    @staticmethod
    def load_prices(
        assets_path: str,
        start: str,
        end: str,
        min_history: int = 500,
    ):
        tickers = DataLoader.load_universe(assets_path)
        logger.info(f"[DataLoader] Downloading OHLCV for {len(tickers)} tickers...")

        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        if raw.empty:
            raise RuntimeError("No OHLCV data downloaded.")

        # Extract frames
        df_open  = clean_df(raw["Open"])
        df_high  = clean_df(raw["High"])
        df_low   = clean_df(raw["Low"])
        df_close = clean_df(raw["Close"])
        df_adj   = clean_df(raw["Adj Close"])
        df_vol   = clean_df(raw["Volume"])

        # -----------------------------
        # FILTER OUT SHORT-HISTORY ASSETS
        # -----------------------------
        def filter_min(df):
            return df.loc[:, df.count() >= min_history]

        frames = {
            "Open":      filter_min(df_open),
            "High":      filter_min(df_high),
            "Low":       filter_min(df_low),
            "Close":     filter_min(df_close),
            "Adj Close": filter_min(df_adj),
            "Volume":    filter_min(df_vol),
        }

        # Keep only common tickers
        common = set(frames["Open"].columns)
        for key in frames:
            common &= set(frames[key].columns)

        common = sorted(list(common))

        for key in frames:
            frames[key] = frames[key][common]
            frames[key] = clean_df(frames[key])   # final panel-wide cleaning

        logger.info(
            f"[DataLoader] Final aligned OHLCV shape: {frames['Close'].shape}"
        )

        return frames

    # --------------------------------------------------------
    # LOAD SINGLE TICKER (BENCHMARK)
    # --------------------------------------------------------
    @staticmethod
    def load_single_price(
        ticker: str,
        start: str,
        end: str,
        price_type="Adj Close"
    ):
        """
        Correct benchmark loader:
        - downloads full historical series
        - returns DataFrame with DatetimeIndex
        - column is the ticker symbol
        """

        try:
            df = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )
        except Exception as e:
            logger.error(f"[DataLoader] Download failed for {ticker}: {e}")
            return pd.DataFrame(index=pd.DatetimeIndex([]), columns=[ticker])

        if df.empty:
            logger.error(f"[DataLoader] No data returned for {ticker}")
            return pd.DataFrame(index=pd.DatetimeIndex([]), columns=[ticker])

        # Normalize column name
        if price_type not in df.columns:
            logger.error(
                f"[DataLoader] '{price_type}' missing for {ticker}. "
                f"Available: {df.columns}"
            )
            return pd.DataFrame(index=pd.DatetimeIndex([]), columns=[ticker])

        # Extract the series with index
        s = df[price_type].dropna().copy()

        if s.empty:
            logger.error(f"[DataLoader] '{price_type}' empty for {ticker}")
            return pd.DataFrame(index=pd.DatetimeIndex([]), columns=[ticker])

        s.index = pd.to_datetime(s.index)

        # Build proper DataFrame — this is the FIX
        out = pd.DataFrame(s)
        out.columns = [ticker]

        return clean_df(out)

