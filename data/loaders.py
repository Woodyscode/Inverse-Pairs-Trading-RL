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
    Shared cleaning for all loaders.
    Ensures:
    - no NaN
    - no Inf
    - ffill + bfill
    - returns deep copy (no view issues)
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    return df.copy()


# ============================================================
#  DATA LOADER CLASS (REFINALIZED VERSION)
# ============================================================
class DataLoader:

    # --------------------------------------------------------
    # LOAD UNIVERSE YAML
    # --------------------------------------------------------
    @staticmethod
    def load_universe(path: str):
        """
        Loads all tickers from a YAML file.

        Example YAML:
        equities:
          - AAPL-MSFT-GOOG
          - [AMZN, META]
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"[DataLoader] Universe YAML empty: {path}")

        tickers = set()

        for category, groups in data.items():
            for group in groups:

                # Case 1: explicit list
                if isinstance(group, list):
                    tickers.update(x.strip() for x in group)

                # Case 2: dash-separated string
                elif isinstance(group, str):
                    tickers.update(x.strip() for x in group.split("-"))

                else:
                    raise TypeError(f"[Universe] Invalid type: {type(group)}")

        tickers = sorted(list(tickers))
        logger.info(f"[DataLoader] Universe tickers loaded: {len(tickers)}")

        return tickers


    # --------------------------------------------------------
    # LOAD OHLCV FOR ALL TICKERS (CENTRAL LOADER)
    # --------------------------------------------------------
    @staticmethod
    def load_prices(
        assets_path: str,
        start: str,
        end: str,
        min_history: int = 500,
    ):
        """
        The ONLY function responsible for loading all OHLCV data.

        Responsibilities:
        - load universe
        - download OHLCV for all tickers
        - drop tickers with insufficient history
        - align index across OHLCV panels
        - return a cleaned dict of:
            {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        """
        tickers = DataLoader.load_universe(assets_path)

        logger.info(
            f"[DataLoader] Pulling OHLCV for {len(tickers)} tickers "
            f"from {start} to {end}..."
        )

        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        if raw.empty:
            raise RuntimeError("[DataLoader] No OHLCV data returned by yfinance.")

        # Each panel (Open, Close, etc.)
        panels = {
            "Open":      clean_df(raw["Open"]),
            "High":      clean_df(raw["High"]),
            "Low":       clean_df(raw["Low"]),
            "Close":     clean_df(raw["Close"]),
            "Adj Close": clean_df(raw["Adj Close"]),
            "Volume":    clean_df(raw["Volume"]),
        }

        # --------------------------------------------------
        # FILTER by min_history
        # --------------------------------------------------
        def filter_min(df):
            """Keep only columns where count() ≥ min_history."""
            return df.loc[:, df.count() >= min_history]

        filtered = {k: filter_min(df) for k, df in panels.items()}

        # --------------------------------------------------
        # KEEP ONLY TICKERS THAT EXIST IN ALL PANELS
        # --------------------------------------------------
        common = set(filtered["Open"].columns)
        for key in filtered:
            common &= set(filtered[key].columns)

        common = sorted(list(common))

        if not common:
            raise RuntimeError("[DataLoader] No commonly-available tickers after filtering.")

        # Apply column filter
        final = {
            key: clean_df(df[common])
            for key, df in filtered.items()
        }

        logger.info(
            f"[DataLoader] Final OHLCV loaded: {final['Close'].shape}"
        )

        return final


    # --------------------------------------------------------
    # LOAD SINGLE SERIES (BENCHMARK, ETC.)
    # --------------------------------------------------------
    @staticmethod
    def load_single_price(
        ticker: str,
        start: str,
        end: str,
        price_type="Adj Close"
    ):
        """
        Loads a single ticker price series.
        Always returns a DataFrame with 1 column: ticker
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
            logger.error(f"[DataLoader] download failed for {ticker}: {e}")
            return pd.DataFrame(index=pd.DatetimeIndex([]), columns=[ticker])

        if df.empty:
            logger.error(f"[DataLoader] no data for {ticker}")
            return pd.DataFrame(index=pd.DatetimeIndex([]), columns=[ticker])

        if price_type not in df.columns:
            logger.error(
                f"[DataLoader] '{price_type}' missing for {ticker}. "
                f"Available: {df.columns}"
            )
            return pd.DataFrame(index=pd.DatetimeIndex([]), columns=[ticker])

        series = df[price_type].dropna().copy()

        if series.empty:
            logger.error(f"[DataLoader] '{price_type}' empty for {ticker}")
            return pd.DataFrame(index=pd.DatetimeIndex([]), columns=[ticker])

        series.index = pd.to_datetime(series.index)

        out = pd.DataFrame(series)
        out.columns = [ticker]

        return clean_df(out)
