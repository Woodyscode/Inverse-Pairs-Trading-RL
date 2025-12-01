# live/alpaca_stream.py

import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from utils.logger import get_logger

load_dotenv()
logger = get_logger("AlpacaStream")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY_ID")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET_KEY")

class AlpacaStream:
    """
    Uses alpaca-py official SDK (NOT alpaca-trade-api).
    Provides safe REST polling for live + historical bars.
    """

    def __init__(self):
        self.client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET)

    def get_historical_bars(self, symbols: list, limit=200):
        """
        Fetch 1-minute historical bars for a list of symbols.
        """
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Minute,
                limit=limit
            )
            bars = self.client.get_stock_bars(req).df

            if bars.empty:
                return None

            # Bars come multi-index (symbol, timestamp)
            return bars

        except Exception as e:
            logger.error(f"Failed to fetch historical bars: {e}")
            return None

    def get_latest_prices(self, symbols: list):
        """
        Returns a dict: { "SPY": 485.5, "QQQ": 397.1, ... } using the most recent bar.
        """
        bars = self.get_historical_bars(symbols, limit=1)
        if bars is None:
            return {}

        latest = {}
        for sym in symbols:
            try:
                price = bars.loc[sym].iloc[-1].close
                latest[sym] = float(price)
            except:
                continue

        return latest
