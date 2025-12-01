# live/alpaca_order_executor.py

import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from utils.logger import get_logger

load_dotenv()
logger = get_logger("AlpacaOrderExecutor")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY_ID")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET_KEY")

class AlpacaOrderExecutor:
    """
    Uses alpaca-py Trading API to submit live orders.
    """

    def __init__(self, paper=True):
        self.client = TradingClient(
            ALPACA_API_KEY,
            ALPACA_SECRET,
            paper=paper
        )

    def submit_order(self, symbol, qty, side):
        """
        side: "buy" or "sell"
        """
        try:
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.GTC
            )

            result = self.client.submit_order(order)
            logger.info(f"Order submitted: {result}")
            return result

        except Exception as e:
            logger.error(f"Order failed: {e}")
            return None
