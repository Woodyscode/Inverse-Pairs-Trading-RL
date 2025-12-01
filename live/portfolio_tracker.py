import csv
import os
from datetime import datetime

class PortfolioTracker:

    def __init__(self, logfile="live_trades.csv"):
        self.logfile = logfile

        if not os.path.exists(self.logfile):
            with open(self.logfile, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "symbol", "side", "qty", "price"])

    def log_trade(self, symbol, side, qty, price):
        with open(self.logfile, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), symbol, side, qty, price])
