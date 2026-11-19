"""Snapshot additional pre-2014 inverse ETF histories from Yahoo's daily chart endpoint."""

import json
from pathlib import Path

import pandas as pd
import yaml
from curl_cffi import requests

COHORT = Path("config/inverse_discovery.yaml")
LOCAL = Path("outputs/daily_validation/cache/yfinance_daily_500plus_2014_2024.csv")
EXTRA = Path("outputs/runs/cache/inverse_discovery")


def main():
    EXTRA.mkdir(parents=True, exist_ok=True)
    cohort = yaml.safe_load(COHORT.read_text())
    symbols = {s for family in cohort["families"].values() for s in family}
    local = pd.read_csv(LOCAL, nrows=0)
    for symbol in sorted(symbols):
        target = EXTRA / f"{symbol}.json"
        if target.exists() or f"Adj_Close_{symbol}" in local:
            continue
        response = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={
                "period1": 1388534400,
                "period2": 1704067200,
                "interval": "1d",
                "events": "div,splits",
            },
            impersonate="chrome",
            timeout=30,
        )
        if response.status_code != 404:
            response.raise_for_status()
        chart = response.json()["chart"]
        if not chart["result"] and response.status_code != 404:
            raise ValueError(chart["error"])
        tmp = target.with_suffix(".part")
        tmp.write_text(json.dumps(chart, separators=(",", ":")))
        tmp.replace(target)
        has_bars = bool((chart["result"] or [{}])[0].get("timestamp"))
        print(symbol, "available" if has_bars else "no historical bars", flush=True)


if __name__ == "__main__":
    main()
