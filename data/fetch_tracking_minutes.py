"""Fetch pinned one-minute files for the existing liquid inverse-ETF discovery universe."""

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests
from curl_cffi import requests as chart_requests

from data.historical_panel import CACHE

SCREEN = Path("outputs/runs/inverse_discovery_universe/training_diagnostics.csv")
REVISION = "eff0dac55d843b3b83caec819efa213021a3e3e5"
DAILY = CACHE.parent / "inverse_tracking_daily"


def fetch_daily(symbol, directory=DAILY, period_end=1767225600):
    path = directory / f"{symbol}.json"
    if path.exists():
        return
    response = chart_requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={
            "period1": 1577836800,
            "period2": period_end,
            "interval": "1d",
            "events": "div,splits",
        },
        impersonate="chrome",
        timeout=45,
    )
    response.raise_for_status()
    chart = response.json()["chart"]["result"][0]
    if not chart.get("timestamp"):
        raise ValueError(f"No daily history for {symbol}")
    part = path.with_suffix(".part")
    part.write_text(json.dumps(chart, separators=(",", ":")))
    part.replace(path)


def fetch(symbol):
    path = CACHE / f"{symbol}.parquet"
    if path.exists():
        return
    url = (
        f"https://huggingface.co/datasets/syncster-data/historical-market-data/resolve/{REVISION}"
        f"/data/stocks/1m/{symbol[0]}/{symbol}.parquet"
    )
    with requests.get(url, stream=True, timeout=(20, 180)) as response:
        if response.status_code == 404:
            print(symbol, "absent from pinned archive", flush=True)
            return
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(dir=CACHE, delete=False) as tmp:
            part = Path(tmp.name)
        try:
            with part.open("wb") as out:
                for chunk in response.iter_content(1024 * 1024):
                    out.write(chunk)
            if pq.ParquetFile(part).metadata.num_rows < 1:
                raise ValueError(f"Empty minute file for {symbol}")
            part.replace(path)
            print(symbol, "downloaded", flush=True)
        finally:
            part.unlink(missing_ok=True)


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    DAILY.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(SCREEN).query("category == 'inverse_etf' and window == 13")
    symbols = sorted(set(candidates.a) | set(candidates.b))
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, symbols))
        list(pool.map(fetch_daily, symbols))
