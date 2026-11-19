"""Historical equity prices, dollar volume and effective-date index membership."""

from pathlib import Path

import pandas as pd

CACHE = Path("outputs/runs/cache/rv_sp500")
FOLDS = Path("outputs/runs/residual_control_liquidity_50k/folds.csv")


def load_market(cache, dates):
    intervals = pd.read_parquet(cache / "membership_intervals.parquet")
    first, last = dates.train_start.min(), dates.test_end.max()
    rows = pd.read_parquet(
        cache / "prices.parquet",
        columns=["ticker", "date", "close", "volume", "adj_close"],
        filters=[("date", ">=", first), ("date", "<=", last)],
    )
    calendar = pd.DatetimeIndex(rows.loc[rows.ticker.eq("SPY"), "date"].sort_values())
    if calendar.has_duplicates:
        raise ValueError("Duplicate SPY trading date")
    px = rows.pivot(index="date", columns="ticker", values="adj_close").reindex(
        calendar
    )
    dv = (
        rows.assign(dollars=rows.close * rows.volume)
        .pivot(index="date", columns="ticker", values="dollars")
        .reindex(calendar)
    )
    halts = pd.read_csv(cache / "no_trade_days.csv", parse_dates=["date"])
    for row in halts.itertuples():
        if row.date in calendar and row.ticker in dv:
            dv.at[row.date, row.ticker] = 0
    # Keep the full historical membership, including labels absent from the price panel.
    active = pd.DataFrame(
        False, index=calendar, columns=sorted(set(intervals.ticker) | {"SPY"})
    )
    for row in intervals.itertuples():
        mask = calendar >= row.start
        if pd.notna(row.end):
            mask &= calendar < row.end
        active.loc[mask, row.ticker] = True
    active["SPY"] = True
    return px, dv, active
