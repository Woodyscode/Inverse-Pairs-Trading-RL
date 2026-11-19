"""Summarize synchronized observed morning trades, including US early-close sessions."""

from functools import lru_cache

import numpy as np
import pandas as pd


@lru_cache(maxsize=8)
def morning_closes(path):
    bars = pd.read_parquet(path, columns=["ts", "close", "volume"])
    stamp = pd.DatetimeIndex(bars.ts).tz_convert("America/New_York").tz_localize(None)
    active = (bars.volume.to_numpy() > 0) & (bars.close.to_numpy() > 0)
    series = pd.Series(bars.close.to_numpy()[active], index=stamp[active])
    series = series.between_time("09:30", "12:59").sort_index()
    if series.index.has_duplicates:
        raise ValueError(f"Duplicate minute in {path}")
    return series


@lru_cache(maxsize=12)
def morning_moments(cache, pair):
    a, b = [morning_closes(cache / f"{sym}.parquet") for sym in pair]
    prices = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    return paired_moments(prices)


def paired_moments(prices):
    ret = np.log(prices).diff()
    # No overnight jumps, synthetic prints or multi-minute returns in realized variance.
    ret = ret.loc[ret.index.to_series().diff().eq(pd.Timedelta(minutes=1))]
    days = ret.index.normalize()
    summed = ret.groupby(days).sum()
    sumsquares = ret.pow(2).groupby(days).sum()
    cross = ret.a.mul(ret.b).groupby(days).sum()
    count = ret.a.groupby(days).count()
    return pd.DataFrame(
        {
            "a": summed.a,
            "b": summed.b,
            "aa": sumsquares.a,
            "bb": sumsquares.b,
            "ab": cross,
            "synchronized_returns": count,
        }
    )


def minute_pair_features(cache, pair, calendar, hedge):
    if any(not (cache / f"{s}.parquet").exists() for s in pair):
        return pd.DataFrame(
            np.nan, index=calendar, columns=["morning_residual", "realized_vol"]
        )
    moments = morning_moments(cache, pair).reindex(calendar)
    moments = moments.where(moments.synchronized_returns >= 200)
    scale = 1 + hedge.abs()
    return pd.DataFrame(
        {
            "morning_residual": (moments.a - hedge * moments.b) / scale,
            "realized_vol": np.sqrt(
                (moments.aa - 2 * hedge * moments.ab + hedge**2 * moments.bb).clip(
                    lower=0
                )
            )
            / scale,
        }
    )
