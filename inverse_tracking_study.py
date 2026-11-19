"""Test extreme daily-reset tracking residuals inside the frozen strict inverse-ETF universe."""

import argparse
import hashlib
import itertools
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from data.fetch_tracking_minutes import CACHE, DAILY, REVISION, SCREEN
from inverse_pair_discovery import (
    ADV_LIMIT,
    BORROW,
    CAPITAL,
    FEE,
    FUNDING,
    MIN_ADV,
    RHO,
    TARGET,
    holm,
    ols_slope,
    signed_weights,
)

TRAIN_START, TRAIN_END = "2020-01-07", "2023-01-05"
PERIODS = {
    "validation": ("2023-01-06", "2023-12-29"),
    "test": ("2024-01-02", "2025-12-31"),
}
HORIZONS = ("one_hour", "closing_window", "one_session", "five_sessions")
ORIGINS = ("prior_close", "opening_hour")


def chart(symbol, directory=DAILY):
    record = json.loads((directory / f"{symbol}.json").read_text())
    index = (
        pd.to_datetime(record["timestamp"], unit="s", utc=True)
        .tz_convert("America/New_York")
        .normalize()
        .tz_localize(None)
    )
    frame = pd.DataFrame(record["indicators"]["quote"][0], index=index)
    frame["adj"] = record["indicators"]["adjclose"][0]["adjclose"]
    frame["split"] = 1.0
    frame["dividend"] = 0.0
    for kind, events in record.get("events", {}).items():
        for event in events.values():
            date = (
                pd.Timestamp(event["date"], unit="s", tz="UTC")
                .tz_convert("America/New_York")
                .tz_localize(None)
                .normalize()
            )
            if date not in frame.index:
                continue
            if kind == "splits":
                frame.at[date, "split"] *= event["numerator"] / event["denominator"]
            elif kind in ("dividends", "capitalGains"):
                frame.at[date, "dividend"] += event["amount"]
    return frame


class Tape:
    def __init__(
        self, symbol, calendar, directory=DAILY, stop="2026-01-01", fit_end=TRAIN_END
    ):
        daily = chart(symbol, directory).reindex(calendar)
        minute = pd.read_parquet(
            CACHE / f"{symbol}.parquet",
            columns=["ts", "open", "close", "volume"],
            filters=[
                ("ts", ">=", pd.Timestamp("2020-01-01", tz="UTC")),
                ("ts", "<", pd.Timestamp(stop, tz="UTC")),
            ],
        )
        ts = (
            pd.DatetimeIndex(minute.ts).tz_convert("America/New_York").tz_localize(None)
        )
        if ts.has_duplicates:
            raise ValueError(f"Duplicate timestamps: {symbol}")
        day = calendar.get_indexer(ts.normalize())
        slot = ts.hour * 60 + ts.minute - 570
        keep = (
            (day >= 0)
            & (slot >= 0)
            & (slot < 390)
            & minute.volume.gt(0)
            & minute.close.gt(0)
            & minute.open.gt(0)
        )
        d, m = day[keep], slot[keep]
        scale = daily.split.fillna(1).cumprod().to_numpy()
        self.price = np.full((len(calendar), 390), np.nan)
        self.dv = np.zeros_like(self.price)
        raw = minute.close.to_numpy()[keep]
        self.dv[d, m] = raw * minute.volume.to_numpy()[keep]
        self.open = np.full(len(calendar), np.nan)
        opening = m == 0
        self.open[d[opening]] = minute.open.to_numpy()[keep][opening]
        # Chart OHLC and dividends share a split-adjusted basis. Fit the constant
        # unit conversion on training opening prints only. Minute-to-minute fills
        # use observed prices and actual effective split ratios directly.
        fit = (calendar >= TRAIN_START) & (calendar <= fit_end)
        self.unit_mode, scale, self.basis, self.fit_error = price_units(
            self.open, daily.open.to_numpy(), scale, fit
        )
        self.open *= scale
        self.price[d, m] = raw * scale[d]
        self.close = daily.close.to_numpy() * self.basis
        self.div = daily.dividend.fillna(0).to_numpy() * self.basis
        self.total = daily.adj.pct_change(fill_method=None).to_numpy()
        self.adv = (daily.close * daily.volume).rolling(20).mean().shift().to_numpy()
        self.daily_dollars = (daily.close * daily.volume).to_numpy()
        self.previous = np.r_[np.nan, self.close[:-1]]
        self.from_reset = (self.price + self.div[:, None]) / self.previous[:, None] - 1
        self.overnight = (self.open + self.div) / self.previous - 1
        self.basis_error = self.open / (daily.open.to_numpy() * self.basis) - 1


def price_units(minute_open, daily_open, splits, fit):
    # Some instruments in the archive already include split adjustments.
    # Select the representation solely by contemporaneous training-price parity.
    choices = []
    for mode, scale in (
        ("as_traded", splits),
        ("split_adjusted", np.ones(len(splits))),
    ):
        ratios = (minute_open * scale / daily_open)[fit]
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        basis = np.median(ratios) if len(ratios) else np.nan
        err = (
            np.quantile(abs(ratios / basis - 1), 0.99) if len(ratios) >= 252 else np.inf
        )
        choices.append((mode, scale, basis, err))
    return min(choices, key=lambda choice: choice[-1])


def thresholds(residual, fit):
    history = residual[fit & np.isfinite(residual)]
    if len(history) < 252:
        return np.nan, np.nan
    center = np.median(history)
    return center, np.quantile(abs(history - center), 0.95)


def effective_beta(beta, reference_return, inverse_return):
    return beta * (1 + reference_return) / (1 + inverse_return)


def calibration(calendar, screen, tapes):
    fit = (calendar >= TRAIN_START) & (calendar <= TRAIN_END)
    validation = (calendar >= PERIODS["validation"][0]) & (
        calendar <= PERIODS["validation"][1]
    )
    rows, signals = {}, {}
    for row in screen.itertuples():
        pair = (row.a, row.b)
        a, b = (tapes[s] for s in pair)
        idx = np.flatnonzero(fit)[1:]
        ra, rb = a.total[idx], b.total[idx]
        beta = ols_slope(ra, rb)
        rho = np.corrcoef(ra, rb)[0, 1]
        worst = max(
            np.corrcoef(ra[t], rb[t])[0, 1]
            for t in np.array_split(np.arange(len(idx)), 3)
        )
        val_rho = np.corrcoef(a.total[validation], b.total[validation])[0, 1]
        liquid = min(
            np.median(a.daily_dollars[idx[-60:]]), np.median(b.daily_dollars[idx[-60:]])
        )
        verified_units = max(a.fit_error, b.fit_error) <= 0.01
        tr_ok = bool(
            verified_units
            and rho <= RHO
            and worst <= RHO
            and beta < 0
            and liquid >= MIN_ADV
        )
        intercept = np.mean(ra - beta * rb)
        daily = a.total - intercept - beta * b.total
        beta_open = effective_beta(beta, b.overnight, a.overnight)
        hour = a.price[:, 59] / a.open - 1 - beta_open * (b.price[:, 59] / b.open - 1)
        for origin, resid in zip(ORIGINS, (daily, hour)):
            center, cut = thresholds(resid, fit)
            score = (
                (center - resid) / cut if cut > 0 else np.full(len(calendar), np.nan)
            )
            hedge = (
                effective_beta(beta, b.from_reset[:, 59], a.from_reset[:, 59])
                if origin == "opening_hour"
                else np.full(len(calendar), beta)
            )
            if origin == "prior_close":
                score = np.r_[np.nan, score[:-1]]
            signals[origin, pair] = (score, hedge)
            rows[origin, pair] = {
                "a": row.a,
                "b": row.b,
                "origin": origin,
                "beta": beta,
                "intercept": intercept,
                "train_rho": rho,
                "worst_train_rho": worst,
                "val_rho": val_rho,
                "train_adv": liquid,
                "center": center,
                "cutoff": cut,
                "training_eligible": tr_ok and np.isfinite(cut) and cut > 0,
                "verified_units": verified_units,
                "validation_stable": bool(val_rho <= RHO),
            }
    return rows, signals


def closing_slot(date):
    early = (date.month, date.day) in [(7, 3), (12, 24)] or (
        date.month == 11 and date.weekday() == 4 and 23 <= date.day <= 29
    )
    return 209 if early else 389


def expected_growth(reference, beta, d, m, e, n):
    # Daily reset restarts beta at each close. The entry day's partial exposure
    # is divided out before compounding the intervening daily reference returns.
    whole = np.prod(1 + beta * reference.total[d:e])
    return (
        whole
        * (1 + beta * reference.from_reset[e, n])
        / (1 + beta * reference.from_reset[d, m])
    )


def carry_years(calendar, d, m, e, n, entry_values, official_marks):
    # Integrate marked gross/short inventory from fill to fill using the last
    # close across nontrading hours. Calendar timestamps include DST transitions.
    def stamp(day, slot):
        return (calendar[day] + pd.Timedelta(minutes=571 + slot)).tz_localize(
            "America/New_York"
        )

    periods = [
        stamp(d, m),
        *[stamp(k, closing_slot(calendar[k])) for k in range(d, e)],
        stamp(e, n),
    ]
    years = np.array(
        [
            (right - left).total_seconds() / (365 * 86400)
            for left, right in itertools.pairwise(periods)
        ]
    )
    values = np.vstack([entry_values, official_marks])
    return years @ np.abs(values).sum(axis=1) * FUNDING, years @ np.maximum(
        -values, 0
    ).sum(axis=1) * BORROW


def enter(tapes, pair, d, slot, hedge):
    return enter_weighted(tapes, pair, d, slot, signed_weights(hedge))


def enter_weighted(tapes, pair, d, slot, w):
    w = np.asarray(w, dtype=float)
    if w.shape != (2,) or not np.isclose(abs(w).sum(), 1.0):
        raise ValueError("Pair weights require unit gross")
    active = w != 0
    a, b = (tapes[s] for s in pair)
    for m in range(slot, slot + 5):
        px = np.array([a.price[d, m], b.price[d, m]])
        if not np.isfinite(px).all():
            continue
        budgets = ADV_LIMIT * np.minimum([a.dv[d, m], b.dv[d, m]], [a.adv[d], b.adv[d]])
        size = min(CAPITAL * TARGET, np.min(budgets[active] / abs(w[active])))
        if size > 0:
            return m, size, w, px


def exit_trade(tapes, pair, calendar, d, entry, side, beta, e, slot, end, fills=None):
    m, size, w, px = entry
    a, b = (tapes[s] for s in pair)
    left = 1.0
    result = {
        "gross": 0.0,
        "execution": size * FEE,
        "funding": 0.0,
        "borrow": 0.0,
        "reset_excess": 0.0,
        "turnover": size,
    }
    for day in range(e, end):
        first = slot if day == e else 0
        for n in range(first, closing_slot(calendar[day]) + 1):
            marks = np.array([a.price[day, n], b.price[day, n]])
            if not np.isfinite(marks).all():
                continue
            exit_values = size * abs(w) * marks / px
            fraction = min(
                left,
                ADV_LIMIT
                * np.min(
                    np.minimum([a.dv[day, n], b.dv[day, n]], [a.adv[day], b.adv[day]])[
                        w != 0
                    ]
                    / exit_values[w != 0]
                ),
            )
            if fraction <= 0:
                continue
            if fills is not None:
                fills.append((day, n, fraction, marks.copy()))
            distributions = np.array([x.div[d + 1 : day + 1].sum() for x in (a, b)])
            gains = (marks + distributions) / px - 1
            f, borr = carry_years(
                calendar,
                d,
                m,
                day,
                n,
                side * size * w,
                np.column_stack([a.close[d:day], b.close[d:day]])
                * (side * size * w / px),
            )
            if not np.isfinite([*gains, f, borr]).all():
                raise ValueError(f"Missing held mark for {pair} on {calendar[day]}")
            result["gross"] += fraction * side * size * (w @ gains)
            result["turnover"] += fraction * exit_values.sum()
            result["execution"] += fraction * FEE * exit_values.sum()
            result["funding"] += fraction * f
            result["borrow"] += fraction * borr
            if beta is None:
                result["reset_excess"] = np.nan
            else:
                expected = expected_growth(b, beta, d, m, day, n)
                result["reset_excess"] += (
                    fraction * side * size * w[0] * (1 + gains[0] - expected)
                )
            left -= fraction
            if left < 1e-10:
                result["net"] = (
                    result["gross"]
                    - result["execution"]
                    - result["funding"]
                    - result["borrow"]
                )
                result.update(
                    exit_date=calendar[day],
                    exit_minute=n,
                    exit_delay_sessions=day - e,
                    exit_delay_minutes=n - first if day == e else np.nan,
                )
                return result, day
    raise ValueError(
        f"Unresolved exit for {pair}, entry {calendar[d]}, target {calendar[e]}"
    )


def replay(calendar, pairs, tapes, models, signals, origin, split, book):
    first, last = PERIODS[split]
    start = calendar.searchsorted(first)
    end = calendar.searchsorted(last, side="right")
    rows = []
    eligible = [
        p
        for p in pairs
        if models[origin, p]["training_eligible"]
        and (split == "validation" or models[origin, p]["validation_stable"])
    ]
    slot = 1 if origin == "prior_close" else 61
    next_day = start
    attempts = missed = 0
    for d in range(start, end - 6):
        if d < next_day:
            continue
        extremes = []
        for pair in eligible:
            score, hedge = signals[origin, pair]
            if abs(score[d]) >= 1 and np.isfinite(hedge[d]) and hedge[d] < 0:
                extremes.append((-abs(score[d]), pair, score[d], hedge[d]))
        if not extremes:
            continue
        _, pair, score, hedge = min(extremes)
        attempts += 1
        entry = enter(tapes, pair, d, slot, hedge)
        if entry is None:
            missed += 1
            continue
        m, size, w, _ = entry
        # Identical entry and a common flat interval across every exit variant.
        # No filter consults an exit print or a future return before committing.
        next_day = d + 6
        for name, (e, n) in zip(
            HORIZONS,
            [(d, m + 60), (d, closing_slot(calendar[d]) - 9), (d + 1, m), (d + 5, m)],
        ):
            result, flat = exit_trade(
                tapes,
                pair,
                calendar,
                d,
                entry,
                np.sign(score),
                models[origin, pair]["beta"],
                e,
                n,
                end,
            )
            next_day = max(next_day, flat + 1)
            rows.append(
                dict(
                    book=book,
                    split=split,
                    origin=origin,
                    horizon=name,
                    a=pair[0],
                    b=pair[1],
                    date=calendar[d],
                    entry_minute=m,
                    side=int(np.sign(score)),
                    normalized_signal=score,
                    beta_at_signal=hedge,
                    gross_entry=size,
                    entry_a=size * w[0] * np.sign(score),
                    entry_b=size * w[1] * np.sign(score),
                    **result,
                )
            )
    return rows, {
        "book": book,
        "split": split,
        "origin": origin,
        "eligible_pairs": len(eligible),
        "attempts": attempts,
        "unfilled": missed,
    }


def summarize(ledger, calendar):
    rows = []
    for (book, split, origin, horizon), g in ledger.groupby(
        ["book", "split", "origin", "horizon"]
    ):
        gross = g.gross_entry.sum()
        stat = {
            "book": book,
            "split": split,
            "origin": origin,
            "horizon": horizon,
            "trades": len(g),
            "gross_bp": g.gross.sum() / gross * 10000,
            "net_bp": g.net.sum() / gross * 10000,
            "reset_excess_bp": g.reset_excess.sum() / gross * 10000,
            "convergence_rate": g.reset_excess.gt(0).mean(),
            "account_return": g.net.sum() / CAPITAL,
            "avg_entry": g.gross_entry.mean(),
            "breakeven_bp": (g.gross - g.funding - g.borrow).sum()
            / g.turnover.sum()
            * 10000,
        }
        if book == "primary":
            lo, hi = PERIODS[split]
            dates = calendar[(calendar >= lo) & (calendar <= hi)]
            daily = (
                g.groupby("date")[["net", "gross_entry"]]
                .sum()
                .reindex(dates, fill_value=0)
                .to_numpy()
            )
            rng = np.random.default_rng(42)
            starts = rng.integers(0, len(dates), (3000, (len(dates) + 19) // 20))
            ix = ((starts[:, :, None] + np.arange(20)) % len(dates)).reshape(3000, -1)[
                :, : len(dates)
            ]
            boot = daily[ix].sum(axis=1)
            bp = np.divide(
                boot[:, 0] * 10000,
                boot[:, 1],
                out=np.full(len(boot), np.nan),
                where=boot[:, 1] > 0,
            )
            stat["ci_low_bp"], stat["ci_high_bp"] = np.nanquantile(bp, [0.025, 0.975])
            stat["p_positive"] = (
                1 + np.count_nonzero(boot[:, 0] - g.net.sum() >= g.net.sum())
            ) / (len(boot) + 1)
        rows.append(stat)
    out = pd.DataFrame(rows)
    for split in PERIODS:
        mask = out.book.eq("primary") & out.split.eq(split)
        out.loc[mask, "holm_p"] = holm(out.loc[mask, "p_positive"])
    return out


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    screen = pd.read_csv(SCREEN).query("category == 'inverse_etf' and window == 13")
    calendar = chart("SPY").index
    symbols = sorted(set(screen.a) | set(screen.b))
    available = [s for s in symbols if (CACHE / f"{s}.parquet").exists()]
    tapes = {s: Tape(s, calendar) for s in available}
    quality = []
    for s in symbols:
        if s not in tapes:
            quality.append({"symbol": s, "available": False})
            continue
        t = tapes[s]
        error = np.abs(t.basis_error)
        quality.append(
            {
                "symbol": s,
                "available": True,
                "minute_bars": np.isfinite(t.price).sum(),
                "opening_prints": np.isfinite(error).sum(),
                "unit_basis": t.basis,
                "price_representation": t.unit_mode,
                "train_p99_error": t.fit_error,
                "p99_open_basis_error": np.nanquantile(error, 0.99),
                "days_over_5pct": int(np.sum(error > 0.05)),
            }
        )
    pd.DataFrame(quality).to_csv(output / "data_quality.csv", index=False)
    screen = screen[screen.a.isin(available) & screen.b.isin(available)]
    models, signals = calibration(calendar, screen, tapes)
    pairs = list(zip(screen.a, screen.b))
    pd.DataFrame(models.values()).to_csv(output / "calibration.csv", index=False)
    records, coverage = [], []
    for split in PERIODS:
        for origin in ORIGINS:
            for book, group in [
                ("primary", pairs),
                *(("/".join(p), [p]) for p in pairs),
            ]:
                trades, count = replay(
                    calendar, group, tapes, models, signals, origin, split, book
                )
                records.extend(trades)
                coverage.append(count)
    df = pd.DataFrame(records)
    df.to_csv(output / "trades.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    stats = summarize(df, calendar)
    stats.to_csv(output / "results.csv", index=False)
    inputs = [
        Path(__file__),
        Path("data/fetch_tracking_minutes.py"),
        SCREEN,
        Path("inverse_pair_discovery.py"),
        *(CACHE / f"{s}.parquet" for s in available),
        *(DAILY / f"{s}.json" for s in symbols),
    ]
    checksums = {}
    for p in inputs:
        with p.open("rb") as f:
            checksums[str(p)] = hashlib.file_digest(f, "sha256").hexdigest()
    manifest = {
        "revision": REVISION,
        "inputs": checksums,
        "train": [TRAIN_START, TRAIN_END],
        "periods": PERIODS,
        "packages": {
            k: version(k)
            for k in ("numpy", "pandas", "pyarrow", "statsmodels", "scipy")
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(stats.query("book == 'primary' and split == 'test'").to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
