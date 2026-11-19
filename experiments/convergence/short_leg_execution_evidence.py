"""Audit archived NBBO and stock-loan evidence against fixed simulated fills."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def evidence_table(path, times, numeric):
    fields = ["symbol", *times, *numeric]
    data = pd.read_csv(path, usecols=fields) if path else pd.DataFrame(columns=fields)
    for field in times:
        if not data[field].astype(str).str.contains(r"(?:Z|[+-]\d\d:\d\d)$", regex=True).all():
            raise ValueError(f"Missing explicit timezone in {field}")
        data[field] = pd.to_datetime(data[field], format="mixed", utc=True)
    for field in numeric:
        data[field] = pd.to_numeric(data[field], errors="raise")
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Incomplete numeric evidence")
    if data.duplicated(["symbol", times[0]]).any():
        raise ValueError("Conflicting evidence timestamps")
    return data.sort_values(times[0])


def audit(fills, quotes=None, locates=None):
    q = evidence_table(quotes, ["ts"], ["bid", "ask", "bid_size", "ask_size", "price_multiplier"])
    loan = evidence_table(locates, ["available_at", "valid_until"], ["max_shares", "price_multiplier"])
    if ((q.bid <= 0) | (q.ask < q.bid) | (q.price_multiplier <= 0) |
            (q.bid_size < 0) | (q.ask_size < 0)).any():
        raise ValueError("Invalid quote or unit conversion")
    if ((loan.valid_until < loan.available_at) | (loan.max_shares < 0) |
            (loan.price_multiplier <= 0)).any():
        raise ValueError("Invalid locate")
    keys = ["fold", "split", "scope", "book", "entry_date", "symbol"]
    out = fills.sort_values([*keys, "date", "minute"]).reset_index(drop=True).copy()
    decision = pd.to_datetime(out.entry_date) + pd.Timedelta(hours=10, minutes=30)
    boundary = pd.to_datetime(out.date) + pd.to_timedelta(571 + out.minute, unit="min")
    out["decision"] = decision.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    out["bar_end"] = boundary.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    short_entry = out.groupby(keys).cumcount().eq(0) & out.units.lt(0)
    out["quote_state"] = "unknown"
    out["locate_state"] = np.where(short_entry, "unknown", "not_required")
    out["quote_pnl_change"] = np.nan
    quote_history = {s: frame.set_index("ts") for s, frame in q.groupby("symbol")}
    loan_history = {s: frame.set_index("available_at") for s, frame in loan.groupby("symbol")}
    for i, fill in enumerate(out.itertuples()):
        quotes_for_symbol = quote_history.get(fill.symbol)
        k = -1 if quotes_for_symbol is None else quotes_for_symbol.index.searchsorted(fill.bar_end) - 1
        if k >= 0 and fill.bar_end - quotes_for_symbol.index[k] <= pd.Timedelta(seconds=2):
            mark = quotes_for_symbol.iloc[k]
            buy = fill.units > 0
            depth = mark.ask_size if buy else mark.bid_size
            if depth >= abs(fill.units) * mark.price_multiplier:
                px = (mark.ask if buy else mark.bid) * mark.price_multiplier
                out.at[i, "quote_state"] = "covered"
                out.at[i, "quote_pnl_change"] = fill.units * (fill.price - px)
            else:
                out.at[i, "quote_state"] = "insufficient_depth"
        if short_entry.iat[i] and fill.symbol in loan_history:
            known = loan_history[fill.symbol]
            k = known.index.searchsorted(fill.decision, side="right") - 1
            if k >= 0:
                rec = known.iloc[k]
                if rec.valid_until >= fill.bar_end:
                    enough = rec.max_shares >= abs(fill.units) * rec.price_multiplier
                    out.at[i, "locate_state"] = "documented" if enough else "insufficient"
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fills", type=Path, required=True)
    parser.add_argument("--quotes", type=Path)
    parser.add_argument("--locates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(pd.read_parquet(args.fills), args.quotes, args.locates)
    args.output.mkdir(parents=True, exist_ok=False)
    result.to_parquet(args.output / "evidence.parquet", index=False)
    result.groupby(["fold", "split", "scope", "book", "quote_state", "locate_state"]).size().rename(
        "fills"
    ).to_csv(args.output / "coverage.csv")
