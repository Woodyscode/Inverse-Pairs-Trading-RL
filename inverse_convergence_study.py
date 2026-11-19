"""Buy the prior leader and sell the laggard after an observed inverse-pair contraction."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.wavelet import donoho_johnstone_endpoint
from inverse_tracking_study import enter_weighted
from strict_alpha_evidence import block_interval
from strict_alpha_study import (
    CAPITAL,
    SNAPSHOT,
    STOP,
    TARGET,
    Tape,
    chart,
    common_price,
    hac_p,
    holm,
    marked_trade,
)

BASE = Path("outputs/runs/strict_inverse_alpha_final")
PAIR_WEIGHTS = np.array([0.5, -0.5])


def events(a, b):
    """Evaluate the gap at yesterday's close and 10:30 against the same rebasing date."""
    delta = np.log1p(a.total) - np.log1p(b.total)
    history = pd.Series(delta).rolling(20).sum().shift().to_numpy()
    ra, rb = a.from_reset[:, 59], b.from_reset[:, 59]
    now = history + np.log1p(ra) - np.log1p(rb)
    leader = np.sign(history)
    pullback = np.where(leader > 0, ra, rb) < 0
    laggard_up = np.where(leader > 0, rb, ra) > 0
    open_gap = leader * now > 0
    # Reconstruct both endpoints in the same local log-wealth units. Each of the
    # 64-point filters ends at the corresponding observation, including session gaps.
    changes = np.lib.stride_tricks.sliding_window_view(delta, 63)[:-1]
    path = np.column_stack([np.zeros(len(changes)), np.cumsum(changes, axis=1)])
    latest = path[:, -1] + np.log1p(ra[63:]) - np.log1p(rb[63:])
    shifted = np.column_stack([path[:, 1:], latest])
    wave_change = np.r_[
        np.full(63, np.nan),
        donoho_johnstone_endpoint(shifted) - donoho_johnstone_endpoint(path),
    ]
    pair_vol = pd.Series(0.5 * (a.total - b.total)).rolling(60).std().shift()
    leader_vol = np.where(
        leader > 0,
        pd.Series(a.total).rolling(60).std().shift(),
        pd.Series(b.total).rolling(60).std().shift(),
    )
    raw = open_gap & pullback & laggard_up
    return pd.DataFrame(
        {
            "day": np.arange(len(now)),
            "gap_before": history,
            "gap_at_signal": now,
            "a_return": ra,
            "b_return": rb,
            "side_a": leader,
            "leader_pullback": pullback & open_gap,
            "convergence": raw,
            "wavelet": raw & (leader * wave_change < 0),
            "wave_change": wave_change,
            "risk_ratio": pair_vol / leader_vol,
        }
    ).replace([np.inf, -np.inf], np.nan)


def fill(calendar, tapes, row, weights, side, limit, last):
    pair = (row.a, row.b)
    entry = enter_weighted(tapes, pair, row.day, 61, weights)
    if entry is None:
        return None
    m, size, w, px = entry
    size = min(size, limit)
    if size <= 0:
        return None
    result, flat, marks, ledger = marked_trade(
        calendar, tapes, pair, row.day, (m, size, w, px), side, None, 1, last
    )
    rec = {
        "date": calendar[row.day],
        "a": row.a,
        "b": row.b,
        "side_a": int(row.side_a),
        "gap_before": row.gap_before,
        "gap_at_signal": row.gap_at_signal,
        "entry_slot": m,
        "entry_gross": size,
        **{k: v for k, v in result.items() if k != "reset_excess"},
    }
    return flat, rec, marks, [x for x in ledger if x["units"] != 0]


def portfolio(calendar, tapes, sample, start, end, rule):
    paired = rule == "convergence"
    names = [rule, "leader_only"] if paired else [rule]
    days = np.flatnonzero((calendar >= start) & (calendar <= end))
    accounts = dict.fromkeys(names, CAPITAL)
    pnl = pd.DataFrame(0.0, index=calendar[days], columns=names)
    ledger, trades = [], []
    flat = -1
    for d, rows in sample[sample[rule]].groupby("day", sort=True):
        if d <= flat or d < days[0] or d > days[-1] - 4:
            continue
        # This ranking uses only the pre-event gap and the training cutoff.
        row = next(
            rows.sort_values(
                ["strength", "a", "b"], ascending=[False, True, True]
            ).itertuples()
        )
        result = fill(
            calendar,
            tapes,
            row,
            PAIR_WEIGHTS,
            row.side_a,
            TARGET * min(CAPITAL, accounts[rule]),
            days[-1],
        )
        if result is None:
            continue
        legs = {rule: result}
        if paired:
            leader = np.array([float(row.side_a > 0), float(row.side_a < 0)])
            budget = min(
                result[1]["entry_gross"] * row.risk_ratio,
                TARGET * min(CAPITAL, accounts["leader_only"]),
            )
            shadow = fill(calendar, tapes, row, leader, 1.0, budget, days[-1])
            if shadow is None or shadow[1]["entry_slot"] != result[1]["entry_slot"]:
                raise ValueError(
                    "Leader comparison lacks a matched execution timestamp"
                )
            legs["leader_only"] = shadow
        for name, (closed, record, marks, executions) in legs.items():
            flat = max(flat, closed)
            accounts[name] += record["net"]
            trades.append({"rule": name, **record})
            ledger.extend(
                {"rule": name, "entry_date": record["date"], **x} for x in executions
            )
            for k, v in marks.items():
                pnl.at[calendar[k], name] += v[0] - v[1:4].sum()
    for name in names:
        np.testing.assert_allclose(pnl[name].sum(), accounts[name] - CAPITAL, atol=1e-7)
    return pnl, trades, ledger


def score_events(signals, calendar, tapes, last):
    """Outcomes for every joint event, computed separately from all trading decisions."""
    results = []
    for (asset_a, asset_b), rows in signals[signals.convergence].groupby(["a", "b"]):
        a, b = tapes[asset_a], tapes[asset_b]
        rows = rows[rows.day < calendar.searchsorted(last, side="right") - 1].copy()
        d = rows.day.to_numpy()
        prices = common_price(a, b, 61)
        entry, exit_px = prices[d], prices[d + 1]
        exit_total = exit_px + np.column_stack([a.div[d + 1], b.div[d + 1]])
        forward = np.log(exit_total / entry)
        # Link yesterday's close, today's official total return and tomorrow's
        # distribution-adjusted mark to keep the original 20-session anchor fixed.
        partial = np.log(exit_total / np.column_stack([a.close[d], b.close[d]]))
        end_change = (
            np.log1p(a.total[d]) - np.log1p(b.total[d]) + partial[:, 0] - partial[:, 1]
        )
        rows["directional_gap_change"] = rows.side_a * (forward[:, 0] - forward[:, 1])
        rows["change_from_previous_gap"] = rows.side_a * end_change
        rows = rows.dropna(
            subset=["directional_gap_change", "change_from_previous_gap"]
        )
        rows["reopened_from_entry"] = rows.directional_gap_change > 0
        rows["recovered_previous_gap"] = rows.change_from_previous_gap >= 0
        results.append(rows)
    return (
        pd.concat(results, ignore_index=True)
        if results
        else pd.DataFrame(columns=["reopened_from_entry", "recovered_previous_gap"])
    )


def run(out):
    config = json.loads((BASE / "manifest.json").read_text())
    for filename, digest in config["sha256"].items():
        p = Path(filename.replace("\\", "/"))
        if p.suffix in {".json", ".parquet"}:
            with p.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                    raise ValueError(f"Market snapshot mismatch: {p}")
    out.mkdir(parents=True, exist_ok=False)
    cal = chart("SPY", SNAPSHOT).index
    tapes = {s: Tape(s, cal, SNAPSHOT, STOP, "2021-12-31") for s in config["cohort"]}
    stats, tests, alltrades, allfills, curves, counts = [], [], [], [], [], []
    for year in (2025, 2026):
        root = BASE / str(year)
        train = pd.read_csv(root / "qualification.csv").query("eligible")
        panels = []
        for row in train.itertuples():
            if (
                max(
                    row.rho,
                    row.sub1,
                    row.sub2,
                    row.sub3,
                    row.supervised_train_rho,
                    row.prevalidation_rho,
                )
                > -0.80
            ):
                raise ValueError("Invalid inverse-pair qualification")
            frame = events(tapes[row.a], tapes[row.b])
            fit = (cal >= "2020-01-07") & (cal <= row.cutoff)
            cutoff = frame.loc[fit, "gap_before"].abs().quantile(0.80)
            frame = frame.assign(
                a=row.a,
                b=row.b,
                date=cal,
                cutoff=cutoff,
                strength=abs(frame.gap_before) / cutoff,
            )
            panels.append(frame)
        panel = pd.concat(panels, ignore_index=True)
        for split, y in [("validation", year - 1), ("test", year)]:
            dep = pd.read_csv(root / split / "deployment.csv").query("eligible")
            assert dep.rho.le(-0.8).all()
            start, last = f"{y}-01-01", min(cal[-1], pd.Timestamp(f"{y}-12-31"))
            tags = {"fold": year, "split": split}
            sample = panel[panel.date.between(start, last)].merge(
                dep[["a", "b"]], on=["a", "b"], validate="many_to_one"
            )
            sample = sample.dropna(
                subset=["gap_before", "gap_at_signal", "wave_change", "risk_ratio"]
            )
            signals = sample[(sample.strength >= 1) & (sample.risk_ratio > 0)]
            signals.to_parquet(out / f"signals_{year}_{split}.parquet", index=False)
            outcomes = score_events(signals, cal, tapes, last)
            outcomes.to_parquet(out / f"outcomes_{year}_{split}.parquet", index=False)
            counts.append(
                {
                    **tags,
                    "qualified_pairs": len(dep),
                    "observations": len(sample),
                    "large_gap": len(signals),
                    "scored_convergence": len(outcomes),
                    "reopened_from_entry": outcomes.reopened_from_entry.mean(),
                    "recovered_previous_gap": outcomes.recovered_previous_gap.mean(),
                    **{
                        key: int(signals[key].sum())
                        for key in ("convergence", "leader_pullback", "wavelet")
                    },
                }
            )
            books = {}
            for rule in ("convergence", "wavelet", "leader_pullback"):
                pnl, trades, fills = portfolio(cal, tapes, signals, start, last, rule)
                alltrades.extend({**tags, **x} for x in trades)
                allfills.extend({**tags, **x} for x in fills)
                for name in pnl.columns:
                    series = pnl[name] / CAPITAL
                    books[name] = series
                    curve = CAPITAL + pnl[name].cumsum()
                    selected = [x for x in trades if x["rule"] == name]
                    lo, hi = block_interval(series)
                    stats.append(
                        {
                            **tags,
                            "rule": name,
                            "net_return": series.sum(),
                            "ci_low": lo,
                            "ci_high": hi,
                            "drawdown": (
                                1 - curve / curve.cummax().clip(lower=CAPITAL)
                            ).max(),
                            "trades": len(selected),
                            **{
                                k: sum(x[k] for x in selected)
                                for k in (
                                    "gross",
                                    "execution",
                                    "funding",
                                    "borrow",
                                    "turnover",
                                )
                            },
                        }
                    )
                    curves.append(
                        pd.DataFrame(
                            {
                                "date": pnl.index,
                                "rule": name,
                                "pnl": pnl[name],
                                "equity": curve,
                                **tags,
                            }
                        )
                    )
            for left, right in [
                ("convergence", "cash"),
                ("convergence", "leader_only"),
                ("convergence", "leader_pullback"),
                ("wavelet", "convergence"),
            ]:
                diff = books[left] - (0 if right == "cash" else books[right])
                lo, hi = block_interval(diff)
                tests.append(
                    {
                        **tags,
                        "rule": left,
                        "benchmark": right,
                        "increment": diff.sum(),
                        "ci_low": lo,
                        "ci_high": hi,
                        "p": hac_p(diff),
                    }
                )
    paired = pd.DataFrame(tests)
    paired["p_holm"] = paired.groupby("split").p.transform(holm)
    paired.to_csv(out / "paired.csv", index=False)
    pd.DataFrame(stats).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(counts).to_csv(out / "coverage.csv", index=False)
    pd.DataFrame(alltrades).to_csv(out / "trades.csv", index=False)
    pd.DataFrame(allfills).to_parquet(out / "executions.parquet", index=False)
    pd.concat(curves, ignore_index=True).to_parquet(
        out / "daily_equity.parquet", index=False
    )
    sources = [
        Path(__file__),
        Path("data/wavelet.py"),
        Path("strict_alpha_study.py"),
        Path("inverse_tracking_study.py"),
        Path("strict_alpha_evidence.py"),
        BASE / "manifest.json",
        *BASE.glob("20*/qualification.csv"),
        *BASE.glob("20*/*/deployment.csv"),
    ]
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "prior_sessions": 20,
                "training_gap_quantile": 0.80,
                "holding_sessions": 1,
                "risk_lookback": 60,
                "signal_slot": 59,
                "entry_slot": 61,
                "db4_window": 64,
                "wavelet": "Donoho-Johnstone universal soft, level 1",
                "account": CAPITAL,
                "source_sha256": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
