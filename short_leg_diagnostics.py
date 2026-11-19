"""Factor and execution attribution for the frozen inverse-gap rule."""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from inverse_convergence_study import BASE, PAIR_WEIGHTS, events, fill, portfolio
from inverse_tracking_study import carry_years
from strict_alpha_evidence import block_interval
from strict_alpha_study import (
    CAPITAL,
    FEE,
    SNAPSHOT,
    STOP,
    TARGET,
    Tape,
    chart,
    common_price,
    hac_p,
    holm,
)

ORIGINAL = Path("outputs/runs/inverse_gap_reopening")
PROTOCOL = Path("config/inverse_short_leg_forward.json")
PROXIES = ["SPY", "TLT", "USO", "QQQ", "IWM", "DIA"]
FACTORS = ["SPY", "TLT", "USO", "QQQ_minus_SPY", "IWM_minus_SPY", "DIA_minus_SPY"]


def factors(values):
    x = np.array(values, dtype=float, copy=True)
    x[..., 3:] -= x[..., :1]
    return x


def exposure(x, y):
    """OLS with intercept. Return leg betas, projected leader size, and signed L-on-S slope."""
    if len(x) != 60 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Missing trailing exposure observations")
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    b, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != x.shape[1]:
        raise ValueError("Unidentified factor exposures")
    modeled = x @ b
    pair = (modeled[:, 0] - modeled[:, 1]) / 2
    ratio = modeled[:, 0] @ pair / (modeled[:, 0] @ modeled[:, 0])
    signed = y[:, 1] @ y[:, 0] / (y[:, 1] @ y[:, 1])
    return b.T, ratio, signed


def half_spread(price):
    """Roll half-spread proxy from consecutive minute log changes over 20 prior sessions."""
    r = np.diff(np.log(price), axis=1)
    x, y = r[:, :-1], r[:, 1:]
    keep = np.isfinite(x) & np.isfinite(y)
    x = np.where(keep, x, 0.0)
    y = np.where(keep, y, 0.0)
    daily = pd.DataFrame(
        {
            "n": keep.sum(axis=1),
            "x": x.sum(axis=1),
            "y": y.sum(axis=1),
            "xy": (x * y).sum(axis=1),
        }
    )
    s = daily.rolling(20).sum().shift()
    cov = (s.xy - s.x * s.y / s.n) / (s.n - 1)
    return np.sqrt(-cov.where((cov < 0) & (s.n >= 100))).to_numpy()


def factor_mark(tape, d, m):
    quotes = tape.price[d, max(0, m - 5) : m + 1]
    valid = quotes[np.isfinite(quotes)]
    if not len(valid):
        raise ValueError("No contemporaneous factor mark within five minutes")
    return valid[-1]


def leg_attribution(cal, tapes, spreads, trade, fills, betas):
    d = cal.get_loc(trade["date"])
    m = trade["entry_slot"]
    indices = [tapes[s] for s in PROXIES]
    ref = np.array([factor_mark(t, d, m) for t in indices])
    results = []
    for symbol, group in pd.DataFrame(fills).groupby("symbol"):
        opening = group.iloc[0]
        qty = opening.units
        t = tapes[symbol]
        beta = betas[symbol]
        gross = funding = borrow = factor_pnl = 0.0
        for ex in group.iloc[1:].itertuples():
            e = cal.get_loc(ex.date)
            q = -ex.units
            gross += q * (ex.price - opening.price + t.div[d + 1 : e + 1].sum())
            f, b = carry_years(
                cal,
                d,
                m,
                e,
                ex.minute,
                np.array([q * opening.price]),
                (q * t.close[d:e]).reshape(-1, 1),
            )
            funding += f
            borrow += b
            ends = np.array(
                [
                    factor_mark(tape, e, ex.minute) + tape.div[d + 1 : e + 1].sum()
                    for tape in indices
                ]
            )
            factor_pnl += q * opening.price * (beta @ factors(ends / ref - 1))
        np.testing.assert_allclose(group.units.sum(), 0.0, atol=1e-7)
        rates = spreads[symbol][cal.get_indexer(group.date)]
        unknown = ~np.isfinite(rates)
        base = group.fee.sum()
        estimated = group.notional.abs() @ np.maximum(np.nan_to_num(rates), FEE)
        net = gross - funding - borrow - base
        results.append(
            {
                "symbol": symbol,
                "position": "long" if qty > 0 else "short",
                "entry_notional": opening.notional,
                "gross": gross,
                "funding": funding,
                "borrow": borrow,
                "execution": base,
                "net": net,
                "factor_pnl": factor_pnl,
                "factor_adjusted": net - factor_pnl,
                "spread_cost": estimated,
                "spread_net": net + base - estimated,
                "turnover": group.notional.abs().sum(),
                "unidentified_spread_notional": group.loc[unknown, "notional"]
                .abs()
                .sum(),
                **dict(zip(FACTORS, beta, strict=True)),
            }
        )
    for name in ("gross", "funding", "borrow", "execution", "net"):
        np.testing.assert_allclose(
            sum(x[name] for x in results), trade[name], atol=1e-7
        )
    return results


def evaluate_book(cal, tapes, spreads, x, signals, start, end):
    daily, deals, _ = portfolio(cal, tapes, signals, start, end, "convergence")
    days = daily.index
    last = cal.get_loc(days[-1])
    balances = {}
    flat = {}
    legs = []
    sizes = []
    daily = daily.drop(columns="leader_only").rename(
        columns={"convergence": "original_replay"}
    )
    executions = []

    def simulate(row, betas, h, name, weights, direction, budget):
        a, b = row.a, row.b
        leader, lag = (a, b) if row.side_a == 1 else (b, a)
        acct = balances.get(name, CAPITAL)
        data = fill(
            cal,
            tapes,
            row,
            weights,
            direction,
            min(budget, TARGET * min(acct, CAPITAL)),
            last,
        )
        if data is None or flat.get(name, -1) >= row.day:
            raise ValueError(f"Cannot match {name} at {row.date}")
        closed, record, marks, fills = data
        if record["entry_slot"] != row.entry_slot:
            raise ValueError("Mismatched entry minute")
        flat[name] = closed
        balances[name] = acct + record["net"]
        if name not in daily:
            daily[name] = 0.0
        for k, mark in marks.items():
            daily.at[cal[k], name] += mark[0] - mark[1:4].sum()
        executions.extend({"entry_date": row.date, "book": name, **item} for item in fills)
        detail = leg_attribution(cal, tapes, spreads, record, fills, betas)
        legs.extend(
            {
                "date": row.date,
                "a": a,
                "b": b,
                "book": name,
                "leader": leader,
                "laggard": lag,
                "hedge_beta": h,
                **v,
            }
            for v in detail
        )
        return record["entry_gross"]

    for tr in (t for t in deals if t["rule"] == "convergence"):
        d = cal.get_loc(tr["date"])
        a, b = tr["a"], tr["b"]
        side = tr["side_a"]
        leader, lag = (a, b) if side == 1 else (b, a)
        before = slice(d - 60, d)
        y = np.column_stack([tapes[k].total[before] for k in (leader, lag)])
        slopes, ratio, h = exposure(x[before], y)
        betas = dict(zip((leader, lag), slopes, strict=True))
        if ratio <= 0:
            raise ValueError("Long leader cannot match the pair's factor direction")
        row = SimpleNamespace(**tr, day=d)
        leader_weights = np.array([float(side == 1), float(side == -1)])

        replay = partial(simulate, row, betas, h)
        original = replay("pair", PAIR_WEIGHTS, side, tr["entry_gross"])
        actual = replay("factor_leader", leader_weights, 1.0, ratio * original)
        # Scale the pair down as well when the leader's liquidity or gross limit binds.
        common = replay(
            "common_pair", PAIR_WEIGHTS, side, min(original, actual / ratio)
        )
        if not np.isclose(common * ratio, actual, rtol=1e-8):
            raise ValueError("Common exposure projection binds asymmetrically")
        hedge = np.array([1.0, -h]) / (1 + abs(h))
        replay("signed_hedge", hedge if side == 1 else hedge[::-1], 1.0, original)
        fl = x[before] @ slopes.T
        diff = 0.5 * common * (fl[:, 0] - fl[:, 1]) - actual * fl[:, 0]
        sizes.append(
            {
                "date": tr["date"],
                "a": a,
                "b": b,
                "leader": leader,
                "laggard": lag,
                "hedge_beta": h,
                "target_ratio": ratio,
                "actual_ratio": actual / original,
                "original_gross": original,
                "matched_pair_gross": common,
                "leader_gross": actual,
                "unmatched_daily_factor_risk_usd": diff.std(ddof=1),
                "pair_daily_factor_risk_usd": (
                    0.5 * common * (fl[:, 0] - fl[:, 1])
                ).std(ddof=1),
            }
        )
    for name in ("pair", "factor_leader", "common_pair", "signed_hedge"):
        if name not in daily:
            daily[name] = 0.0
    np.testing.assert_allclose(daily.pair, daily.original_replay, atol=1e-7)
    return daily, legs, sizes, executions


def opening_hour(tape):
    """First regular-session open to the close of the 10:29 minute."""
    return tape.price[:, 59] / tape.open - 1


def prediction_rows(cal, tapes, train, dep, year, split):
    frames = []
    market = factors(np.column_stack([tapes[s].from_reset[:, 59] for s in PROXIES]))
    for row in train.itertuples():
        a, b = (tapes[s] for s in (row.a, row.b))
        f = events(a, b)
        fit = (cal >= "2020-01-07") & (cal <= row.cutoff)
        cutoff = f.loc[fit, "gap_before"].abs().quantile(0.8)
        f = f[
            (abs(f.gap_before) >= cutoff) & f.leader_pullback & f.wave_change.notna()
        ].copy()
        d = f.day.to_numpy()
        f = f[(d >= 60) & (d < len(cal) - 1)].copy()
        d = f.day.to_numpy()
        positive = f.side_a.to_numpy() == 1
        returns = pd.DataFrame({"a": a.total, "b": b.total})
        mom = np.log1p(returns).rolling(20).sum().shift().to_numpy()[d]
        vol = returns.rolling(60).std().shift().to_numpy()[d]
        fa, fb = opening_hour(a)[d], opening_hour(b)[d]
        f["pullback"] = np.where(positive, f.a_return, f.b_return)
        f["leader_hour"] = np.where(positive, fa, fb)
        f["momentum"] = np.where(positive, mom[:, 0], mom[:, 1])
        f["volatility"] = np.where(positive, vol[:, 0], vol[:, 1])
        f["laggard_move"] = np.where(positive, f.b_return, f.a_return)
        f["laggard_up"] = (f.laggard_move > 0).astype(float)
        f["laggard_hour"] = np.where(positive, fb, fa)
        f["laggard_hour_up"] = (f.laggard_hour > 0).astype(float)
        f[FACTORS] = market[d]
        marks = common_price(a, b, 61)
        r = (marks[d + 1] + np.column_stack([a.div[d + 1], b.div[d + 1]])) / marks[
            d
        ] - 1
        f["pair_return"] = f.side_a * 0.5 * (r[:, 0] - r[:, 1])
        f["leader_return"] = np.where(positive, r[:, 0], r[:, 1])
        f["a"], f["b"] = row.a, row.b
        f["date"], f["end"] = cal[d], cal[d + 1]
        f["deployed"] = (row.a, row.b) in dep
        frames.append(f)
    data = pd.concat(frames, ignore_index=True)
    features = ["pullback", "leader_hour", "momentum", "volatility", *FACTORS]
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[*features, "laggard_move", "laggard_hour", "pair_return", "leader_return"]
    )
    tr = data[(data.date > f"{year - 4}-12-31") & (data.end < f"{year - 1}-01-01")]
    sy = year if split == "test" else year - 1
    te = data[
        (data.date >= f"{sy}-01-01") & (data.end <= f"{sy}-12-31") & data.deployed
    ].copy()
    metrics = []
    for target in ("pair_return", "leader_return"):
        for name, cols in [
            ("base", features),
            ("since_close", [*features, "laggard_move", "laggard_up"]),
            ("opening_hour", [*features, "laggard_hour", "laggard_hour_up"]),
        ]:
            model = make_pipeline(StandardScaler(), Ridge(alpha=100.0)).fit(
                tr[cols], tr[target]
            )
            te[target + "_" + name] = model.predict(te[cols])
        for scope, part in [
            ("all", te),
            ("ex_SCO_USO", te[~(te.a.eq("SCO") & te.b.eq("USO"))]),
        ]:
            for clock in ("since_close", "opening_hour"):
                base, aug = part[target + "_base"], part[target + "_" + clock]
                loss0 = (part[target] - base) ** 2
                loss1 = (part[target] - aug) ** 2
                improvement = (loss0 - loss1).groupby(part.date).mean()
                metrics.append(
                    {
                        "target": target,
                        "scope": scope,
                        "clock": clock,
                        "train_rows": len(tr),
                        "rows": len(part),
                        "dates": part.date.nunique(),
                        "base_mse": loss0.mean(),
                        "augmented_mse": loss1.mean(),
                        "paired_date_mse_gain": improvement.mean(),
                        "base_rank_ic": base.corr(part[target], method="spearman"),
                        "augmented_rank_ic": aug.corr(part[target], method="spearman"),
                        "p": hac_p(improvement),
                    }
                )
    return te[
        [
            "date",
            "end",
            "a",
            "b",
            *features,
            "laggard_move",
            "laggard_up",
            "laggard_hour",
            "laggard_hour_up",
            *[c for c in te if c.startswith(("pair_return", "leader_return"))],
        ]
    ], metrics


def forward_signals(cal, tapes, train, deployed, start, end):
    panels = []
    thresholds = pd.read_parquet(
        ORIGINAL / "signals_2026_test.parquet", columns=["a", "b", "cutoff"]
    ).drop_duplicates().set_index(["a", "b"], verify_integrity=True).cutoff
    for r in train.itertuples():
        if (r.a, r.b) not in deployed:
            continue
        f = events(tapes[r.a], tapes[r.b])
        cutoff = thresholds.loc[(r.a, r.b)]
        f = f.assign(date=cal, a=r.a, b=r.b, strength=f.gap_before.abs() / cutoff)
        panels.append(
            f[
                f.date.between(start, end)
                & f.strength.ge(1)
                & f.risk_ratio.gt(0)
                & f.wave_change.notna()
            ]
        )
    return pd.concat(panels, ignore_index=True)


def run(output, forward_daily=None, forward_minute=None):
    from data.fetch_tracking_minutes import CACHE

    protocol = json.loads(PROTOCOL.read_text())
    manifest = json.loads((BASE / "manifest.json").read_text())
    forward = forward_daily is not None or forward_minute is not None
    if forward and (forward_daily is None or forward_minute is None):
        raise ValueError("Supply both forward snapshot directories")
    start = protocol["first_signal_session"]
    end = protocol["last_session"]
    if (
        forward
        and pd.Timestamp.now(tz="America/New_York").date() <= pd.Timestamp(end).date()
    ):
        raise ValueError(f"Score the registered period only after {end}")
    directory = forward_daily if forward else SNAPSHOT
    minutes = forward_minute if forward else CACHE
    cal = chart("SPY", directory).index
    if forward and (
        end not in cal or start not in cal or cal[0] > pd.Timestamp("2020-01-07")
    ):
        raise ValueError(
            "Forward snapshots must include calibration history and the complete test interval"
        )
    if forward:
        cal = cal[cal <= end]
        sessions = pd.bdate_range(start, end).difference(pd.to_datetime(["2026-11-26", "2026-12-25"]))
        if not cal[cal >= start].equals(sessions):
            raise ValueError("Incomplete registered US exchange calendar")
    stop = str((cal[-1] + pd.Timedelta(days=1)).date()) if forward else STOP
    raw_inputs = [
        p
        for s in manifest["cohort"]
        for p in [directory / f"{s}.json", minutes / f"{s}.parquet"]
    ]
    hashes = {}
    expected = {
        Path(k.replace("\\", "/")).as_posix(): v for k, v in manifest["sha256"].items()
    }
    for p in raw_inputs:
        with p.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if not forward and expected[p.as_posix()] != digest:
            raise ValueError(f"Unexpected snapshot: {p}")
        hashes[str(p)] = digest
    tapes = {
        s: Tape(s, cal, directory, stop, "2021-12-31", minute_directory=minutes)
        for s in manifest["cohort"]
    }
    if forward:
        required = pd.read_csv(BASE / "2026/test/deployment.csv").query("eligible")
        relevant = set(required.a) | set(required.b) | set(PROXIES)
        for symbol in relevant:
            if (
                not np.isfinite(tapes[symbol].price[(cal >= start) & (cal <= end)])
                .any(axis=1)
                .all()
            ):
                raise ValueError(f"Missing forward session for {symbol}")
    output.mkdir(exist_ok=False, parents=True)
    x = factors(np.column_stack([tapes[s].total for s in PROXIES]))
    spreads = {s: half_spread(t.price) for s, t in tapes.items()}
    all_legs, all_sizes, all_curves, all_stats, tests, preds = [], [], [], [], [], []
    fill_ledger = []
    for year in [2026] if forward else [2025, 2026]:
        train = pd.read_csv(BASE / str(year) / "qualification.csv").query("eligible")
        if (
            train[
                [
                    "rho",
                    "sub1",
                    "sub2",
                    "sub3",
                    "supervised_train_rho",
                    "prevalidation_rho",
                ]
            ]
            .max()
            .max()
            > -0.80
        ):
            raise ValueError("Invalid training qualification")
        for split in ["forward"] if forward else ["validation", "test"]:
            root = BASE / str(year) / ("test" if forward else split)
            dep = pd.read_csv(root / "deployment.csv").query("eligible")
            if dep.rho.max() > -0.8:
                raise ValueError("Invalid deployment correlation")
            deployed = set(zip(dep.a, dep.b))
            if forward:
                signals = forward_signals(cal, tapes, train, deployed, start, end)
            else:
                sy = year if split == "test" else year - 1
                start, end = f"{sy}-01-01", min(pd.Timestamp(f"{sy}-12-31"), cal[-1])
                signals = pd.read_parquet(ORIGINAL / f"signals_{year}_{split}.parquet")
                forecasts, metrics = prediction_rows(
                    cal, tapes, train, deployed, year, split
                )
                forecasts.to_parquet(
                    output / f"forecasts_{year}_{split}.parquet", index=False
                )
                preds.extend({"fold": year, "split": split, **m} for m in metrics)
            for scope in ["all", "ex_SCO_USO"]:
                sample = (
                    signals
                    if scope == "all"
                    else signals[~(signals.a.eq("SCO") & signals.b.eq("USO"))]
                )
                tags = {"fold": year, "split": split, "scope": scope}
                curves, legs, sizes, executions = evaluate_book(
                    cal, tapes, spreads, x, sample, start, end
                )
                fill_ledger.extend({**tags, **r} for r in executions)
                if scope == "all" and not forward:
                    old = pd.read_parquet(ORIGINAL / "daily_equity.parquet")
                    orig = (
                        old[
                            (old.fold == year)
                            & (old.split == split)
                            & old.rule.eq("convergence")
                        ]
                        .set_index("date")
                        .pnl
                    )
                    np.testing.assert_allclose(
                        curves.pair.reindex(orig.index), orig, atol=1e-7
                    )
                all_curves.append(curves.reset_index(names="date").assign(**tags))
                all_legs.extend({**tags, **r} for r in legs)
                all_sizes.extend({**tags, **r} for r in sizes)
                for book in ("pair", "factor_leader", "common_pair", "signed_hedge"):
                    part = pd.DataFrame([r for r in legs if r["book"] == book])
                    sums = {
                        k: (part[k].sum() if len(part) else 0.0)
                        for k in (
                            "gross",
                            "funding",
                            "borrow",
                            "execution",
                            "net",
                            "factor_pnl",
                            "factor_adjusted",
                            "spread_cost",
                            "spread_net",
                            "turnover",
                            "unidentified_spread_notional",
                        )
                    }
                    all_stats.append(
                        {**tags, "book": book, "trades": len(sizes), **sums}
                    )
                for rule in ("pair", "common_pair"):
                    diff = (curves[rule] - curves.factor_leader) / CAPITAL
                    lo, hi = block_interval(diff)
                    tests.append(
                        {
                            **tags,
                            "comparison": rule + " minus factor_leader",
                            "difference": diff.sum(),
                            "ci_low": lo,
                            "ci_high": hi,
                            "p": hac_p(diff),
                        }
                    )
            print(year, split, "complete", flush=True)
    paired = pd.DataFrame(tests)
    paired["holm"] = paired.groupby(["split", "comparison"]).p.transform(holm)
    paired.to_csv(output / "paired.csv", index=False)
    if preds:
        models = pd.DataFrame(preds)
        models["holm"] = models.groupby("split").p.transform(holm)
        models.to_csv(output / "forecast_quality.csv", index=False)
    pd.DataFrame(fill_ledger, columns=["fold","split","scope","entry_date","book","date","minute","symbol","units","price","notional","fee"]).to_parquet(output / "fills.parquet", index=False)
    pd.DataFrame(all_stats).to_csv(output / "books.csv", index=False)
    pd.DataFrame(all_legs).to_csv(output / "legs.csv", index=False)
    pd.DataFrame(all_sizes).to_csv(output / "exposure_matching.csv", index=False)
    pd.concat(all_curves, ignore_index=True).to_parquet(
        output / "daily_pnl.parquet", index=False
    )
    sources = [
        PROTOCOL,
        Path("strict_alpha_evidence.py"),
        *ORIGINAL.glob("signals_*.parquet"),
        Path(__file__),
        Path("inverse_convergence_study.py"),
        Path("inverse_tracking_study.py"),
        Path("strict_alpha_study.py"),
        BASE / "manifest.json",
        ORIGINAL / "manifest.json",
        *BASE.glob("20*/qualification.csv"),
        *BASE.glob("20*/*/deployment.csv"),
    ]
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "mode": "forward" if forward else "retrospective",
                "last_session": str(cal[-1].date()),
                "market_sha256": hashes,
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
    parser.add_argument("--forward-daily", type=Path)
    parser.add_argument("--forward-minute", type=Path)
    args = parser.parse_args()
    run(args.output, args.forward_daily, args.forward_minute)
