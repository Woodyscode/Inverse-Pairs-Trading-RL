"""Walk-forward forecasts and matched minute execution for strict inverse pairs."""

import argparse
import hashlib
import itertools
import json
import platform
from collections import defaultdict
from functools import cache
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from data.fetch_tracking_minutes import CACHE, DAILY, REVISION, fetch, fetch_daily
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
)
from inverse_tracking_study import (
    Tape,
    chart,
    closing_slot,
    effective_beta,
    enter,
    exit_trade,
)

SNAPSHOT = DAILY.parent / "strict_alpha_daily_202610"
COHORT = sorted(
    [
        "DIA",
        "DOG",
        "DXD",
        "HYG",
        "IWM",
        "PSQ",
        "QID",
        "QQQ",
        "RWM",
        "SCO",
        "SDS",
        "SH",
        "SJB",
        "SPXU",
        "SPY",
        "SQQQ",
        "TBF",
        "TBT",
        "TLT",
        "TMF",
        "TMV",
        "TWM",
        "TZA",
        "UCO",
        "FXE",
        "EUO",
        "UUP",
        "UDN",
        "FXY",
        "YCS",
        "GLD",
        "GLL",
        "DBC",
        "USO",
        "AGG",
        "BND",
    ]
)
DAILY_FEATURES = [
    "daily_z",
    "daily_residual",
    "momentum",
    "daily_vol",
    "reference_vol",
    "beta",
    "daily_interaction",
]
MINUTE_FEATURES = [
    "opening_residual",
    "opening_z",
    "realized_vol",
    "relative_volume",
    "opening_interaction",
]
HORIZONS = (0, 1, 5, 10)
SEED = 42
STOP = "2026-10-24"


def qualify(tapes, pair, calendar, cutoff):
    a, b = (tapes[s] for s in pair)
    idx = np.flatnonzero((calendar >= "2020-01-07") & (calendar <= cutoff))
    ra, rb = a.total[idx], b.total[idx]
    reasons = []
    complete = np.isfinite(ra).all() and np.isfinite(rb).all()
    if not complete:
        reasons.append("incomplete synchronized history")
    rho = np.corrcoef(ra, rb)[0, 1]
    parts = [
        np.corrcoef(ra[i], rb[i])[0, 1] for i in np.array_split(np.arange(len(idx)), 3)
    ]
    beta = ols_slope(ra, rb)
    adv = min(np.median(t.daily_dollars[idx[-60:]]) for t in (a, b))
    if not rho <= RHO:
        reasons.append("correlation")
    if not np.all(np.array(parts) <= RHO):
        reasons.append("subwindow instability")
    if not beta < 0:
        reasons.append("hedge sign")
    if not adv >= MIN_ADV:
        reasons.append("liquidity")
    if not max(a.fit_error, b.fit_error) <= 0.01:
        reasons.append("minute price units")
    return {
        "a": pair[0],
        "b": pair[1],
        "cutoff": cutoff,
        "rho": rho,
        "sub1": parts[0],
        "sub2": parts[1],
        "sub3": parts[2],
        "beta": beta,
        "intercept": np.mean(ra - beta * rb),
        "min_adv": adv,
        "eligible": not reasons,
        "reasons": "|".join(reasons),
    }


def feature_frame(a, b, beta, intercept):
    residual = pd.Series(a.total - beta * b.total - intercept).shift()
    vol = residual.rolling(30).std()
    refvol = pd.Series(b.total).shift().rolling(30).std()
    z = (residual - residual.rolling(30).mean()) / vol
    opening = pd.Series(
        a.price[:, 59] / a.open
        - 1
        - effective_beta(beta, b.overnight, a.overnight) * (b.price[:, 59] / b.open - 1)
    )
    hour_z = (
        opening - opening.shift().rolling(30, min_periods=20).mean()
    ) / opening.shift().rolling(30, min_periods=20).std()
    minutes = np.diff(np.log(a.price[:, :60]), axis=1) - beta * np.diff(
        np.log(b.price[:, :60]), axis=1
    )
    count = np.isfinite(minutes).sum(axis=1)
    realized = np.sqrt(np.nansum(minutes**2, axis=1) * 59 / np.maximum(count, 1))
    realized[count < 45] = np.nan
    dv = pd.Series(np.minimum(a.dv[:, :60].sum(axis=1), b.dv[:, :60].sum(axis=1)))
    hedge = effective_beta(beta, b.from_reset[:, 59], a.from_reset[:, 59])
    return pd.DataFrame(
        {
            "daily_z": z,
            "daily_residual": residual,
            "momentum": residual.rolling(5).sum(),
            "daily_vol": vol,
            "reference_vol": refvol,
            "beta": beta,
            "daily_interaction": z * refvol,
            "opening_residual": opening,
            "opening_z": hour_z,
            "realized_vol": realized,
            "relative_volume": dv / dv.shift().rolling(30).median(),
            "opening_interaction": hour_z * refvol,
            "hedge": hedge,
        }
    )


def common_price(a, b, slot):
    prices = np.stack(
        (a.price[:, slot : slot + 5], b.price[:, slot : slot + 5]), axis=2
    )
    valid = np.isfinite(prices).all(axis=2)
    out = prices[np.arange(len(prices)), valid.argmax(axis=1)].copy()
    out[~valid.any(axis=1)] = np.nan
    return out


def opportunities(tapes, rows, calendar, horizon):
    frames = []
    for row in rows:
        pair = row["a"], row["b"]
        a, b = (tapes[s] for s in pair)
        f = feature_frame(a, b, row["beta"], row["intercept"])
        px = common_price(a, b, 61)
        exit_px = common_price(a, b, 121 if horizon == 0 else 61)
        distributions = np.cumsum(np.column_stack((a.div, b.div)), axis=0)
        future = (
            pd.DataFrame(exit_px + distributions).shift(-horizon).to_numpy()
            - distributions
        )
        r = future / px - 1
        f["label"] = (r[:, 0] - f.hedge * r[:, 1]) / (1 - f.hedge)
        f["a"], f["b"], f["day"] = *pair, np.arange(len(calendar))
        f["date"] = calendar
        f["label_end"] = pd.Series(calendar).shift(-horizon)
        frames.append(
            f.replace([np.inf, -np.inf], np.nan).dropna(
                subset=DAILY_FEATURES + MINUTE_FEATURES + ["hedge"]
            )
        )
    return pd.concat(frames, ignore_index=True).sort_values(["day", "a", "b"])


def models(train, test, model_dir):
    model_dir.mkdir(parents=True)
    y = train.label
    out = test.copy()
    out["zero"] = 0.0
    for name, cols in [
        ("daily", DAILY_FEATURES),
        ("minute", DAILY_FEATURES + MINUTE_FEATURES),
    ]:
        score = train.daily_z if name == "daily" else train.opening_z
        test_score = test.daily_z if name == "daily" else test.opening_z
        # Both prespecified signs have the same scale, estimated without using future labels.
        scale = y.abs().median() / max(score.abs().median(), 1e-8)
        for rule, sign in [("reversal", -1), ("continuation", 1)]:
            out[rule + "_" + name] = sign * scale * test_score.clip(-5, 5)
        for label, est in [
            ("linear", LinearRegression()),
            ("ridge", Ridge(alpha=100.0)),
            (
                "xgboost",
                XGBRegressor(
                    n_estimators=80,
                    max_depth=2,
                    learning_rate=0.04,
                    reg_lambda=10,
                    objective="reg:squarederror",
                    random_state=SEED,
                    n_jobs=1,
                ),
            ),
        ]:
            model = make_pipeline(StandardScaler(), est).fit(train[cols], y)
            key = label + "_" + name
            out[key] = model.predict(test[cols])
            joblib.dump(model, model_dir / (key + ".joblib"))
    out["regime"] = np.where(
        out.reference_vol > train.reference_vol.median(), "high", "low"
    )
    return out


def hac_p(series, lag=10):
    x = np.array(series, dtype=float)
    if len(x) < 20 or np.std(x) < 1e-15:
        return 1.0
    u = x - x.mean()
    v = u @ u
    for k in range(1, min(lag + 1, len(x))):
        v += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k])
    return float(norm.sf(x.sum() / np.sqrt(v))) if v > 0 else 1.0


def accuracy(data, key):
    d = data.dropna(subset=["label", key])
    y, p = d.label, d[key]
    loss = (y - p) ** 2
    diff = y**2 - loss
    variable = p.nunique() > 1
    return {
        "n": len(d),
        "mse": loss.mean(),
        "mse_gain": diff.mean(),
        "ic": y.corr(p) if variable else np.nan,
        "rank_ic": y.corr(p, method="spearman") if variable else np.nan,
        "directional_accuracy": ((p * y) > 0)[p != 0].mean(),
        "calibration_slope": ols_slope(y.to_numpy(), p.to_numpy())
        if variable
        else np.nan,
        "actual_mean": y.mean(),
        "predicted_mean": p.mean(),
        "cost_clear": (abs(p) > 2 * FEE).mean(),
        "mse_p": hac_p(diff.groupby(d.day).mean()),
    }


def stamp(calendar, day, minute):
    return (calendar[day] + pd.Timedelta(minutes=571 + minute)).tz_localize(
        "America/New_York"
    )


def marked_trade(calendar, tapes, pair, d, entry, side, beta, horizon, last):
    m, size, w, px = entry
    e = d + horizon
    fills = []
    result, flat = exit_trade(
        tapes,
        pair,
        calendar,
        d,
        entry,
        side,
        beta,
        e,
        121 if horizon == 0 else 61,
        last + 1,
        fills,
    )
    ts = [tapes[s] for s in pair]
    values = defaultdict(
        lambda: np.zeros(7)
    )  # gross, fees, funding, borrow, turnover, closing gross, session-weighted gross
    executions = []

    def book(day, slot, symbol, units, price):
        value = units * price
        executions.append(
            {
                "date": calendar[day],
                "minute": slot,
                "symbol": symbol,
                "units": units,
                "price": price,
                "notional": value,
                "fee": abs(value) * FEE,
            }
        )
        values[day][1] += abs(value) * FEE
        values[day][4] += abs(value)

    qty = side * size * w / px
    for s, q, p in zip(pair, qty, px):
        book(d, m, s, q, p)
    for day, slot, frac, marks in fills:
        q = qty * frac
        previous = px
        start = stamp(calendar, d, m)
        for k in range(d, day + 1):
            close = marks if k == day else np.array([t.close[k] for t in ts])
            end = stamp(calendar, k, slot if k == day else closing_slot(calendar[k]))
            years = (end - start).total_seconds() / (365 * 86400)
            div = np.array([t.div[k] for t in ts]) if k > d else 0.0
            v = q * previous
            session_minutes = closing_slot(calendar[k]) + 1
            held_minutes = (slot + 1 if k == day else session_minutes) - (
                m + 1 if k == d else 0
            )
            values[k][6] += abs(v).sum() * held_minutes / session_minutes
            values[k][0] += q @ (close + div - previous)
            values[k][2] += years * FUNDING * abs(v).sum()
            values[k][3] += years * BORROW * np.maximum(-v, 0).sum()
            if k < day:
                values[k][5] += np.abs(q * close).sum()
            previous, start = close, end
        for s, units, price in zip(pair, -q, marks):
            book(day, slot, s, units, price)
    sums = np.sum(list(values.values()), axis=0)[:5]
    np.testing.assert_allclose(
        sums,
        [result[k] for k in ["gross", "execution", "funding", "borrow", "turnover"]],
        atol=1e-7,
    )
    return result, flat, values, executions


def replay(
    calendar, tapes, sample, key, horizon, start, last, simulate, allow_short=True
):
    rng = np.random.default_rng(SEED)
    days = np.flatnonzero((calendar >= start) & (calendar <= last))
    daily = np.zeros((len(days), 9))
    available_equity = CAPITAL
    actions, trades, fills = [], [], []
    flat = -1
    for day, group in sample.groupby("day", sort=True):
        if day < days[0] or day > days[-1] - horizon - 3:
            continue
        if day <= flat:
            continue
        forecasts = (
            group[key].to_numpy()
            if key != "random"
            else rng.choice([-1.0, 0.0, 1.0], len(group))
        )
        # Include calendar-time financing and side-specific borrow through the scheduled exit.
        years = (
            stamp(calendar, day + horizon, 121 if horizon == 0 else 61)
            - stamp(calendar, day, 61)
        ).total_seconds() / (365 * 86400)
        edges = (
            np.abs(forecasts) - 2 * FEE - years * (FUNDING + BORROW * (forecasts < 0))
        )
        j = rng.integers(len(group)) if key == "random" else np.argmax(edges)
        row = group.iloc[j]
        side = int(np.sign(forecasts[j])) if edges[j] > 0 else 0
        if not allow_short and side < 0:
            side = 0
        pair = (row.a, row.b)
        rec = {
            "date": calendar[day],
            "a": row.a,
            "b": row.b,
            "forecast": forecasts[j],
            "side": side,
            "regime": row.regime,
        }
        actions.append(rec)
        if not side:
            continue
        limit = min(CAPITAL, available_equity) * TARGET
        sim = simulate(pair, int(day), side, horizon, int(days[-1]), limit)
        if sim is None:
            rec["filled"] = False
            continue
        result, flat, marks, events, size = sim
        available_equity += result["net"]
        rec["filled"] = True
        trades.append(
            {
                **rec,
                **result,
                "capacity": size,
                "holding_sessions": flat - day,
                "holding_hours": (
                    stamp(calendar, flat, result["exit_minute"])
                    - stamp(calendar, day, events[0]["minute"])
                ).total_seconds()
                / 3600,
            }
        )
        for k, arr in marks.items():
            daily[k - days[0], :7] += arr
            daily[k - days[0], 7:] += side * arr[5:7]
        fills.extend(dict(**ev, entry_day=day) for ev in events)
    curve = pd.DataFrame(
        daily,
        columns=[
            "gross",
            "execution",
            "funding",
            "borrow",
            "turnover",
            "closing_gross",
            "session_gross",
            "closing_net",
            "session_net",
        ],
        index=calendar[days],
    )
    curve["pnl"] = curve.gross - curve[["execution", "funding", "borrow"]].sum(axis=1)
    curve["equity"] = CAPITAL + curve.pnl.cumsum()
    returns = curve.pnl / curve.equity.shift(fill_value=CAPITAL)
    equity = curve.equity.to_numpy() / CAPITAL
    dd = (1 - equity / np.maximum.accumulate(np.r_[1.0, equity])[1:]).max()
    annual = equity[-1] ** (252 / len(equity)) - 1
    downside = np.sqrt(np.mean(returns.clip(upper=0) ** 2))
    rng = np.random.default_rng(SEED)
    starts = rng.integers(len(returns), size=(999, (len(returns) + 9) // 10))
    ix = ((starts[:, :, None] + np.arange(10)) % len(returns)).reshape(999, -1)[
        :, : len(returns)
    ]
    ci = np.quantile(curve.pnl.to_numpy()[ix].sum(axis=1) / CAPITAL, [0.025, 0.975])
    metrics = {
        "net": equity[-1] - 1,
        "annual": annual,
        "sharpe": returns.mean() / returns.std() * np.sqrt(252)
        if returns.std()
        else np.nan,
        "sortino": returns.mean() / downside * np.sqrt(252) if downside else np.nan,
        "max_dd": dd,
        "calmar": annual / dd if dd else np.nan,
        "turnover": curve.turnover.sum() / CAPITAL,
        "trades": len(trades),
        "gross": curve.gross.sum(),
        "execution": curve.execution.sum(),
        "funding": curve.funding.sum(),
        "borrow": curve.borrow.sum(),
        "mean_closing_gross": (curve.closing_gross / curve.equity).mean(),
        "mean_closing_net": (curve.closing_net / curve.equity).mean(),
        "mean_session_gross": (curve.session_gross / curve.equity).mean(),
        "mean_session_net": (curve.session_net / curve.equity).mean(),
        "hold_hours": np.mean([t["holding_hours"] for t in trades])
        if trades
        else np.nan,
        "median_fill": np.median([t["capacity"] for t in trades]) if trades else np.nan,
        "net_p": hac_p(returns),
        "ci_low": ci[0],
        "ci_high": ci[1],
    }
    return metrics, curve.rename_axis("date").reset_index(), actions, trades, fills


def study(output):
    output.mkdir(parents=True, exist_ok=False)
    calendar = chart("SPY", SNAPSHOT).index
    if calendar[-1] != pd.Timestamp("2026-10-23") or not calendar.is_unique:
        raise ValueError("Unexpected daily snapshot calendar")
    symbols = [
        s
        for s in COHORT
        if (CACHE / f"{s}.parquet").exists() and (SNAPSHOT / f"{s}.json").exists()
    ]
    tapes = {s: Tape(s, calendar, SNAPSHOT, STOP, "2021-12-31") for s in symbols}
    files = [
        Path(__file__),
        Path("inverse_tracking_study.py"),
        Path("inverse_pair_discovery.py"),
        Path("data/fetch_tracking_minutes.py"),
        *[CACHE / f"{s}.parquet" for s in symbols],
        *[SNAPSHOT / f"{s}.json" for s in symbols],
    ]
    config = {
        "seed": SEED,
        "cohort": COHORT,
        "missing": sorted(set(COHORT) - set(symbols)),
        "threshold": RHO,
        "subwindows": 3,
        "account": CAPITAL,
        "fee": FEE,
        "borrow": BORROW,
        "funding": FUNDING,
        "participation": ADV_LIMIT,
        "gross_limit": TARGET,
        "adv_floor": MIN_ADV,
        "horizons": list(HORIZONS),
        "daily_features": DAILY_FEATURES,
        "minute_features": MINUTE_FEATURES,
        "xgboost": {
            "n_estimators": 80,
            "max_depth": 2,
            "learning_rate": 0.04,
            "reg_lambda": 10,
        },
        "ridge_alpha": 100,
        "gate": "Both date-clustered MSE gain over zero and net profit over cash positive with Holm p < .05; minimum 8 validation trades. Rank passing models by validation net.",
        "minute_revision": REVISION,
        "python": platform.python_version(),
        "packages": {
            p: version(p)
            for p in [
                "numpy",
                "pandas",
                "statsmodels",
                "scikit-learn",
                "xgboost",
                "scipy",
                "pyarrow",
                "stable-baselines3",
                "torch",
            ]
        },
        "sha256": {
            str(p): hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
            for p in files
        },
    }
    (output / "manifest.json").write_text(json.dumps(config, indent=2))
    panel = pd.DataFrame(
        [
            {
                "symbol": s,
                "basis": t.unit_mode,
                "training_error": t.fit_error,
                "observed_minutes": np.isfinite(t.price).sum(),
            }
            for s, t in tapes.items()
        ]
    )
    panel.to_csv(output / "tape_quality.csv", index=False)
    # A trained relationship is frozen before the first forecast-model training row.
    for year in (2025, 2026):
        root = output / str(year)
        root.mkdir()
        cutoff = f"{year - 4}-12-31"
        rows = [
            qualify(tapes, p, calendar, cutoff)
            for p in itertools.combinations(symbols, 2)
        ]
        for r in rows:
            if r["eligible"]:
                # Reconfirm eligibility at the last close before validation. Keep the original hedge.
                recent = (calendar >= f"{year - 2}-01-01") & (
                    calendar < f"{year - 1}-01-01"
                )
                a, b = tapes[r["a"]], tapes[r["b"]]
                train_dates = (calendar > cutoff) & (calendar < f"{year - 1}-01-01")
                r["supervised_train_rho"] = np.corrcoef(
                    a.total[train_dates], b.total[train_dates]
                )[0, 1]
                r["prevalidation_rho"] = np.corrcoef(a.total[recent], b.total[recent])[
                    0, 1
                ]
                r["prevalidation_adv"] = min(
                    np.median(t.daily_dollars[recent][-60:]) for t in (a, b)
                )
                if not (
                    r["supervised_train_rho"] <= RHO
                    and r["prevalidation_rho"] <= RHO
                    and r["prevalidation_adv"] >= MIN_ADV
                ):
                    r.update(
                        eligible=False, reasons="prevalidation correlation or liquidity"
                    )
        pd.DataFrame(rows).to_csv(root / "qualification.csv", index=False)
        accepted = [r for r in rows if r["eligible"]]
        print(year, "eligible", len(accepted), flush=True)
        bypair = {(r["a"], r["b"]): r for r in accepted}
        frames = {}
        for h in HORIZONS:
            f = opportunities(tapes, accepted, calendar, h)
            train = f[(f.date > cutoff) & (f.label_end < f"{year - 1}-01-01")].dropna(
                subset=["label"]
            )
            prediction = models(
                train,
                f[(f.date >= f"{year - 1}-01-01") & (f.date < f"{year + 1}-01-01")],
                root / f"models_{h}",
            )
            frames[h] = prediction
        for split, split_year in [("validation", year - 1), ("test", year)]:
            splitdir = root / split
            splitdir.mkdir()
            start = f"{split_year}-01-01"
            last = min(pd.Timestamp(f"{split_year}-12-31"), calendar[-1])
            metrics, detail, portfolios, actions, trades, fills, curves = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            deployment = []
            restricted = []
            for pair, row in bypair.items():
                recent = (calendar >= f"{split_year - 1}-01-01") & (calendar < start)
                a, b = (tapes[s] for s in pair)
                rho = np.corrcoef(a.total[recent], b.total[recent])[0, 1]
                adv = np.array(
                    [np.median(t.daily_dollars[recent][-60:]) for t in (a, b)]
                )
                w = np.array([1.0, -row["beta"]]) / (1.0 - row["beta"])
                eligible = bool(rho <= RHO and np.all(adv >= MIN_ADV))
                deployment.append(
                    {
                        "a": pair[0],
                        "b": pair[1],
                        "effective_date": start,
                        "rho": rho,
                        "adv_a": adv[0],
                        "adv_b": adv[1],
                        "adv_gross_capacity": np.min(ADV_LIMIT * adv / w),
                        "eligible": eligible,
                        "reason": "qualified"
                        if eligible
                        else "trailing correlation or liquidity",
                    }
                )
            pd.DataFrame(deployment).to_csv(splitdir / "deployment.csv", index=False)
            for h, f in frames.items():
                sample = f[(f.date >= start) & (f.date <= last)].copy()
                for decision in deployment:
                    if not decision["eligible"]:
                        sample = sample[
                            ~(sample.a.eq(decision["a"]) & sample.b.eq(decision["b"]))
                        ]
                sample.loc[sample.label_end > last, "label"] = np.nan
                sample.to_parquet(splitdir / f"forecasts_{h}.parquet", index=False)
                lookup = sample.set_index(["a", "b", "day"]).hedge.to_dict()

                @cache
                def simulate(
                    pair, d, side, horizon, end, limit, lookup=lookup, bypair=bypair
                ):
                    entry = enter(tapes, pair, d, 61, lookup[(*pair, d)])
                    if entry is None:
                        return None
                    m, size, w, px = entry
                    entry = (m, min(size, limit), w, px)
                    return (
                        *marked_trade(
                            calendar,
                            tapes,
                            pair,
                            d,
                            entry,
                            side,
                            bypair[pair]["beta"],
                            horizon,
                            end,
                        ),
                        entry[1],
                    )

                keys = [
                    "zero",
                    *[
                        m + "_" + s
                        for s in ("daily", "minute")
                        for m in (
                            "linear",
                            "ridge",
                            "xgboost",
                            "reversal",
                            "continuation",
                        )
                    ],
                ]
                for key in [*keys, "random"]:
                    tags = {"horizon": h, "model": key}
                    if key != "random":
                        metrics.append(dict(**tags, **accuracy(sample, key)))
                        for group, part in sample.groupby(["a", "b", "regime"]):
                            if part.label.count() >= 10:
                                detail.append(
                                    dict(
                                        **tags,
                                        a=group[0],
                                        b=group[1],
                                        regime=group[2],
                                        **accuracy(part, key),
                                    )
                                )
                    perf, curve, act, tr, ex = replay(
                        calendar, tapes, sample, key, h, start, last, simulate
                    )
                    portfolios.append(dict(**tags, **perf))
                    blocked, _, _, _, _ = replay(
                        calendar,
                        tapes,
                        sample,
                        key,
                        h,
                        start,
                        last,
                        simulate,
                        allow_short=False,
                    )
                    restricted.append(dict(**tags, **blocked))
                    curves.append(curve.assign(**tags))
                    for target, source in [(actions, act), (trades, tr), (fills, ex)]:
                        target.extend({**tags, **row} for row in source)
                simulate.cache_clear()
                print(year, split, "horizon", h, flush=True)
            result = pd.DataFrame(portfolios).merge(
                pd.DataFrame(metrics), on=["model", "horizon"], how="left"
            )
            # Each hypothesis is a fixed portfolio across all pairs. Jointly adjust both endpoints and all arms.
            p = result[["net_p", "mse_p"]].fillna(1).to_numpy()
            result[["net_holm", "mse_holm"]] = holm(p.ravel()).reshape(p.shape)
            result.to_csv(splitdir / "metrics.csv", index=False)
            pd.DataFrame(restricted).to_csv(splitdir / "no_locates.csv", index=False)
            pd.DataFrame(detail).to_csv(splitdir / "pair_regime.csv", index=False)
            for name, items in [
                ("actions", actions),
                ("trades", trades),
                ("executions", fills),
            ]:
                pd.DataFrame(items).to_parquet(
                    splitdir / (name + ".parquet"), index=False
                )
            pd.concat(curves).to_parquet(splitdir / "daily_equity.parquet", index=False)
            if split == "validation":
                survivors = result[
                    (result.net_holm < 0.05)
                    & (result.mse_holm < 0.05)
                    & (result.trades >= 8)
                ].sort_values("net", ascending=False)
                survivors.to_csv(root / "validated_alpha.csv", index=False)
                print(year, "validation survivors", len(survivors), flush=True)


def fetch_data():
    from concurrent.futures import ThreadPoolExecutor

    SNAPSHOT.mkdir(exist_ok=True, parents=True)
    CACHE.mkdir(exist_ok=True, parents=True)
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(lambda s: fetch_daily(s, SNAPSHOT, 1792800000), COHORT))
        list(pool.map(fetch, COHORT))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.fetch:
        fetch_data()
    if args.output:
        study(args.output)
