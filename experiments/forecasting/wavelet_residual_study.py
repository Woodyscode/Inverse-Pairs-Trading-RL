"""Matched one-session Ridge forecasts using raw, moving-average and wavelet residuals."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.wavelet import donoho_johnstone_endpoint
from experiments.forecasting.strict_alpha_evidence import block_interval
from experiments.forecasting.strict_alpha_study import (
    CAPITAL,
    DAILY_FEATURES,
    MINUTE_FEATURES,
    SNAPSHOT,
    STOP,
    Tape,
    accuracy,
    chart,
    effective_beta,
    enter,
    hac_p,
    holm,
    marked_trade,
    opportunities,
    replay,
)

BASE = Path("outputs/runs/strict_inverse_alpha_final")
ARMS = ("raw", "ma3", "db4")


def transforms(tapes, row):
    a, b = tapes[row.a], tapes[row.b]
    daily = pd.Series(a.total - row.beta * b.total - row.intercept).shift()
    daily_windows = np.lib.stride_tricks.sliding_window_view(daily.to_numpy(), 64)
    daily_wave = pd.Series(
        np.r_[np.full(63, np.nan), donoho_johnstone_endpoint(daily_windows)]
    )
    hedge = effective_beta(row.beta, b.overnight, a.overnight)
    opening = (a.price[:, 28:60] / a.open[:, None] - 1) - hedge[:, None] * (
        b.price[:, 28:60] / b.open[:, None] - 1
    )
    f = pd.DataFrame({"day": np.arange(len(daily)), "a": row.a, "b": row.b})
    for name, day, hour in [
        ("ma3", daily.rolling(3).mean(), pd.Series(opening[:, -3:].mean(axis=1))),
        ("db4", daily_wave, pd.Series(donoho_johnstone_endpoint(opening))),
    ]:
        hist = day.rolling(30)
        f[name + "_daily_residual"] = day
        f[name + "_daily_z"] = (day - hist.mean()) / hist.std()
        f[name + "_momentum"] = day.rolling(5).sum()
        prev = hour.shift().rolling(30, min_periods=20)
        f[name + "_opening_residual"] = hour
        f[name + "_opening_z"] = (hour - prev.mean()) / prev.std()
    return f


def compare_fold(output, year, tapes, calendar):
    archived = BASE / str(year)
    pairs = pd.read_csv(archived / "qualification.csv").query("eligible")
    assert (
        (pairs[["rho", "supervised_train_rho", "prevalidation_rho"]] <= -0.80)
        .all()
        .all()
    )
    beta = pairs.set_index(["a", "b"]).beta.to_dict()
    f = opportunities(tapes, pairs.to_dict("records"), calendar, 1)
    extra = pd.concat([transforms(tapes, r) for r in pairs.itertuples()])
    f = f.merge(extra, on=["a", "b", "day"], validate="one_to_one").sort_values(
        ["day", "a", "b"]
    )
    for arm in ARMS[1:]:
        f[arm + "_daily_interaction"] = f[arm + "_daily_z"] * f.reference_vol
        f[arm + "_opening_interaction"] = f[arm + "_opening_z"] * f.reference_vol
    f = f.replace([np.inf, -np.inf], np.nan)
    metrics, comparisons, forecasts, curves, fills, coverage = [], [], [], [], [], []
    coefficients = []
    for frequency in ("daily", "opening"):
        columns = (
            DAILY_FEATURES if frequency == "daily" else DAILY_FEATURES + MINUTE_FEATURES
        )
        replace = (
            ["daily_residual", "daily_z", "momentum", "daily_interaction"]
            if frequency == "daily"
            else ["opening_residual", "opening_z", "opening_interaction"]
        )
        features = {
            arm: [
                arm + "_" + c if arm != "raw" and c in replace else c for c in columns
            ]
            for arm in ARMS
        }
        matched = f.dropna(subset=features["db4"] + features["ma3"]).copy()
        train = matched[
            (matched.date > f"{year - 4}-12-31")
            & (matched.label_end < f"{year - 1}-01-01")
        ].dropna(subset="label")
        matched["regime"] = np.where(
            matched.reference_vol > train.reference_vol.median(), "high", "low"
        )
        for arm, cols in features.items():
            model = make_pipeline(StandardScaler(), Ridge(alpha=100.0)).fit(
                train[cols], train.label
            )
            matched[arm] = model.predict(matched[cols])
            scaler, reg = model.steps[0][1], model.steps[1][1]
            coefficients.append(
                {
                    "frequency": frequency,
                    "arm": arm,
                    "columns": cols,
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(),
                    "coef": reg.coef_.tolist(),
                    "intercept": reg.intercept_,
                    "train_rows": len(train),
                }
            )
        for split, y in [("validation", year - 1), ("test", year)]:
            dep = pd.read_csv(archived / split / "deployment.csv").query("eligible")[
                ["a", "b"]
            ]
            start, last = f"{y}-01-01", min(pd.Timestamp(f"{y}-12-31"), calendar[-1])
            available = f[f.date.between(start, last)].merge(dep, on=["a", "b"])
            sample = (
                matched[matched.date.between(start, last)]
                .merge(dep, on=["a", "b"])
                .sort_values(["day", "a", "b"])
            )
            sample.loc[sample.label_end > last, "label"] = np.nan
            tags = {"fold": year, "split": split, "frequency": frequency}
            signal = "daily_residual" if frequency == "daily" else "opening_residual"
            strong = sample.loc[
                sample[signal].abs() >= train[signal].abs().quantile(0.9)
            ]
            coverage.append(
                {
                    **tags,
                    "available": len(available),
                    "matched": len(sample),
                    "scored": sample.label.count(),
                    "strong_count": len(strong),
                    "strong_median_absolute_ratio": (
                        strong["db4_" + signal].abs() / strong[signal].abs()
                    ).median(),
                    "strong_sign_retention": (
                        strong[signal] * strong["db4_" + signal] > 0
                    ).mean(),
                }
            )
            lookup = sample.set_index(["a", "b", "day"]).hedge.to_dict()

            def simulate(pair, d, side, horizon, end, limit, lookup=lookup):
                entry = enter(tapes, pair, d, 61, lookup[(*pair, d)])
                if entry is None:
                    return None
                m, size, w, px = entry
                size = min(size, limit)
                return (
                    *marked_trade(
                        calendar,
                        tapes,
                        pair,
                        d,
                        (m, size, w, px),
                        side,
                        beta[pair],
                        horizon,
                        end,
                    ),
                    size,
                )

            forecasts.append(
                sample[["a", "b", "date", "label_end", "label", *ARMS]].assign(**tags)
            )
            books = {}
            for arm in ARMS:
                perf, curve, _, _, executions = replay(
                    calendar, tapes, sample, arm, 1, start, last, simulate
                )
                metrics.append({**tags, "arm": arm, **accuracy(sample, arm), **perf})
                curves.append(curve.assign(arm=arm, **tags))
                fills.extend({**tags, "arm": arm, **e} for e in executions)
                books[arm] = curve.set_index("date").pnl / CAPITAL
            for control in ("raw", "ma3"):
                diff = books["db4"] - books[control]
                low, high = block_interval(diff)
                gains = (
                    (
                        (sample.label - sample[control]) ** 2
                        - (sample.label - sample.db4) ** 2
                    )
                    .groupby(sample.date)
                    .mean()
                    .dropna()
                )
                comparisons.append(
                    {
                        **tags,
                        "control": control,
                        "net_difference": diff.sum(),
                        "lower": low,
                        "upper": high,
                        "mse_gain": gains.mean(),
                        "mse_p": hac_p(gains),
                        "net_p": hac_p(diff),
                    }
                )
    (output / f"models_{year}.json").write_text(json.dumps(coefficients, indent=2))
    pd.concat(forecasts).to_parquet(output / f"forecasts_{year}.parquet", index=False)
    pd.concat(curves).to_parquet(output / f"equity_{year}.parquet", index=False)
    pd.DataFrame(fills).to_parquet(output / f"fills_{year}.parquet", index=False)
    return metrics, comparisons, coverage


def run(output):
    prior = json.loads((BASE / "manifest.json").read_text())
    for filename, digest in prior["sha256"].items():
        path = Path(filename.replace("\\", "/"))
        if path.suffix in {".json", ".parquet"}:
            with path.open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != digest:
                    raise ValueError(f"Unexpected market snapshot: {path}")
    output.mkdir(parents=True, exist_ok=False)
    calendar = chart("SPY", SNAPSHOT).index
    tapes = {
        s: Tape(s, calendar, SNAPSHOT, STOP, "2021-12-31") for s in prior["cohort"]
    }
    metrics, comparisons, coverage = [], [], []
    for year in (2025, 2026):
        rows, tests, counts = compare_fold(output, year, tapes, calendar)
        metrics.extend(rows)
        comparisons.extend(tests)
        coverage.extend(counts)
    result = pd.DataFrame(comparisons)
    for indices in result.groupby("split").groups.values():
        p = result.loc[indices, ["mse_p", "net_p"]].to_numpy()
        result.loc[indices, ["mse_holm", "net_holm"]] = holm(p.ravel()).reshape(p.shape)
    result.to_csv(output / "paired.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    inputs = [
        Path(__file__),
        BASE / "manifest.json",
        Path("experiments/forecasting/strict_alpha_study.py"),
        Path("experiments/forecasting/strict_alpha_evidence.py"),
        Path("data/wavelet.py"),
        *BASE.glob("20*/qualification.csv"),
        *BASE.glob("20*/*/deployment.csv"),
    ]
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "wavelet": "db4",
                "level": 1,
                "mode": "symmetric",
                "threshold": "MAD universal soft",
                "window_sessions": 64,
                "opening_minutes": 32,
                "holding_sessions": 1,
                "ridge_alpha": 100,
                "pywavelets": pywt.__version__,
                "source_sha256": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
