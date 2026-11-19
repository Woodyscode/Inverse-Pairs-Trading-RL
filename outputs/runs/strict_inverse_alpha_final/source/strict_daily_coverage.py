"""Forecast audit across the archived point-in-time stock and cross-asset ETF screens."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from data.historical_panel import CACHE, FOLDS, load_market
from inverse_pair_discovery import COHORT, RHO, holm, load_etfs, ols_slope, pair_data
from strict_alpha_study import SEED, accuracy

FEATURES = [
    "residual",
    "z",
    "momentum",
    "vol",
    "ref_vol",
    "relative_volume",
    "interaction",
    "beta",
]
SCREEN = Path("outputs/runs/inverse_discovery_universe/training_diagnostics.csv")


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    dates = pd.read_csv(
        FOLDS,
        parse_dates=[
            "train_start",
            "train_end",
            "validation_start",
            "test_start",
            "test_end",
        ],
    )
    stocks, dollars, active = load_market(CACHE, dates)
    cohort = yaml.safe_load(COHORT.read_text())
    funds = set(cohort["sectors"]) | {s for g in cohort["families"].values() for s in g}
    ep, ed, _ = load_etfs(stocks.index, funds)
    px = pd.concat([stocks.drop(columns=list(funds), errors="ignore"), ep], axis=1)
    dv = pd.concat([dollars.drop(columns=list(funds), errors="ignore"), ed], axis=1)
    members = active.drop(columns=list(funds), errors="ignore")
    screen = pd.read_csv(SCREEN)
    forecasts, coverage = [], []
    for fold in dates.itertuples():
        frames = []
        hist = screen[
            (screen.window == fold.window)
            & (screen.rho <= RHO)
            & (screen.worst_train_rho <= RHO)
        ]
        for row in hist.itertuples():
            price, dol, ok = pair_data(
                px, dv, members, fold, row.a, row.b, fold.test_end
            )
            ret = price.pct_change(fill_method=None)
            idx = ret.loc[: fold.train_end].index
            cutoff = idx[2 * len(idx) // 3]
            fit = ret.loc[:cutoff].iloc[1:]
            rho = fit.iloc[:, 0].corr(fit.iloc[:, 1])
            h = ols_slope(fit.iloc[:, 0].to_numpy(), fit.iloc[:, 1].to_numpy())
            if not (rho <= RHO and h < 0):
                continue
            r = ret.iloc[:, 0] - h * ret.iloc[:, 1]
            residual = (r - r.loc[fit.index].mean()) / (1 - h)
            z = (residual - residual.rolling(30).mean()) / residual.rolling(30).std()
            refvol = ret.iloc[:, 1].rolling(30).std()
            f = pd.DataFrame(
                dict(
                    residual=residual,
                    z=z,
                    momentum=residual.rolling(5).sum(),
                    vol=residual.rolling(30).std(),
                    ref_vol=refvol,
                    relative_volume=(dol / dol.shift().rolling(20).mean()).min(
                        axis=1, skipna=False
                    ),
                    interaction=z * refvol,
                    beta=h,
                )
            )
            for horizon in [1, 5, 10]:
                # Observe today's close, trade next close, mark fixed units h sessions later.
                future = price.shift(-1 - horizon) / price.shift(-1) - 1
                f[f"y{horizon}"] = (future.iloc[:, 0] - h * future.iloc[:, 1]) / (1 - h)
                f[f"end{horizon}"] = f.index.to_series().shift(-1 - horizon)
            valret = ret.loc[fold.validation_start : fold.test_start].iloc[:-1]
            valrho = (
                valret.iloc[:, 0].corr(valret.iloc[:, 1])
                if valret.notna().all().all()
                else np.nan
            )
            eligible = f.index.to_series().gt(cutoff) & ok
            eligible &= (f.index < fold.test_start) | (valrho <= RHO)
            frames.append(
                f.loc[eligible].assign(
                    a=row.a, b=row.b, category=row.category, day=f.index[eligible]
                )
            )
            coverage.append(
                dict(
                    window=fold.window,
                    category=row.category,
                    a=row.a,
                    b=row.b,
                    hedge_cutoff=cutoff,
                    calibration_rho=rho,
                    train_rho=row.rho,
                    validation_rho=valrho,
                    beta=h,
                    test_eligible=valrho <= RHO,
                )
            )
        data = (
            pd.concat(frames).replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
        )
        model_dir = out / str(fold.window)
        model_dir.mkdir()
        for h in (1, 5, 10):
            d = data.rename(columns={f"y{h}": "label", f"end{h}": "end"})
            tr = d[(d.end <= fold.train_end) & d.label.notna()]
            te = d[d.day >= fold.validation_start].copy()
            te["split"] = np.where(te.day < fold.test_start, "validation", "test")
            te.loc[
                (te.split == "validation") & (te.end >= fold.test_start), "label"
            ] = np.nan
            te["zero"] = 0.0
            scale = tr.label.abs().median() / tr.z.abs().median()
            te["continuation"] = te.z.clip(-5, 5) * scale
            te["reversal"] = -te.continuation
            for name, model in [
                ("linear", LinearRegression()),
                ("ridge", Ridge(alpha=100)),
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
                fitted = make_pipeline(StandardScaler(), model).fit(
                    tr[FEATURES], tr.label
                )
                te[name] = fitted.predict(te[FEATURES])
                joblib.dump(fitted, model_dir / f"{h}_{name}.joblib")
            forecasts.append(
                te[
                    [
                        "day",
                        "end",
                        "a",
                        "b",
                        "category",
                        "split",
                        "label",
                        "zero",
                        "continuation",
                        "reversal",
                        "linear",
                        "ridge",
                        "xgboost",
                    ]
                ].assign(window=fold.window, horizon=h)
            )
        print("daily coverage window", fold.window, flush=True)
    frame = pd.concat(forecasts, ignore_index=True)
    frame.to_parquet(out / "forecasts.parquet", index=False)
    pd.DataFrame(coverage).to_csv(out / "pairs.csv", index=False)
    rows = []
    for (split, h, group), part in frame.groupby(["split", "horizon", "category"]):
        for key in ["zero", "continuation", "reversal", "linear", "ridge", "xgboost"]:
            rows.append(
                dict(
                    split=split,
                    horizon=h,
                    category=group,
                    model=key,
                    **accuracy(part, key),
                )
            )
    metrics = pd.DataFrame(rows)
    metrics["mse_holm"] = metrics.groupby("split").mse_p.transform(holm)
    metrics.to_csv(out / "metrics.csv", index=False)
    inputs = [
        Path(__file__),
        Path("strict_alpha_study.py"),
        FOLDS,
        SCREEN,
        COHORT,
        Path("outputs/runs/inverse_discovery_universe/manifest.json"),
    ]
    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                features=FEATURES,
                seed=SEED,
                source_sha256={
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
                },
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
