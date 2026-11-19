"""Strict inverse-pair discovery. Freeze validation rankings before opening each test fold."""

import argparse
import hashlib
import json
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint

from backtest.inverse_discovery_report import write_report
from data.fetch_inverse_etfs import COHORT, EXTRA, LOCAL
from data.historical_panel import CACHE, FOLDS, load_market

RHO = -0.80
MIN_ADV = 10_000_000
CAPITAL, TARGET = 100_000.0, 0.25
FEE, FUNDING, BORROW, ADV_LIMIT = 0.0002, 0.02, 0.005, 0.01
HORIZONS = (1, 5, 10)
CATEGORIES = ("inverse_etf", "cross_asset", "within_macro_class", "stock_sector_hedge")


def load_etfs(calendar, tickers):
    fields = {
        f"{field}_{s}" for s in tickers for field in ("Adj_Close", "Close", "Volume")
    } | {"Date"}
    raw = pd.read_csv(
        LOCAL,
        usecols=lambda c: c in fields,
        index_col="Date",
        parse_dates=True,
    ).reindex(calendar)
    prices, dollars, coverage = {}, {}, []
    for s in sorted(tickers):
        if f"Adj_Close_{s}" in raw:
            p = raw[f"Adj_Close_{s}"]
            d = raw[f"Close_{s}"] * raw[f"Volume_{s}"]
            source = "original_yahoo_snapshot"
        else:
            path = EXTRA / f"{s}.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"Fetch the ETF cohort first: python -m data.fetch_inverse_etfs ({s})"
                )
            record = (json.loads(path.read_text())["result"] or [{}])[0]
            source = "supplemental_yahoo_snapshot"
            p = d = pd.Series(np.nan, index=calendar)
            if record.get("timestamp"):
                ix = (
                    pd.to_datetime(record["timestamp"], unit="s", utc=True)
                    .tz_convert("America/New_York")
                    .normalize()
                    .tz_localize(None)
                )
                q = pd.DataFrame(record["indicators"]["quote"][0], index=ix)
                p = pd.Series(
                    record["indicators"]["adjclose"][0]["adjclose"], index=ix
                ).reindex(calendar)
                d = (q.close * q.volume).reindex(calendar)
        prices[s], dollars[s] = p.where(p > 0), d.where(d >= 0)
        coverage.append(
            {
                "ticker": s,
                "source": source,
                "observations": p.count(),
                "first": p.first_valid_index(),
                "last": p.last_valid_index(),
            }
        )
    return pd.DataFrame(prices), pd.DataFrame(dollars), pd.DataFrame(coverage)


def pair_universe(cohort, stocks):
    family = {s: f for f, product in cohort["families"].items() for s in product}
    multiple = {
        s: m for product in cohort["families"].values() for s, m in product.items()
    }
    classes = {
        f: k for k, families in cohort["asset_classes"].items() for f in families
    }
    if set(classes) != set(cohort["families"]):
        raise ValueError("Each ETF family needs an asset class")
    equity = set(cohort["asset_classes"]["equity"])
    for a, b in combinations(sorted(family), 2):
        if family[a] == family[b] and multiple[a] * multiple[b] < 0:
            # Put the inverse product first so the slope is per unit of the reference.
            inverse, reference = (a, b) if multiple[a] < 0 else (b, a)
            yield (
                "inverse_etf",
                inverse,
                reference,
                multiple[inverse] / multiple[reference],
            )
        elif family[a] != family[b] and not (
            family[a] in equity and family[b] in equity
        ):
            category = (
                "cross_asset"
                if classes[family[a]] != classes[family[b]]
                else "within_macro_class"
            )
            yield category, a, b, np.nan
    hedges = sorted(s for s in family if multiple[s] < 0 or family[s] not in equity)
    for a in sorted(set(stocks) | set(cohort["sectors"])):
        for b in hedges:
            if a != b:
                yield "stock_sector_hedge", a, b, np.nan


def ols_slope(y, x):
    centered = x - x.mean()
    return (
        np.dot(y - y.mean(), centered) / np.dot(centered, centered)
        if centered.std() > 1e-12
        else np.nan
    )


def signed_weights(beta):
    if not np.isfinite(beta) or beta >= 0:
        raise ValueError("An inverse hedge requires a negative finite slope")
    return np.array([1.0, -beta]) / (1 + abs(beta))


def holm(values):
    return (
        multipletests(
            np.nan_to_num(np.asarray(values, dtype=float), nan=1.0), method="holm"
        )[1]
        if len(values)
        else []
    )


def signals(prices, mode, beta, intercept):
    if mode == "tracking":
        r = prices.pct_change(fill_method=None)
        residual = r.iloc[:, 0] - intercept - beta * r.iloc[:, 1]
    else:
        log = np.log(prices)
        residual = log.iloc[:, 0] - intercept - beta * log.iloc[:, 1]
    # Previous 30 observations define normal behavior before today's dislocation.
    prior = residual.shift(1).rolling(30)
    return -(residual - prior.mean()) / prior.std().where(lambda x: x > 1e-10)


def predictive_p(score, realized, h):
    known = np.isfinite(score) & np.isfinite(realized)
    x, y = score[known], realized[known]
    if len(x) < 30 or x.std() < 1e-9 or y.std() < 1e-12:
        return np.nan, 1.0
    fit = sm.OLS(y, sm.add_constant(x)).fit(
        cov_type="HAC", cov_kwds={"maxlags": max(10, h)}
    )
    return float(fit.params[1]), float(norm.sf(fit.tvalues[1])) if np.isfinite(
        fit.tvalues[1]
    ) else 1.0


def evaluate(prices, dollars, active, score, beta, start, end, h):
    """Hold fixed shares between next-close entry and horizon exit. Costs accrue on both legs."""
    w = signed_weights(beta)
    px = prices.to_numpy()
    budget = (
        ADV_LIMIT
        * np.minimum(dollars.rolling(20).mean().shift(1), dollars).fillna(0).to_numpy()
    )
    flag = active.to_numpy()
    s = np.arange(start - 1, end - h - 1)
    forecast = score.to_numpy()
    realized = (px[s + 1 + h] / px[s + 1] - 1) @ w
    slope, p = predictive_p(np.where(flag[s], forecast[s], np.nan), realized, h)
    capital = CAPITAL
    fees = carry = gross = 0.0
    trades = []
    next_entry = start
    for i in s:
        entry, exit_day = i + 1, i + 1 + h
        if (
            entry < next_entry
            or not flag[i]
            or not flag[entry]
            or not np.isfinite(forecast[i])
            or abs(forecast[i]) < 1
        ):
            continue
        side = np.sign(forecast[i])
        notional = capital * TARGET
        if np.any(notional * w > budget[entry]) or not (px[entry] > 0).all():
            continue
        marks = px[entry : exit_day + 1]
        if not (np.isfinite(marks) & (marks > 0)).all():
            return {
                "valid": False,
                "invalid_reason": "missing_held_mark",
                "p": 1.0,
            }, trades
        positions = side * notional * w * marks / marks[0]
        if np.any(np.abs(positions[-1]) > budget[exit_day]):
            return {
                "valid": False,
                "invalid_reason": "exit_exceeds_volume_cap",
                "p": 1.0,
            }, trades
        pnl = (positions[-1] - positions[0]).sum()
        leg_costs = FEE * (np.abs(positions[0]) + np.abs(positions[-1]))
        cost = leg_costs.sum()
        ages = (
            np.diff(prices.index[entry : exit_day + 1].to_numpy())
            .astype("timedelta64[D]")
            .astype(int)
            / 365
        )
        funding = FUNDING * (ages @ np.abs(positions[:-1]).sum(axis=1))
        borrow = BORROW * (ages @ np.maximum(-positions[:-1], 0).sum(axis=1))
        finance = funding + borrow
        capital += pnl - cost - finance
        trades.append(
            {
                "signal": prices.index[i],
                "entry": prices.index[entry],
                "exit": prices.index[exit_day],
                "side": int(side),
                "entry_gross": notional,
                "entry_a": positions[0, 0],
                "entry_b": positions[0, 1],
                "exit_a": positions[-1, 0],
                "exit_b": positions[-1, 1],
                "execution_a": leg_costs[0],
                "execution_b": leg_costs[1],
                "funding": funding,
                "borrow": borrow,
                "gross_pnl": pnl,
                "execution": cost,
                "carry": finance,
                "equity": capital,
            }
        )
        gross += pnl
        fees += cost
        carry += finance
        next_entry = exit_day + 1
    return {
        "valid": True,
        "invalid_reason": "",
        "p": p,
        "slope": slope,
        "trades": len(trades),
        "net": capital / CAPITAL - 1,
        "gross_pnl": gross,
        "execution": fees,
        "carry": carry,
    }, trades


def prepare_fold(prices, dollars, active, cohort, fold):
    train = prices.loc[fold.train_start : fold.train_end]
    dv = dollars.loc[train.index]
    stock_members = active.columns[active.loc[fold.train_end]].difference(
        ["SPY", *cohort["sectors"]]
    )
    candidates = pd.DataFrame(
        pair_universe(cohort, stock_members),
        columns=["category", "a", "b", "design_ratio"],
    )
    needed = sorted(set(candidates.a) | set(candidates.b))
    train = train.reindex(columns=needed)
    dv = dv.reindex(columns=needed)
    history = (train.gt(0) & dv.notna()).all()
    adv = dv.tail(60).median()
    ret = train.pct_change(fill_method=None)
    # Only requested pairs enter the search. Incomplete history remains a recorded failure.
    corr = ret.corr()
    matrix = corr.to_numpy()
    candidates["rho"] = matrix[
        corr.index.get_indexer(candidates.a), corr.index.get_indexer(candidates.b)
    ]
    candidates["min_adv"] = np.minimum(candidates.a.map(adv), candidates.b.map(adv))
    ok_history = candidates.a.map(history) & candidates.b.map(history)
    candidates["status"] = np.select(
        [
            ~ok_history,
            candidates.rho.isna() | candidates.rho.gt(RHO),
            candidates.min_adv.lt(MIN_ADV),
        ],
        ["history", "correlation", "liquidity"],
        default="screened",
    )
    diagnostics = []
    for rec in candidates.loc[candidates.status.eq("screened")].itertuples():
        a, b = rec.a, rec.b
        ra, rb = ret[[a, b]].iloc[1:].to_numpy().T
        la, lb = np.log(train[[a, b]]).to_numpy().T
        rt = ols_slope(ra, rb)
        lp = ols_slope(la, lb)
        intercept = ra.mean() - rt * rb.mean()
        thirds = np.array_split(np.arange(len(ra)), 3)
        worst = np.max([np.corrcoef(ra[idx], rb[idx])[0, 1] for idx in thirds])
        w = signed_weights(rt)
        spy = (
            prices.SPY.loc[train.index].pct_change(fill_method=None).iloc[1:].to_numpy()
        )
        ep = coint(la, lb, maxlag=1, autolag=None)[1]
        spread = la - lp * lb
        phi = ols_slope(spread[1:], spread[:-1])
        ten_day = np.expm1(la[10:] - la[:-10]) - rec.design_ratio * np.expm1(
            lb[10:] - lb[:-10]
        )
        diagnostics.append(
            {
                "a": a,
                "b": b,
                "category": rec.category,
                "rho": rec.rho,
                "worst_train_rho": worst,
                "return_beta": rt,
                "return_intercept": intercept,
                "level_beta": lp,
                "largest_subwindow_level_beta": max(
                    ols_slope(la[idx], lb[idx])
                    for idx in np.array_split(np.arange(len(la)), 3)
                ),
                "spread_half_life_sessions": -np.log(2) / np.log(phi)
                if 0 < phi < 1
                else np.nan,
                "level_intercept": la.mean() - lp * lb.mean(),
                "eg_p": ep,
                "ten_day_static_multiplier_gap_rmse_bp": np.sqrt(np.mean(ten_day**2))
                * 10000,
                "tracking_error_bp": np.std(ra - intercept - rt * rb) * 10000,
                "designed_daily_rmse_bp": np.sqrt(
                    np.mean((ra - rec.design_ratio * rb) ** 2)
                )
                * 10000
                if np.isfinite(rec.design_ratio)
                else np.nan,
                "train_spy_beta_long": ols_slope(w[0] * ra + w[1] * rb, spy),
            }
        )
    diag = pd.DataFrame(diagnostics)
    if len(diag):
        diag["eg_holm"] = holm(diag.eg_p)
    return candidates, diag


def pair_data(prices, dollars, active, fold, a, b, last):
    px = prices.loc[fold.train_start : last, [a, b]]
    dv = dollars.loc[px.index, [a, b]]
    ok = (
        active.loc[px.index, a]
        if a in active and a not in ("SPY",)
        else pd.Series(True, index=px.index)
    )
    return px, dv, ok


def validate(prices, dollars, active, fold, diag):
    rows = []
    # This function receives no bars from the final test.
    for rec in diag.itertuples():
        px, dv, ok = pair_data(
            prices, dollars, active, fold, rec.a, rec.b, prices.index[-1]
        )
        start = px.index.get_loc(fold.validation_start)
        vr = px.iloc[start - 1 :].pct_change(fill_method=None).iloc[1:]
        val_rho = (
            vr.iloc[:, 0].corr(vr.iloc[:, 1]) if vr.notna().all().all() else np.nan
        )
        for mode in ("tracking", "stationary_spread"):
            beta = rec.return_beta if mode == "tracking" else rec.level_beta
            intercept = (
                rec.return_intercept if mode == "tracking" else rec.level_intercept
            )
            reason = ""
            if not rec.worst_train_rho <= RHO:
                reason = "unstable_training_correlation"
            elif not np.isfinite(beta) or beta >= 0:
                reason = "noninverse_hedge"
            elif mode == "stationary_spread" and rec.eg_holm >= 0.05:
                reason = "stationarity"
            capacity = np.nan
            wa = wb = spy_beta = np.nan
            if np.isfinite(beta) and beta < 0:
                w = signed_weights(beta)
                wa, wb = w
                capacity = (
                    dv.iloc[:start].tail(60).median().to_numpy() * ADV_LIMIT / w
                ).min() / TARGET
                tr = (
                    px.iloc[:start].pct_change(fill_method=None).iloc[1:].to_numpy() @ w
                )
                spy_beta = ols_slope(
                    tr,
                    prices.SPY.loc[px.index[:start]]
                    .pct_change(fill_method=None)
                    .iloc[1:]
                    .to_numpy(),
                )
                if not reason and capacity < CAPITAL:
                    reason = "capacity"
            if not reason:
                score = signals(px, mode, beta, intercept)
            for h in HORIZONS:
                record = {
                    "category": rec.category,
                    "a": rec.a,
                    "b": rec.b,
                    "mode": mode,
                    "horizon": h,
                    "beta": beta,
                    "intercept": intercept,
                    "weight_a": wa,
                    "weight_b": wb,
                    "spy_beta_long": spy_beta,
                    "capacity": capacity,
                    "val_rho": val_rho,
                    "reason": reason,
                }
                if not reason:
                    result, _ = evaluate(px, dv, ok, score, beta, start, len(px), h)
                    record.update(result)
                rows.append(record)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    has_test = frame.reason.eq("")
    frame["validation_holm"] = np.nan
    if has_test.any():
        frame.loc[has_test, "validation_holm"] = holm(frame.loc[has_test, "p"])
        for idx, row in frame[has_test].iterrows():
            checks = [
                (not row.valid, row.invalid_reason),
                (not row.val_rho <= RHO, "validation_correlation"),
                (row.get("trades", 0) < 8, "fewer_than_8_trades"),
                (not row.get("net", np.nan) > 0, "net_after_costs"),
                (not row.validation_holm < 0.05, "predictive_evidence"),
            ]
            frame.at[idx, "reason"] = (
                "|".join(label for failed, label in checks if failed) or "qualified"
            )
    ranked = frame[frame.reason.eq("qualified")].sort_values(
        ["category", "net", "val_rho", "a", "b", "mode", "horizon"],
        ascending=[True, False, True, True, True, True, True],
    )
    frame["rank"] = ranked.groupby("category").cumcount() + 1
    return frame


def run(output):
    output.mkdir(parents=True, exist_ok=False)
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
    stocks, stock_dv, active = load_market(CACHE, dates)
    cohort = yaml.safe_load(COHORT.read_text())
    funds = set(cohort["sectors"]) | {s for g in cohort["families"].values() for s in g}
    active = active.drop(columns=list(funds), errors="ignore")
    ep, ed, coverage = load_etfs(stocks.index, funds)
    classes = {
        ticker: kind
        for kind, families in cohort["asset_classes"].items()
        for family in families
        for ticker in cohort["families"][family]
    }
    coverage["asset_class"] = coverage.ticker.map(classes).fillna("equity")
    prices = pd.concat([stocks.drop(columns=list(funds), errors="ignore"), ep], axis=1)
    dollars = pd.concat(
        [stock_dv.drop(columns=list(funds), errors="ignore"), ed], axis=1
    )
    # These reference funds were documented before the first fold. Require observed quotes on every training date.
    if pd.Timestamp(cohort["as_of"]) >= dates.train_start.min():
        raise ValueError("ETF cohort must predate the study")
    screened, diagnostics, hypotheses, decisions, ledger = [], [], [], [], []
    for fold in dates.itertuples():
        search, diag = prepare_fold(prices, dollars, active, cohort, fold)
        screened.append(search.assign(window=fold.window))
        diagnostics.append(diag.assign(window=fold.window))
        val = validate(
            prices.loc[: fold.test_start].iloc[:-1], dollars, active, fold, diag
        )
        hypotheses.append(val.assign(window=fold.window))
        for category in CATEGORIES:
            leader = (
                val[val.category.eq(category) & val["rank"].eq(1)] if len(val) else val
            )
            row = {
                "window": fold.window,
                "category": category,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
            }
            if leader.empty:
                decisions.append(dict(**row, status="cash", net=0.0))
                continue
            model = leader.iloc[0]
            px, dv, ok = pair_data(
                prices, dollars, active, fold, model.a, model.b, fold.test_end
            )
            score = signals(px, model["mode"], model.beta, model.intercept)
            start = px.index.get_loc(fold.test_start)
            test, deals = evaluate(
                px, dv, ok, score, model.beta, start, len(px), int(model.horizon)
            )
            test_ret = px.iloc[start - 1 :].pct_change(fill_method=None).iloc[1:]
            decisions.append(
                dict(
                    **row,
                    a=model.a,
                    b=model.b,
                    mode=model["mode"],
                    horizon=model.horizon,
                    test_rho=test_ret.iloc[:, 0].corr(test_ret.iloc[:, 1]),
                    status="evaluated" if test["valid"] else "unscored",
                    **test,
                )
            )
            ledger.extend(dict(**row, a=model.a, b=model.b, **deal) for deal in deals)
        print(f"Fold {fold.window}: {len(diag)} screened pairs", flush=True)
    frames = {
        "candidate_screen.csv.gz": pd.concat(screened),
        "training_diagnostics.csv": pd.concat(diagnostics),
        "validation_rankings.csv": pd.concat(hypotheses),
        "test_decisions.csv": pd.DataFrame(decisions),
        "test_trades.csv": pd.DataFrame(
            ledger,
            columns=[
                "window",
                "category",
                "test_start",
                "test_end",
                "a",
                "b",
                "signal",
                "entry",
                "exit",
                "side",
                "entry_gross",
                "entry_a",
                "entry_b",
                "exit_a",
                "exit_b",
                "execution_a",
                "execution_b",
                "funding",
                "borrow",
                "gross_pnl",
                "execution",
                "carry",
                "equity",
            ],
        ),
        "etf_coverage.csv": coverage,
    }
    for frame in frames.values():
        if "a" in frame:
            frame["asset_a"] = (
                frame.a.map(classes).fillna("equity").where(frame.a.notna())
            )
            frame["asset_b"] = (
                frame.b.map(classes).fillna("equity").where(frame.b.notna())
            )
    for name, frame in frames.items():
        frame.to_csv(output / name, index=False)
    write_report(output, frames)
    paths = [
        Path(__file__),
        COHORT,
        LOCAL,
        FOLDS,
        Path("data/historical_panel.py"),
        Path("data/fetch_inverse_etfs.py"),
        Path("backtest/inverse_discovery_report.py"),
        *sorted(EXTRA.glob("*.json")),
        *(
            CACHE / n
            for n in (
                "prices.parquet",
                "membership_intervals.parquet",
                "no_trade_days.csv",
            )
        ),
    ]
    hashes = {}
    for path in paths:
        with path.open("rb") as f:
            hashes[str(path)] = hashlib.file_digest(f, "sha256").hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "sha256": hashes,
                "dependencies": {
                    k: version(k)
                    for k in (
                        "numpy",
                        "pandas",
                        "scipy",
                        "statsmodels",
                        "pyarrow",
                        "curl_cffi",
                    )
                },
                "etf_sources": cohort["sources"],
                "correlation_threshold": RHO,
                "minimum_adv": MIN_ADV,
                "capital": CAPITAL,
                "target": TARGET,
                "fee": FEE,
                "annual_funding": FUNDING,
                "annual_borrow": BORROW,
                "participation": ADV_LIMIT,
                "horizons": HORIZONS,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
