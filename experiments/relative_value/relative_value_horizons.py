"""Walk-forward Ridge forecasts of executable stock-pair and market-hedged returns."""

import argparse
import hashlib
import json
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint

from data.historical_panel import CACHE, FOLDS, load_market
from data.minute_features import minute_pair_features, morning_moments

HORIZONS = (1, 5, 10)
FEE, FUNDING, BORROW, PARTICIPATION = 0.0002, 0.02, 0.005, 0.01
CAPITAL, GROSS = 100_000.0, 0.25


def select_pairs(px, dollars, members):
    eligible = members.index[members].difference(["SPY"])
    available = eligible.intersection(px.columns)
    good = (px[available].gt(0) & dollars[available].gt(0)).all()
    adv = dollars[good.index[good]].tail(60).median()
    leaders = (
        adv[adv >= 10_000_000].sort_values(ascending=False, kind="stable").head(30)
    )
    tests = []
    for a, b in combinations(sorted(leaders.index), 2):
        # Fix the regression orientation and lag order before looking at the p-value.
        x, y = np.log(px[[a, b]]).to_numpy().T
        tests.append((a, b, coint(x, y, maxlag=1, autolag=None)[1]))
    tests = pd.DataFrame(tests, columns=["a", "b", "p"])
    chosen = None
    if len(tests):
        tests["holm_p"] = multipletests(tests.p.fillna(1), method="holm")[1]
        passed = tests[tests.holm_p < 0.05].sort_values(["holm_p", "a", "b"])
        if len(passed):
            chosen = tuple(passed.iloc[0][["a", "b"]])
    pairs = {
        "market_hedged": (leaders.index[0], "SPY") if len(leaders) else None,
        "cointegrated": chosen,
    }
    counts = {
        "members": len(eligible),
        "complete_history": int(good.sum()),
        "liquid": int((adv >= 10_000_000).sum()),
        "screened_stocks": len(leaders),
        "pairs_tested": len(tests),
        "pairs_passed": int(tests.get("holm_p", pd.Series(dtype=float)).lt(0.05).sum()),
    }
    return pairs, tests, counts


def spread_features(px, construction):
    a, b = (np.log(px[col]) for col in px)
    x, y = (a.diff(), b.diff()) if construction == "market_hedged" else (a, b)
    beta = x.rolling(120).cov(y) / y.rolling(120).var()
    scale = 1 + beta.abs()
    var = (
        a.rolling(30).var()
        + beta**2 * b.rolling(30).var()
        - 2 * beta * a.rolling(30).cov(b)
    )
    z = ((a - a.rolling(30).mean()) - beta * (b - b.rolling(30).mean())) / np.sqrt(
        var.clip(lower=1e-12)
    )
    state = pd.DataFrame(
        {
            "z": z,
            "beta": beta,
            "momentum5": (a.diff(5) - beta * b.diff(5)) / scale,
            "momentum20": (a.diff(20) - beta * b.diff(20)) / scale,
            "spread_vol": np.sqrt(var.clip(lower=0)) / scale,
        }
    )
    return state.replace([np.inf, -np.inf], np.nan)


def basket_label(px, beta, h):
    # Trade fixed entry quantities at s+1. Mark both total-return legs at s+1+h.
    r = px.shift(-1 - h) / px.shift(-1) - 1
    return (r.iloc[:, 0] - beta * r.iloc[:, 1]) / (1 + beta.abs())


def fit_predict(features, target, train_n, h, eligible):
    known = eligible & features.notna().all(axis=1) & target.notna()
    known.iloc[train_n - 1 - h :] = False  # last permitted exit is train_n-1
    model = make_pipeline(StandardScaler(), Ridge(alpha=10))
    if known.sum() < 126:
        return None
    model.fit(features[known], target[known])
    valid = eligible & features.notna().all(axis=1)
    pred = pd.Series(np.nan, index=features.index)
    pred.loc[valid] = model.predict(features[valid])
    return pred


def simulate(px, dv, membership, beta, pred, start, end, h, capital):
    """Fixed units between fills. An exit over the volume cap invalidates the run."""
    adv = dv.rolling(20).mean().shift(1)
    rows, trades = [], []
    cash = capital
    units = np.zeros(2)
    entry = None
    for t in range(start, end):
        date = px.index[t]
        held = units.any()
        prices = px.iloc[t].to_numpy()
        costs = carry = gross_pnl = turnover = 0.0
        if held:
            if not np.all(np.isfinite(prices) & (prices > 0)):
                raise ValueError(f"Missing held mark on {date} for {list(px)}")
            previous = px.iloc[t - 1].to_numpy()
            marked = units * previous
            carry = (
                (np.abs(marked).sum() * FUNDING + np.maximum(-marked, 0).sum() * BORROW)
                * (date - px.index[t - 1]).days
                / 365
            )
            gross_pnl = units @ (prices - previous)
            cash += gross_pnl - carry
        budget = (
            np.minimum(adv.iloc[t], dv.iloc[t]).fillna(0).to_numpy() * PARTICIPATION
        )
        if held and t == entry + h:
            exit_values = units * prices
            if (np.abs(exit_values) > budget + 1e-8).any():
                raise ValueError(f"Unfillable scheduled exit {date}, {list(px)}")
            turnover = np.abs(exit_values).sum()
            units[:] = 0
            move = basket_move(px.iloc[entry].to_numpy(), prices, beta.iloc[entry - 1])
            trades[-1].update(exit=date, gross_basket_return=trades[-1]["side"] * move)
        elif (
            (t - start) % (h + 1) == 0
            and t + h < end
            and membership.iloc[t - 1 : t + 1].all()
            and np.isfinite(pred.iloc[t - 1])
        ):
            hedge = beta.iloc[t - 1]
            w = np.array([1.0, -hedge]) / (1 + abs(hedge))
            side = np.sign(pred.iloc[t - 1])
            hurdle = (
                2 * FEE + (FUNDING + np.maximum(-side * w, 0).sum() * BORROW) * h / 252
            )
            if abs(pred.iloc[t - 1]) > hurdle and np.all(
                np.isfinite(prices) & (prices > 0)
            ):
                desired = cash * GROSS
                allowed = np.min(budget / np.maximum(np.abs(w), 1e-15))
                size = min(desired, allowed)
                if size > 0:
                    units = side * w * size / prices
                    entry = t
                    turnover = size
                    trades.append(
                        {
                            "signal": px.index[t - 1],
                            "entry": date,
                            "side": side,
                            "forecast": pred.iloc[t - 1],
                            "hurdle": hurdle,
                            "entry_notional": size,
                        }
                    )
        costs = turnover * FEE
        cash -= costs
        rows.append(
            {
                "date": date,
                "equity": cash,
                "gross_pnl": gross_pnl,
                "execution": costs,
                "carry": carry,
                "turnover": turnover,
            }
        )
    if units.any():
        raise AssertionError("Open inventory at fold end")
    return pd.DataFrame(rows), trades


def basket_move(entry, exit_prices, hedge):
    return np.dot(exit_prices / entry - 1, [1.0, -hedge]) / (1 + abs(hedge))


def main(args):
    out = args.output
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
    prices, dollars, active = load_market(args.cache, dates)
    ledgers, selections, tests, forecasts, trades, coverage = [], [], [], [], [], []
    balances = {}
    for fold in dates.itertuples():
        px = prices.loc[fold.train_start : fold.test_end]
        dv = dollars.loc[px.index]
        n = px.index.get_loc(fold.train_end) + 1
        v = px.index.get_loc(fold.test_start)
        choices, table, counts = select_pairs(
            px.iloc[:n], dv.iloc[:n], active.loc[fold.train_end]
        )
        tests.append(table.assign(window=fold.window))
        for construction, pair in choices.items():
            selections.append(
                dict(
                    window=fold.window,
                    construction=construction,
                    pair="/".join(pair) if pair else None,
                    **counts,
                )
            )
            if pair:
                cols = list(pair)
                instruments = px[cols]
                state = spread_features(instruments, construction)
                member = active.loc[px.index, cols].all(axis=1)
            variants = ["daily"]
            if args.minute:
                if pair:
                    minute = minute_pair_features(
                        args.cache, pair, px.index, state.beta
                    )
                variants.extend(["matched_daily", "minute_augmented"])
            for variant in variants:
                if pair:
                    features = (
                        state.join(minute) if variant == "minute_augmented" else state
                    )
                    eligible = member.copy()
                    if variant != "daily":
                        eligible &= minute.notna().all(axis=1)
                for h in HORIZONS:
                    pred = None
                    if pair:
                        target = basket_label(instruments, state.beta, h)
                        pred = fit_predict(features, target, n, h, eligible)
                    for split, start, end in [
                        ("validation", n, v),
                        ("test", v, len(px)),
                    ]:
                        key = (construction, variant, h)
                        cap = balances.get(key, CAPITAL) if split == "test" else CAPITAL
                        meta = {
                            "construction": construction,
                            "variant": variant,
                            "horizon": h,
                            "window": fold.window,
                            "split": split,
                        }
                        if pred is None:
                            bars = pd.DataFrame(
                                {
                                    "date": px.index[start:end],
                                    "equity": cap,
                                    "gross_pnl": 0.0,
                                    "execution": 0.0,
                                    "carry": 0.0,
                                    "turnover": 0.0,
                                }
                            )
                            deals = []
                            nscore = 0
                        else:
                            bars, deals = simulate(
                                instruments,
                                dv[cols],
                                member,
                                state.beta,
                                pred,
                                start,
                                end,
                                h,
                                cap,
                            )
                            signals = np.arange(start - 1, end - h - 1)
                            scored = pd.DataFrame(
                                {
                                    "signal": px.index[signals],
                                    "entry": px.index[signals + 1],
                                    "exit": px.index[signals + 1 + h],
                                    "forecast": pred.iloc[signals].to_numpy(),
                                    "realized": target.iloc[signals].to_numpy(),
                                }
                            ).dropna()
                            forecasts.append(scored.assign(pair="/".join(pair), **meta))
                            nscore = len(scored)
                        coverage.append(
                            dict(
                                **meta,
                                forecast_days=nscore,
                                possible_days=end - start - h,
                                fitted=pred is not None,
                                trades=len(deals),
                            )
                        )
                        trades.extend(
                            dict(**d, pair="/".join(pair), **meta) for d in deals
                        )
                        bars["net_return"] = (
                            bars.equity / bars.equity.shift(1, fill_value=cap) - 1
                        )
                        ledgers.append(bars.assign(**meta))
                        if split == "test":
                            balances[key] = bars.equity.iloc[-1]
    bars = pd.concat(ledgers, ignore_index=True)
    fc = pd.concat(forecasts, ignore_index=True)
    groups = ["construction", "variant", "horizon"]
    scores = []
    for key, g in bars[bars.split == "test"].groupby(groups):
        r = g.net_return
        fit = sm.OLS(r, np.ones(len(r))).fit(cov_type="HAC", cov_kwds={"maxlags": 20})
        p = norm.sf(fit.tvalues.iloc[0]) if r.std() > 0 else 1.0
        curve = np.r_[CAPITAL, g.equity]
        scores.append(
            dict(zip(groups, key))
            | {
                "net_return": curve[-1] / CAPITAL - 1,
                "max_drawdown": (1 - curve / np.maximum.accumulate(curve)).max(),
                "execution": g.execution.sum(),
                "carry": g.carry.sum(),
                "positive_mean_p": p,
            }
        )
    summary = pd.DataFrame(scores)
    summary["holm_p"] = multipletests(summary.positive_mean_p, method="holm")[1]
    forecast_quality = []
    for key, frame in fc.groupby(groups + ["window", "split"]):
        forecast_quality.append(
            dict(zip(groups + ["window", "split"], key))
            | {
                "observations": len(frame),
                "rank_ic": frame.forecast.corr(frame.realized, method="spearman"),
                "rmse": np.sqrt(np.mean((frame.forecast - frame.realized) ** 2)),
            }
        )
    if args.minute:
        comparisons = []
        for (family, h), frame in bars[bars.split == "test"].groupby(
            ["construction", "horizon"]
        ):
            daily = frame.pivot(index="date", columns="variant", values="net_return")
            delta = daily.minute_augmented - daily.matched_daily
            fit = sm.OLS(delta, np.ones(len(delta))).fit(
                cov_type="HAC", cov_kwds={"maxlags": 20}
            )
            low, high = fit.conf_int().iloc[0]
            comparisons.append(
                {
                    "construction": family,
                    "horizon": h,
                    "cumulative_difference": (1 + daily.minute_augmented).prod()
                    - (1 + daily.matched_daily).prod(),
                    "mean_daily_bp": delta.mean() * 10000,
                    "ci95_low_daily_bp": low * 10000,
                    "ci95_high_daily_bp": high * 10000,
                    "two_sided_p": float(fit.pvalues.iloc[0])
                    if delta.std() > 0
                    else 1.0,
                }
            )
        comparison = pd.DataFrame(comparisons)
        comparison["holm_p"] = multipletests(comparison.two_sided_p, method="holm")[1]
        comparison.to_csv(out / "granularity.csv", index=False)
        inverse_coverage = []
        for fold in dates.loc[dates.selected.notna()].itertuples():
            pair = tuple(fold.selected.split("/"))
            for split, left, right in [
                ("train", fold.train_start, fold.train_end),
                ("test", fold.test_start, fold.test_end),
            ]:
                sessions = prices.loc[left:right].index
                count = (
                    morning_moments(args.cache, pair)
                    .reindex(sessions)
                    .synchronized_returns
                )
                inverse_coverage.append(
                    {
                        "window": fold.window,
                        "pair": fold.selected,
                        "split": split,
                        "sessions": len(count),
                        "qualified": int(count.ge(200).sum()),
                        "median_observed_consecutive_minutes": float(
                            count.fillna(0).median()
                        ),
                        "max_observed_consecutive_minutes": int(count.fillna(0).max()),
                    }
                )
        pd.DataFrame(inverse_coverage).to_csv(
            out / "inverse_minute_coverage.csv", index=False
        )
    for name, frame in [
        ("summary", summary),
        ("folds", pd.DataFrame(selections)),
        ("screen", pd.concat(tests)),
        ("coverage", pd.DataFrame(coverage)),
        ("forecasts", fc),
        ("forecast_quality", pd.DataFrame(forecast_quality)),
        ("bars", bars),
        ("trades", pd.DataFrame(trades)),
    ]:
        frame.to_csv(out / f"{name}.csv", index=False)
    files = [
        args.cache / name
        for name in [
            "membership_intervals.parquet",
            "prices.parquet",
            "no_trade_days.csv",
        ]
    ]
    if args.minute:
        symbols = {
            s for rec in selections if rec["pair"] for s in rec["pair"].split("/")
        }
        symbols.update(s for pair in dates.selected.dropna() for s in pair.split("/"))
        files.extend(
            args.cache / f"{s}.parquet"
            for s in sorted(symbols)
            if (args.cache / f"{s}.parquet").exists()
        )
    files.extend(
        [
            Path(__file__),
            Path("data/minute_features.py"),
            Path("data/historical_panel.py"),
            FOLDS,
        ]
    )
    manifest = {
        "files": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        "packages": {
            p: version(p)
            for p in ["numpy", "pandas", "statsmodels", "scikit-learn", "pyarrow"]
        },
        "daily_source": "https://github.com/Johnbrick123/sp500-data/releases/tag/data",
        "minute_revision": "eff0dac55d843b3b83caec819efa213021a3e3e5"
        if args.minute
        else None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minute", action="store_true")
    main(parser.parse_args())
