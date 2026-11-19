import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from data.loaders import DataLoader
from research_pivot import (
    _backtest_policy,
    _diagnose_pair,
    _eg_pvalue,
    _fit_residual_model,
    _generate_windows,
    _json_safe,
    _load_config,
    _metrics,
    _stitch_equity,
    _training_prices,
    _missing_evaluation_price,
)


OUTPUT_DIR = Path("outputs/economic_pairs")
INITIAL_CAPITAL = 100000.0
MIN_TRAIN_OBSERVATIONS = 252
OOS_ADF_P_THRESHOLD = 0.10
OOS_COINTEGRATION_P_THRESHOLD = 0.10
MAX_SURVIVAL_HALF_LIFE_DAYS = 126


ECONOMIC_FAMILIES = [
    {
        "family": "same_sector_etf",
        "family_type": "normal_etf",
        "pairs": [
            ("XLE", "XOP"),
            ("XLK", "QQQ"),
            ("XLF", "KRE"),
            ("XLI", "ITA"),
            ("XLV", "IBB"),
            ("XRT", "XLY"),
            ("XLU", "VPU"),
        ],
    },
    {
        "family": "commodity_related_etf",
        "family_type": "normal_etf",
        "pairs": [
            ("GLD", "SLV"),
            ("USO", "XLE"),
            ("UNG", "XLE"),
            ("URA", "URNM"),
            ("CCJ", "URA"),
            ("CCJ", "URNM"),
        ],
    },
    {
        "family": "index_market_cap_etf",
        "family_type": "normal_etf",
        "pairs": [
            ("SPY", "IVV"),
            ("QQQ", "XLK"),
            ("IWM", "VB"),
            ("DIA", "SPY"),
        ],
    },
]


VOLATILITY_SYMBOLS = ["VIXY", "UVXY", "SVXY", "VXX", "VIXM", "TVIXF"]


def _volatility_family():
    return {
        "family": "volatility_hedge_isolated",
        "family_type": "volatility_inverse_leveraged_isolated",
        "pairs": list(combinations(VOLATILITY_SYMBOLS, 2)),
    }


def _pair_key(pair):
    return f"{pair[0]}-{pair[1]}"


def _safe_float(value):
    if value is None:
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if math.isfinite(value) else np.nan


def _available_pair(pair, train_prices):
    missing = [sym for sym in pair if sym not in train_prices.columns]
    if missing:
        return False, f"missing_symbols:{','.join(missing)}"
    train_count = train_prices[list(pair)].dropna().shape[0]
    if train_count < MIN_TRAIN_OBSERVATIONS:
        return False, f"insufficient_train_observations:{train_count}"
    return True, "selected"


def _oos_survived(row):
    test_adf = _safe_float(row.get("test_adf_p"))
    train_coint = _safe_float(row.get("train_coint_p"))
    test_coint = _safe_float(row.get("test_coint_p"))
    half_life = _safe_float(row.get("test_half_life"))
    return bool(
        np.isfinite(test_adf)
        and test_adf < OOS_ADF_P_THRESHOLD
        and np.isfinite(train_coint)
        and train_coint < OOS_COINTEGRATION_P_THRESHOLD
        and np.isfinite(test_coint)
        and test_coint < OOS_COINTEGRATION_P_THRESHOLD
        and np.isfinite(half_life)
        and 0 < half_life <= MAX_SURVIVAL_HALF_LIFE_DAYS
    )


def _format_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def _markdown_table(df, cols):
    if df.empty:
        return "No rows."
    use = df[cols].copy()
    lines = ["| " + " | ".join(use.columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(use.columns)) + " |")
    for _, row in use.iterrows():
        lines.append("| " + " | ".join(_format_value(row[c]) for c in use.columns) + " |")
    return "\n".join(lines)


def _collect_pair_contributions(frictions):
    contributions = {}
    if frictions.empty:
        return contributions
    for col in frictions.columns:
        if col.startswith("pnl_"):
            contributions[col[4:]] = float(frictions[col].sum())
    return contributions


def _run_family(family, panels, returns, windows, cfg):
    prices = panels[cfg["data"]["price_type"]]
    summary_rows = []
    diag_rows = []
    availability_notes = []
    equity_curves = {}
    zscore_pair_window_pnl = {}

    curves_by_policy = {"cash": [], "random": [], "zscore": []}
    selected_pair_keys = set()

    for w in windows:
        train_prices = _training_prices(panels, w, cfg)
        test_prices = prices.iloc[w["validation_end_idx"]:w["test_end_idx"]]
        train_returns = returns.loc[train_prices.index.intersection(returns.index)]
        test_returns = returns.loc[test_prices.index.intersection(returns.index)]

        selected_pairs = []
        models = {}

        for pair in family["pairs"]:
            pair = tuple(pair)
            available, reason = _available_pair(pair, train_prices)
            availability_notes.append(
                {
                    "family": family["family"],
                    "window": w["window"],
                    "pair": _pair_key(pair),
                    "status": reason,
                }
            )
            if not available:
                continue

            alpha, beta = _fit_residual_model(train_prices, pair)
            train_coint_p = _eg_pvalue(train_prices, *pair)
            selected_pairs.append(pair)
            selected_pair_keys.add(_pair_key(pair))
            models[pair] = (alpha, beta)
            diag = _diagnose_pair(
                train_prices=train_prices,
                test_prices=test_prices,
                train_returns=train_returns,
                test_returns=test_returns,
                pair=pair,
                alpha=alpha,
                beta=beta,
                selected_meta={
                    "variant": family["family"],
                    "window": w["window"],
                    "side": family["family_type"],
                    "train_coint_p": train_coint_p,
                    "candidate_count": len(family["pairs"]),
                    "coint_candidates_evaluated": len(family["pairs"]),
                },
            )
            diag["family"] = family["family"]
            diag["family_type"] = family["family_type"]
            diag_rows.append(diag)


        missing_quote = _missing_evaluation_price(prices, selected_pairs, w, cfg)
        if missing_quote:
            for policy in curves_by_policy:
                summary_rows.append({"family": family["family"], "family_type": family["family_type"],
                                     "policy": policy, "status": missing_quote, "total_return": np.nan,
                                     "sharpe": np.nan, "max_drawdown": np.nan, "bar_profit_factor": np.nan,
                                     "turnover": np.nan, "total_costs": np.nan})
            return summary_rows, [], {}, availability_notes

        for policy in curves_by_policy:
            curve, pnls, frictions = _backtest_policy(
                policy=policy,
                panels=panels, w=w,
                capital=curves_by_policy[policy][-1][1].iloc[-1] if curves_by_policy[policy] else INITIAL_CAPITAL,
                pairs=selected_pairs,
                models=models,
                cfg=cfg,
                seed=1000 + w["window"],
            )
            curves_by_policy[policy].append((w["window"], curve, pnls, frictions))
            if policy == "zscore":
                for pair_key, pnl in _collect_pair_contributions(frictions).items():
                    zscore_pair_window_pnl[(w["window"], pair_key)] = pnl

    for policy, rows in curves_by_policy.items():
        curve, pnls, frictions, pair_pnls = _stitch_equity(rows)
        name = f"{family['family']}__{policy}"
        equity_curves[name] = curve.rename(name)
        summary_rows.append(
            {
                "family": family["family"],
                "family_type": family["family_type"],
                "policy": policy, "status": "complete",
                "selected_pair_windows": len(rows),
                "declared_pairs": len({_pair_key(pair) for pair in family["pairs"]}),
                "selected_unique_pairs": len(selected_pair_keys),
                **_metrics(curve, pnls, frictions),
            }
        )

    for row in diag_rows:
        pnl = zscore_pair_window_pnl.get((row["window"], row["pair"]), 0.0)
        row["zscore_pair_pnl"] = pnl
        row["zscore_pair_return_on_initial_capital"] = pnl / INITIAL_CAPITAL
        row["oos_survived"] = _oos_survived(row)

    return summary_rows, diag_rows, equity_curves, availability_notes


def _compare_to_broad(summary_df):
    broad_path = Path("outputs/research_pivot/summary.csv")
    if not broad_path.exists() or "bar_profit_factor" not in pd.read_csv(broad_path, nrows=0).columns:
        return {
            "available": False,
            "best_broad_family": None,
            "best_broad_return": np.nan,
            "best_economic_family": None,
            "best_economic_return": np.nan,
        }
    broad = pd.read_csv(broad_path)
    broad_z = broad[broad["policy"] == "zscore"].sort_values("total_return", ascending=False)
    econ_z = summary_df[
        (summary_df["policy"] == "zscore")
        & (summary_df["family_type"] != "volatility_inverse_leveraged_isolated")
    ].sort_values("total_return", ascending=False)
    best_broad = broad_z.iloc[0] if len(broad_z) else None
    best_econ = econ_z.iloc[0] if len(econ_z) else None
    return {
        "available": True,
        "best_broad_family": best_broad["variant"] if best_broad is not None else None,
        "best_broad_return": float(best_broad["total_return"]) if best_broad is not None else np.nan,
        "best_economic_family": best_econ["family"] if best_econ is not None else None,
        "best_economic_return": float(best_econ["total_return"]) if best_econ is not None else np.nan,
    }


def _write_report(summary_df, diag_df, availability_df, broad_compare, cfg):
    zscore = summary_df[(summary_df["policy"] == "zscore") & (summary_df["status"] == "complete")].copy()
    normal_z = zscore[zscore["family_type"] != "volatility_inverse_leveraged_isolated"].sort_values(
        "total_return", ascending=False
    )
    vol_z = zscore[zscore["family_type"] == "volatility_inverse_leveraged_isolated"].sort_values(
        "total_return", ascending=False
    )
    best_normal = normal_z.iloc[0] if len(normal_z) else None
    best_all = zscore.sort_values("total_return", ascending=False).iloc[0] if len(zscore) else None

    survival = (
        diag_df.groupby(["family", "family_type"])
        .agg(
            pairs_tested=("pair", "count"),
            oos_survival_rate=("oos_survived", "mean"),
            coint_persistence_rate=("coint_persisted", "mean"),
            median_test_adf_p=("test_adf_p", "median"),
            median_test_half_life=("test_half_life", "median"),
            total_zscore_pair_pnl=("zscore_pair_pnl", "sum"),
        )
        .reset_index()
        if not diag_df.empty
        else pd.DataFrame()
    )
    skipped = availability_df[availability_df["status"] != "selected"]
    normal_positive = normal_z[normal_z["total_return"] > 0]
    normal_has_edge = (
        len(normal_positive) > 0
        and normal_positive.iloc[0]["sharpe"] > 0
        and normal_positive.iloc[0]["bar_profit_factor"] > 1.05
    )

    if broad_compare["available"]:
        if (
            np.isfinite(broad_compare["best_economic_return"])
            and np.isfinite(broad_compare["best_broad_return"])
            and broad_compare["best_economic_return"] > broad_compare["best_broad_return"]
        ):
            broad_answer = (
                f"Yes in this locked run: best normal economic z-score family "
                f"`{broad_compare['best_economic_family']}` returned "
                f"{broad_compare['best_economic_return']:.4%}, above broad-mined "
                f"`{broad_compare['best_broad_family']}` at {broad_compare['best_broad_return']:.4%}."
            )
        else:
            broad_answer = (
                f"No clear improvement in this locked run: best normal economic z-score family "
                f"`{broad_compare['best_economic_family']}` returned "
                f"{broad_compare['best_economic_return']:.4%}, versus broad-mined "
                f"`{broad_compare['best_broad_family']}` at {broad_compare['best_broad_return']:.4%}."
            )
    else:
        broad_answer = "Broad-mined comparison artifact was not found, so this report only evaluates economic families."

    vol_answer = "No isolated volatility/inverse/leveraged family was available after cache filtering."
    if len(vol_z):
        top_vol = vol_z.iloc[0]
        top_normal_return = float(best_normal["total_return"]) if best_normal is not None else np.nan
        vol_answer = (
            f"Top isolated volatility/inverse/leveraged z-score family returned "
            f"{top_vol['total_return']:.4%}; top normal family returned {top_normal_return:.4%}. "
            "These products are reported separately and are not mixed into normal ETF evidence."
        )

    ppo_answer = (
        "No. No normal economic family met a minimum after-cost baseline gate for PPO."
        if not normal_has_edge
        else "Possibly, but only after expanding the locked walk-forward count; the current run is still too small for PPO tuning."
    )

    lines = [
        "# Economic Pairs Experiment Report",
        "",
        "## Protocol",
        "- Data: fixed daily yfinance cache used by daily validation; the script raises if the cache is missing.",
        f"- Price type: `{cfg['data'].get('price_type', 'Adj Close')}`.",
        "- Walk-forward: locked daily validation windows from `config/daily_validation_config.yaml`.",
        "- Pair selection: predeclared economic families only; no broad correlation mining and no test-set selection.",
        "- Baselines: cash, deterministic random, and residual z-score mean reversion.",
        "- PPO: not run in this experiment.",
        "- Normal ETFs are evaluated separately from volatility/inverse/leveraged ETPs.",
        f"- OOS survival diagnostic: test ADF p < {OOS_ADF_P_THRESHOLD}, train/test Engle-Granger p < {OOS_COINTEGRATION_P_THRESHOLD}, and test half-life in `(0, {MAX_SURVIVAL_HALF_LIFE_DAYS}]` trading days.",
        "",
        "## Summary",
        f"- Best normal economic z-score family: `{best_normal['family'] if best_normal is not None else 'none'}` with total return `{best_normal['total_return'] if best_normal is not None else np.nan:.6g}`.",
        f"- Best all-family z-score result: `{best_all['family'] if best_all is not None else 'none'}` with total return `{best_all['total_return'] if best_all is not None else np.nan:.6g}`.",
        f"- Economically related pairs versus broad-mined pairs: {broad_answer}",
        "",
        "## Baseline Results",
        _markdown_table(
            summary_df.sort_values(["policy", "total_return"], ascending=[True, False]),
            [
                "family",
                "family_type",
                "policy",
                "status",
                "total_return",
                "sharpe",
                "max_drawdown",
                "bar_profit_factor",
                "turnover",
                "total_costs",
            ],
        ),
        "",
        "## Spread Diagnostics By Family",
        _markdown_table(
            survival.sort_values("total_zscore_pair_pnl", ascending=False) if not survival.empty else survival,
            [
                "family",
                "family_type",
                "pairs_tested",
                "oos_survival_rate",
                "coint_persistence_rate",
                "median_test_adf_p",
                "median_test_half_life",
                "total_zscore_pair_pnl",
            ],
        ),
        "",
        "## Skipped Pairs",
        _markdown_table(skipped, ["family", "window", "pair", "status"]) if not skipped.empty else "No skipped pairs.",
        "",
        "## Answers",
        f"- Do economically related pairs mean-revert better than broad-mined pairs? {broad_answer}",
        "- Which pair families survive out of sample? Use `oos_survival_rate` above as the locked survival definition; individual pair/window outcomes are in `pair_diagnostics.csv`.",
        f"- Are volatility/inverse/leveraged ETPs responsible for fake results? {vol_answer}",
        f"- Is there any pair family worth testing with PPO later? {ppo_answer}",
        "",
        "## Interpretation",
        "- This experiment tests the economic-pair premise only. It does not tune entry thresholds, holding rules, hedge ratios, or PPO.",
        "- Pair PnL uses the signed, training-fitted hedge ratio and gross exposure cap in the PPO research simulator.",
    ]

    with open(OUTPUT_DIR / "economic_pairs_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def run_economic_pairs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _load_config()
    cache_path = Path(cfg["data"]["cache_path"])
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Daily cache is required for the economic pairs experiment and was not found: {cache_path}"
        )

    data_cfg = cfg["data"]
    panels = DataLoader.load_prices(
        assets_path=data_cfg["assets_path"],
        start=data_cfg["start_date"],
        end=data_cfg["end_date"],
        min_history=data_cfg["min_history"],
        source=data_cfg["source"],
        cache_path=data_cfg["cache_path"],
        max_ffill_gap=data_cfg["max_ffill_gap"],
        min_retained_symbols=data_cfg["min_retained_symbols"],
        preserve_calendar=True,
        coverage_tolerance_days=data_cfg["coverage_tolerance_days"],
    )
    price_type = data_cfg.get("price_type", "Adj Close")
    prices = panels[price_type]
    returns = prices.pct_change(fill_method=None)
    windows = _generate_windows(prices.index, cfg)

    families = ECONOMIC_FAMILIES + [_volatility_family()]
    summary_rows = []
    diag_rows = []
    availability_rows = []
    all_equity_curves = {}

    for family in families:
        fam_summary, fam_diag, fam_curves, fam_availability = _run_family(
            family=family,
            panels=panels,
            returns=returns,
            windows=windows,
            cfg=cfg,
        )
        summary_rows.extend(fam_summary)
        diag_rows.extend(fam_diag)
        availability_rows.extend(fam_availability)
        all_equity_curves.update(fam_curves)

    summary_df = pd.DataFrame(summary_rows)
    diag_df = pd.DataFrame(diag_rows)
    availability_df = pd.DataFrame(availability_rows)

    if not diag_df.empty:
        diag_df["oos_survived"] = diag_df.apply(_oos_survived, axis=1)
        diag_df = diag_df[
            [
                "family",
                "family_type",
                "window",
                "pair",
                "pair_type",
                "train_corr",
                "test_corr",
                "corr_abs_decay",
                "train_coint_p",
                "test_coint_p",
                "coint_persisted",
                "train_adf_p",
                "test_adf_p",
                "train_half_life",
                "test_half_life",
                "test_hurst",
                "train_spread_vol",
                "test_spread_vol",
                "spread_vol_ratio_test_train",
                "zscore_pair_pnl",
                "zscore_pair_return_on_initial_capital",
                "oos_survived",
                "candidate_count",
                "coint_candidates_evaluated",
            ]
        ]

    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    diag_df.to_csv(OUTPUT_DIR / "pair_diagnostics.csv", index=False)
    if all_equity_curves:
        equity = pd.concat(all_equity_curves.values(), axis=1)
        equity.index.name = "date"
        equity.to_csv(OUTPUT_DIR / "equity_curves.csv")
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / "equity_curves.csv", index=False)

    broad_compare = _compare_to_broad(summary_df)
    _write_report(summary_df, diag_df, availability_df, broad_compare, cfg)

    with open(OUTPUT_DIR / "run_metadata.json", "w") as f:
        json.dump(
            _json_safe(
                {
                    "families": families,
                    "windows": windows,
                    "price_type": price_type,
                    "cache_path": str(cache_path),
                    "oos_adf_p_threshold": OOS_ADF_P_THRESHOLD,
                    "oos_cointegration_p_threshold": OOS_COINTEGRATION_P_THRESHOLD,
                    "max_survival_half_life_days": MAX_SURVIVAL_HALF_LIFE_DAYS,
                    "broad_compare": broad_compare,
                }
            ),
            f,
            indent=2,
        )

    return summary_df, diag_df


if __name__ == "__main__":
    run_economic_pairs()
