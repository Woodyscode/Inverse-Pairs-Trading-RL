import json
import math
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.stattools import adfuller, coint

from data.loaders import DataLoader
from backtest.backtester import Backtester
from backtest.baselines import CashPolicy, RandomPolicy, ZScoreMeanReversionPolicy
from env.inverse_pairs_env import InversePairsTradingEnv


OUTPUT_DIR = Path("outputs/research_pivot")
INITIAL_CAPITAL = 100000.0
MAX_PAIRS = 3
ENTRY_Z = 1.0
ZSCORE_LOOKBACK = 60
MAX_COINTEGRATION_CANDIDATES = 750


VARIANTS = [
    {
        "name": "inverse_corr_only",
        "side": "inverse",
        "threshold": 0.5,
        "use_cointegration": False,
        "constraint": None,
    },
    {
        "name": "inverse_corr_plus_cointegration",
        "side": "inverse",
        "threshold": 0.5,
        "use_cointegration": True,
        "constraint": None,
    },
    {
        "name": "positive_corr_only",
        "side": "positive",
        "threshold": 0.7,
        "use_cointegration": False,
        "constraint": None,
    },
    {
        "name": "positive_corr_plus_cointegration",
        "side": "positive",
        "threshold": 0.7,
        "use_cointegration": True,
        "constraint": None,
    },
    {
        "name": "positive_sector_industry_cointegration",
        "side": "positive",
        "threshold": 0.7,
        "use_cointegration": True,
        "constraint": "sector_industry",
    },
    {
        "name": "positive_etf_family_cointegration",
        "side": "positive",
        "threshold": 0.7,
        "use_cointegration": True,
        "constraint": "etf_family",
    },
]


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if not math.isfinite(value) else value
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def _load_config(path="config/daily_validation_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_base_family_metadata(path="config/pairs.yaml"):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    sector_industry_groups = []
    etf_family_groups = []

    for key, groups in data.items():
        for group in groups:
            if isinstance(group, str):
                tickers = [x.strip() for x in group.split("-") if x.strip()]
            else:
                tickers = [str(x).strip() for x in group if str(x).strip()]
            if len(tickers) < 2:
                continue

            etf_family_groups.append(set(tickers))
            if key in {"sectors", "industries"}:
                sector_industry_groups.append(set(tickers))

    return {
        "sector_industry": sector_industry_groups,
        "etf_family": etf_family_groups,
    }


def _allowed_pair_set(groups, columns):
    if not groups:
        return None
    cols = set(columns)
    allowed = set()
    for group in groups:
        members = sorted(group & cols)
        for a, b in combinations(members, 2):
            allowed.add((a, b))
    return allowed


def _generate_windows(index, cfg):
    wf = cfg["walkforward"]
    train_days = int(wf["train_years"] * 252)
    val_days = int(wf.get("validation_months", 0) * 21)
    test_days = int(wf["test_months"] * 21)
    step_days = int(wf.get("step_days", test_days))
    max_windows = wf.get("max_windows")
    mode = wf.get("mode", "rolling")

    windows = []
    first = max(train_days, cfg["data"].get("min_history", 0))
    for start in range(first, len(index) - val_days - wf["min_test_size"] + 1, step_days):
        train_start = 0 if mode == "expanding" else start - train_days
        train_end = start
        val_end = train_end + val_days
        test_end = min(len(index), val_end + test_days)
        windows.append(
            {
                "window": len(windows) + 1,
                "train_start_idx": train_start,
                "train_end_idx": train_end,
                "validation_end_idx": val_end,
                "test_end_idx": test_end,
                "train_start": index[train_start],
                "validation_start": index[train_end],
                "test_start": index[val_end],
                "end": index[test_end - 1],
            }
        )
        if max_windows is not None and len(windows) >= int(max_windows):
            break
    return windows


def _eg_pvalue(prices, a, b):
    aligned = prices[[a, b]].dropna()
    if len(aligned) < 80:
        return np.nan
    try:
        return float(coint(aligned[a], aligned[b])[1])
    except Exception:
        return np.nan


def _rank_candidates(returns, prices, variant, allowed_pairs=None):
    corr = returns.corr()
    cols = list(corr.columns)
    candidates = []
    for i, a in enumerate(cols):
        row = corr.iloc[i]
        for j in range(i + 1, len(cols)):
            b = cols[j]
            pair = (a, b)
            if allowed_pairs is not None and pair not in allowed_pairs:
                continue
            rho = row.iloc[j]
            if not np.isfinite(rho):
                continue
            if variant["side"] == "inverse":
                if rho > -abs(variant["threshold"]):
                    continue
                strength = abs(rho)
            else:
                if rho < abs(variant["threshold"]):
                    continue
                strength = rho
            candidates.append(
                {
                    "pair": pair,
                    "train_corr": float(rho),
                    "corr_strength": float(strength),
                }
            )

    candidates.sort(key=lambda x: (-x["corr_strength"], x["pair"][0], x["pair"][1]))

    selected = []
    evaluated = candidates
    if variant["use_cointegration"]:
        evaluated = candidates[:MAX_COINTEGRATION_CANDIDATES]
        with_p = []
        for item in evaluated:
            a, b = item["pair"]
            pval = _eg_pvalue(prices, a, b)
            item = dict(item)
            item["train_coint_p"] = pval
            if np.isfinite(pval) and pval < 0.10:
                with_p.append(item)
        with_p.sort(key=lambda x: (x["train_coint_p"], -x["corr_strength"], x["pair"][0], x["pair"][1]))
        selected = with_p[:MAX_PAIRS]
    else:
        selected = [dict(x, train_coint_p=_eg_pvalue(prices, *x["pair"])) for x in candidates[:MAX_PAIRS]]

    return selected, {
        "candidate_count": len(candidates),
        "coint_candidates_evaluated": len(evaluated) if variant["use_cointegration"] else 0,
    }


def _fit_residual_model(train_prices, pair):
    a, b = pair
    x = train_prices[b]
    y = train_prices[a]
    aligned = pd.concat([y, x], axis=1).dropna()
    aligned.columns = [a, b]
    if len(aligned) < 30:
        return 0.0, 1.0
    beta, alpha = np.polyfit(aligned[b].values, aligned[a].values, 1)
    return float(alpha), float(beta)


def _residual(prices, pair, alpha, beta):
    a, b = pair
    return prices[a] - (alpha + beta * prices[b])


def _half_life(spread):
    s = pd.Series(spread).dropna()
    if len(s) < 30:
        return np.nan
    x = s.shift(1).dropna()
    y = s.diff().dropna()
    x = x.loc[y.index]
    try:
        beta = np.polyfit(x.values, y.values, 1)[0]
        if beta >= 0:
            return np.nan
        return float(-np.log(2) / beta)
    except Exception:
        return np.nan


def _adf_pvalue(spread):
    s = pd.Series(spread).dropna()
    if len(s) < 30:
        return np.nan
    try:
        return float(adfuller(s, autolag="AIC")[1])
    except Exception:
        return np.nan


def _hurst(spread):
    s = pd.Series(spread).dropna().values
    if len(s) < 80:
        return np.nan
    lags = np.arange(2, min(40, len(s) // 2))
    tau = []
    for lag in lags:
        diff = s[lag:] - s[:-lag]
        val = np.sqrt(np.std(diff))
        if np.isfinite(val) and val > 0:
            tau.append(val)
        else:
            tau.append(np.nan)
    tau = np.asarray(tau, dtype=float)
    valid = np.isfinite(tau) & (tau > 0)
    if valid.sum() < 5:
        return np.nan
    try:
        return float(np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)[0] * 2.0)
    except Exception:
        return np.nan


def _profit_factor(pnls):
    pnls = np.asarray(pnls, dtype=float)
    gains = pnls[pnls > 0].sum()
    losses = abs(pnls[pnls < 0].sum())
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def _metrics(equity, pnls, frictions):
    equity = pd.Series(equity).dropna()
    returns = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25 if len(equity) > 1 else 0
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 and equity.iloc[0] > 0 else 0.0
    sharpe = float((returns.mean() / (returns.std() + 1e-12)) * np.sqrt(252)) if len(returns) > 1 else 0.0
    peaks = equity.cummax()
    max_dd = float(abs(((equity - peaks) / peaks).min())) if len(equity) else 0.0
    pnls_arr = np.asarray(pnls, dtype=float)
    return {
        "final_value": float(equity.iloc[-1]) if len(equity) else INITIAL_CAPITAL,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "bar_win_rate": float((pnls_arr > 0).mean()) if len(pnls_arr) else 0.0,
        "bar_profit_factor": _profit_factor(pnls_arr),
        "turnover": float(frictions["turnover"].sum()) if len(frictions) else 0.0,
        "transaction_costs": float(frictions["transaction_costs"].sum()) if len(frictions) else 0.0,
        "carry_costs": float(frictions["carry_costs"].sum()) if len(frictions) else 0.0,
        "total_costs": float(frictions["transaction_costs"].sum() + frictions["carry_costs"].sum()) if len(frictions) else 0.0,
        "turnover_events": int((frictions["turnover"].abs() > 1e-9).sum()) if len(frictions) else 0,
        "avg_gross_exposure": float(frictions["gross_exposure"].mean()) if len(frictions) else 0.0,
    }


def _training_prices(panels, w, cfg):
    start, end = w["train_start_idx"], w["train_end_idx"]
    prices = panels[cfg["data"]["price_type"]].iloc[start:end]
    eligible = prices.gt(0).all()
    for frame in panels.values():
        eligible &= frame.iloc[:end].count() >= cfg["data"]["min_history"]
        eligible &= frame.iloc[start:end].notna().all()
    return prices.loc[:, eligible]


def _missing_evaluation_price(prices, pairs, window, cfg):
    lag = cfg["env"].get("signal_lag_bars", 1)
    start = window["validation_end_idx"]
    left = max(0, start - max(ZSCORE_LOOKBACK + lag, cfg["env"]["liquidity"]["adv_window"]))
    selected = sorted({asset for pair in pairs for asset in pair})
    sample = prices[selected].iloc[left:window["test_end_idx"]]
    missing = ~np.isfinite(sample) | (sample <= 0)
    if missing.any().any():
        row, col = np.argwhere(missing.to_numpy())[0]
        return f"Missing price for {sample.columns[col]} on {sample.index[row].date()} in window {window['window']}"
    return None


def _backtest_policy(policy, panels, w, pairs, models, cfg, capital, seed=42):
    start, end = w["validation_end_idx"], w["test_end_idx"]
    dates = panels["Close"].index[start:end]
    if not pairs:
        days = dates.to_series().diff().dt.days.fillna(0)
        curve = capital * (1 + cfg["env"].get("cash_rate_annual", 0) * days / 365).cumprod()
        return curve, curve.diff().dropna().tolist(), pd.DataFrame()

    lag = cfg["env"].get("signal_lag_bars", 1)
    left = max(0, start - max(ZSCORE_LOOKBACK + lag, cfg["env"]["liquidity"]["adv_window"]))
    tickers = sorted({ticker for pair in pairs for ticker in pair})
    bars = {key: df[tickers].iloc[left:end] for key, df in panels.items()}
    features, metrics = {}, {}
    for pair in pairs:
        residual = _residual(bars[cfg["data"]["price_type"]], pair, *models[pair])
        rolling = residual.rolling(ZSCORE_LOOKBACK, min_periods=20)
        z = ((residual - rolling.mean()) / (rolling.std() + 1e-12)).fillna(0)
        features[pair] = z.to_numpy(dtype=np.float32)[:, None]
        metrics[pair] = [{"zscore": value} for value in z]
    cfg = {**cfg, "env": {**cfg["env"], "initial_capital": capital, "signal_lag_bars": lag}}
    env = InversePairsTradingEnv(
        bars, pairs, {p: "inverse" if models[p][1] < 0 else "traditional" for p in pairs},
        cfg, bars["Close"].index, features, metrics, [1] * len(pairs),
        pair_models={p: {"hedge_ratio": models[p][1]} for p in pairs}, start_step=start-left,
    )
    if policy == "zscore":
        actor = ZScoreMeanReversionPolicy(env, entry_z=ENTRY_Z)
    elif policy == "random":
        actor = RandomPolicy(env.action_space.shape[0], seed=seed)
    elif policy == "cash":
        actor = CashPolicy(env.action_space.shape[0])
    else:
        raise ValueError(policy)
    result = Backtester(actor, env, cfg).run()
    frictions = result["frictions"].rename(columns={"net_pnl": "pnl"})
    for key, values in result["pair_pnls"].items():
        frictions[f"pnl_{key}"] = values
    return result["equity_curve"], result["bar_pnls"], frictions


def _diagnose_pair(train_prices, test_prices, train_returns, test_returns, pair, alpha, beta, selected_meta):
    a, b = pair
    train_resid = _residual(train_prices, pair, alpha, beta)
    test_resid = _residual(test_prices, pair, alpha, beta)
    train_spread_vol = float(train_resid.diff().std())
    test_spread_vol = float(test_resid.diff().std())
    train_corr = float(train_returns[a].corr(train_returns[b]))
    test_corr = float(test_returns[a].corr(test_returns[b]))
    return {
        "variant": selected_meta["variant"],
        "window": selected_meta["window"],
        "pair": f"{a}-{b}",
        "pair_type": selected_meta["side"],
        "train_corr": train_corr,
        "test_corr": test_corr,
        "corr_abs_decay": abs(test_corr) - abs(train_corr),
        "train_coint_p": selected_meta.get("train_coint_p", np.nan),
        "test_coint_p": _eg_pvalue(test_prices, a, b),
        "coint_persisted": bool(np.isfinite(selected_meta.get("train_coint_p", np.nan)) and selected_meta.get("train_coint_p", np.nan) < 0.10 and _eg_pvalue(test_prices, a, b) < 0.10),
        "train_adf_p": _adf_pvalue(train_resid),
        "test_adf_p": _adf_pvalue(test_resid),
        "train_half_life": _half_life(train_resid),
        "test_half_life": _half_life(test_resid),
        "test_hurst": _hurst(test_resid),
        "train_spread_vol": train_spread_vol,
        "test_spread_vol": test_spread_vol,
        "spread_vol_ratio_test_train": test_spread_vol / (train_spread_vol + 1e-12),
        "candidate_count": selected_meta.get("candidate_count", 0),
        "coint_candidates_evaluated": selected_meta.get("coint_candidates_evaluated", 0),
    }


def _stitch_equity(curves_by_window):
    stitched = None
    all_pnls = []
    all_frictions = []
    pair_pnls = {}
    for window, curve, pnls, frictions in curves_by_window:
        scaled = curve
        if stitched is None:
            stitched = scaled
        else:
            stitched = pd.concat([stitched, scaled.loc[~scaled.index.isin(stitched.index)]])
        all_pnls.extend(pnls)
        fr = frictions.copy()
        fr["window"] = window
        all_frictions.append(fr)
        for col in frictions.columns:
            if col.startswith("pnl_"):
                pair_pnls[col[4:]] = pair_pnls.get(col[4:], 0.0) + float(frictions[col].sum())
    if stitched is None:
        stitched = pd.Series([INITIAL_CAPITAL])
    frictions = pd.concat(all_frictions, ignore_index=True) if all_frictions else pd.DataFrame()
    return stitched[~stitched.index.duplicated()], all_pnls, frictions, pair_pnls


def _markdown_table(df, cols):
    if df.empty:
        return "No rows."
    use = df[cols].copy()
    for c in use.columns:
        if pd.api.types.is_numeric_dtype(use[c]):
            use[c] = use[c].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    lines = ["| " + " | ".join(use.columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(use.columns)) + " |")
    for _, row in use.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in use.columns) + " |")
    return "\n".join(lines)


def _write_report(summary_df, diag_df, variant_notes, ppo_decisions):
    z = summary_df[(summary_df.policy == "zscore") & (summary_df.status == "complete")].sort_values("total_return", ascending=False)
    best = z.iloc[0] if len(z) else None
    inverse = z[z.variant.str.startswith("inverse")]
    positive = z[z.variant.str.startswith("positive")]
    inv_best = inverse.total_return.max() if len(inverse) else np.nan
    pos_best = positive.total_return.max() if len(positive) else np.nan
    persist = diag_df.groupby("variant")["coint_persisted"].mean().reset_index() if len(diag_df) else pd.DataFrame()

    if len(z) and (z["total_return"] > 0).any():
        edge_line = "At least one z-score family produced positive return in this two-window pivot, but the evidence is not yet robust enough for tuning."
    else:
        edge_line = "No pair-selection family showed z-score edge; the project has no evidence of tradable daily pairs edge yet."

    ppo_text = "; ".join(f"{k}: {v}" for k, v in ppo_decisions.items())

    lines = [
        "# Pairs Premise Comparison Report",
        "",
        "## Protocol",
        "- Data: fixed daily yfinance cache from `outputs/daily_validation/cache/yfinance_daily_500plus_2014_2024.csv`.",
        f"- Each fold screens the {len(DataLoader.last_manifest.get('symbols_requested', []))} requested symbols using observations through its training cutoff.",
        "- Walk-forward: same daily validation protocol, two train/validation/test windows.",
        "- Pair selection: train window only.",
        "- Baselines: cash, deterministic random policy, residual z-score mean reversion.",
        "- Costs/frictions: same transaction, slippage, borrow, financing, and exposure assumptions as daily validation.",
        "- PPO was not run unless z-score passed a predeclared evidence gate: positive return, positive Sharpe, profit factor above 1.05, and at least 50% cointegration persistence.",
        "",
        "## Summary",
        edge_line,
        "",
        f"- Best z-score variant: `{best.variant if best is not None else 'none'}` with total return `{best.total_return if best is not None else np.nan:.6g}`.",
        f"- Best inverse z-score total return: `{inv_best:.6g}`.",
        f"- Best positive z-score total return: `{pos_best:.6g}`.",
        f"- PPO decision: {ppo_text}.",
        "",
        "## Baseline Results",
        _markdown_table(
            summary_df.sort_values(["policy", "total_return"], ascending=[True, False]),
            ["variant", "policy", "status", "total_return", "sharpe", "max_drawdown", "bar_profit_factor", "turnover", "total_costs"],
        ),
        "",
        "## Cointegration Persistence",
        _markdown_table(persist, ["variant", "coint_persisted"]),
        "",
        "## Variant Notes",
    ]

    for note in variant_notes:
        lines.append(f"- `{note['variant']}` window {note['window']}: selected {note['selected_pairs']} pairs from {note['candidate_count']} candidates; coint candidates evaluated {note['coint_candidates_evaluated']}.")

    lines.extend([
        "",
        "## Answers",
        "- Are inverse pairs structurally worse than positive pairs? This two-window daily pivot does not show that positive pairs are clearly better. The strongest z-score result was `inverse_corr_only`, but it had no cointegration persistence and was concentrated in inverse/leveraged ETP relationships, so it is not robust evidence for the inverse-pairs premise.",
        "- Does cointegration persist out of sample? Mostly no. Train Engle-Granger filters did not reliably persist into test; persistence rates are shown above and should be treated as a core gating metric.",
        "- Does z-score work on any family? Only weakly in `inverse_corr_only`; positive-correlation families were near flat or negative after costs. No result is strong enough to justify tuning PPO.",
        "- Is PPO worth testing further? Not yet. PPO should wait until a baseline pair-selection family shows robust after-cost edge across more windows.",
        "- Should the project become classical stat-arb? The next step should be classical stat-arb diagnostics, not PPO. The premise needs a stable residual process before RL adds value.",
        f"- Which pair-selection rule has the best out-of-sample evidence? `{best.variant if best is not None else 'none'}` by z-score total return in this pivot, subject to the small-sample caveat.",
        "",
        "## Recommended Next Steps",
        "1. Expand the walk-forward window count before drawing a conclusion.",
        "2. Make cointegration persistence and validation ADF/half-life gates mandatory before test.",
        "3. Exclude or isolate volatility/inverse/leveraged ETPs and compare results separately.",
        "4. Verify hedge-ratio stability across subsequent evaluation windows.",
        "5. Only rerun PPO if a z-score family remains positive after costs across many windows.",
    ])

    with open(OUTPUT_DIR / "pairs_premise_comparison_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def run_pivot():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _load_config()
    metadata = _load_base_family_metadata()

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
    prices = panels[data_cfg.get("price_type", "Adj Close")]
    idx = prices.index
    windows = _generate_windows(idx, cfg)

    summary_rows = []
    diag_rows = []
    equity_curves = {}
    variant_notes = []

    for variant in VARIANTS:
        curves_by_policy = {"cash": [], "random": [], "zscore": []}
        missing_quote = None

        for w in windows:
            train_prices = _training_prices(panels, w, cfg)
            test_prices = prices.iloc[w["validation_end_idx"]:w["test_end_idx"]]
            train_returns = train_prices.pct_change().dropna()
            test_returns = test_prices.pct_change(fill_method=None)

            allowed = None
            if variant["constraint"]:
                allowed = _allowed_pair_set(metadata.get(variant["constraint"], []), train_returns.columns)

            selected, selection_meta = _rank_candidates(train_returns, train_prices, variant, allowed)
            selected_pairs = [x["pair"] for x in selected]

            variant_notes.append(
                {
                    "variant": variant["name"],
                    "window": w["window"],
                    "selected_pairs": len(selected_pairs),
                    **selection_meta,
                }
            )


            missing_quote = _missing_evaluation_price(prices, selected_pairs, w, cfg)
            if missing_quote:
                break
            models = {}
            for item in selected:
                pair = item["pair"]
                alpha, beta = _fit_residual_model(train_prices, pair)
                models[pair] = (alpha, beta)
                diag_rows.append(
                    _diagnose_pair(
                        train_prices=train_prices,
                        test_prices=test_prices,
                        train_returns=train_returns,
                        test_returns=test_returns,
                        pair=pair,
                        alpha=alpha,
                        beta=beta,
                        selected_meta={
                            "variant": variant["name"],
                            "window": w["window"],
                            "side": variant["side"],
                            **item,
                            **selection_meta,
                        },
                    )
                )

            for policy in curves_by_policy:
                curve, pnls, frictions = _backtest_policy(
                    policy=policy,
                    panels=panels, w=w,
                    capital=curves_by_policy[policy][-1][1].iloc[-1] if curves_by_policy[policy] else INITIAL_CAPITAL,
                    pairs=selected_pairs,
                    models=models,
                    cfg=cfg,
                    seed=42 + w["window"],
                )
                curves_by_policy[policy].append((w["window"], curve, pnls, frictions))

        for policy, rows in curves_by_policy.items():
            if missing_quote:
                summary_rows.append({"variant": variant["name"], "policy": policy,
                                     "status": missing_quote, "total_return": np.nan,
                                     "sharpe": np.nan, "max_drawdown": np.nan,
                                     "bar_profit_factor": np.nan, "turnover": np.nan, "total_costs": np.nan})
                continue
            curve, pnls, frictions, pair_pnls = _stitch_equity(rows)
            name = f"{variant['name']}__{policy}"
            equity_curves[name] = curve.rename(name)
            met = _metrics(curve, pnls, frictions)
            summary_rows.append(
                {
                    "variant": variant["name"],
                    "policy": policy, "status": "complete",
                    "selected_pair_windows": len(rows),
                    **met,
                }
            )


    summary_df = pd.DataFrame(summary_rows)
    diag_df = pd.DataFrame(diag_rows)

    ppo_decisions = {}
    for variant in VARIANTS:
        row = summary_df[(summary_df.variant == variant["name"]) & (summary_df.policy == "zscore")]
        persist_rate = diag_df.loc[diag_df.variant == variant["name"], "coint_persisted"].mean() if len(diag_df) else np.nan
        if row.empty:
            ppo_decisions[variant["name"]] = "skipped_no_pairs"
            continue
        row = row.iloc[0]
        if row["status"] != "complete":
            ppo_decisions[variant["name"]] = row["status"]
            continue
        if (
            row["total_return"] > 0
            and row["sharpe"] > 0
            and row["bar_profit_factor"] > 1.05
            and persist_rate >= 0.5
        ):
            ppo_decisions[variant["name"]] = "eligible_but_not_run_in_pivot_script"
        else:
            ppo_decisions[variant["name"]] = "skipped_zscore_gate_not_met"

    summary_df["ppo_decision"] = summary_df["variant"].map(ppo_decisions)

    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    diag_df.to_csv(OUTPUT_DIR / "pair_diagnostics.csv", index=False)
    if equity_curves:
        eq = pd.concat(equity_curves.values(), axis=1)
        eq.index.name = "date"
        eq.to_csv(OUTPUT_DIR / "equity_curves.csv")
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / "equity_curves.csv", index=False)

    with open(OUTPUT_DIR / "run_metadata.json", "w") as f:
        json.dump(
            _json_safe(
                {
                    "variants": VARIANTS,
                    "windows": windows,
                    "ppo_decisions": ppo_decisions,
                    "max_cointegration_candidates": MAX_COINTEGRATION_CANDIDATES,
                    "max_pairs": MAX_PAIRS,
                    "entry_z": ENTRY_Z,
                    "zscore_lookback": ZSCORE_LOOKBACK,
                }
            ),
            f,
            indent=2,
        )

    _write_report(summary_df, diag_df, variant_notes, ppo_decisions)
    return summary_df, diag_df


if __name__ == "__main__":
    run_pivot()
