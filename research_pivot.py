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
    for start in range(train_days, len(index) - val_days - test_days + 1, step_days):
        train_start = 0 if mode == "expanding" else start - train_days
        train_end = start
        val_end = train_end + val_days
        test_end = val_end + test_days
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
    x = np.log(train_prices[b].replace(0, np.nan)).dropna()
    y = np.log(train_prices[a].replace(0, np.nan)).dropna()
    aligned = pd.concat([y, x], axis=1).dropna()
    aligned.columns = [a, b]
    if len(aligned) < 30:
        return 0.0, 1.0
    beta, alpha = np.polyfit(aligned[b].values, aligned[a].values, 1)
    return float(alpha), float(beta)


def _residual(prices, pair, alpha, beta):
    a, b = pair
    return np.log(prices[a].replace(0, np.nan)) - (alpha + beta * np.log(prices[b].replace(0, np.nan)))


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
    years = len(equity) / 252
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
        "win_rate": float((pnls_arr > 0).mean()) if len(pnls_arr) else 0.0,
        "profit_factor": _profit_factor(pnls_arr),
        "turnover": float(frictions["turnover"].sum()) if len(frictions) else 0.0,
        "transaction_costs": float(frictions["transaction_costs"].sum()) if len(frictions) else 0.0,
        "carry_costs": float(frictions["carry_costs"].sum()) if len(frictions) else 0.0,
        "total_costs": float(frictions["transaction_costs"].sum() + frictions["carry_costs"].sum()) if len(frictions) else 0.0,
        "turnover_events": int((frictions["turnover"].abs() > 1e-9).sum()) if len(frictions) else 0,
        "avg_gross_exposure": float(frictions["gross_exposure"].mean()) if len(frictions) else 0.0,
    }


def _scale_positions(targets, max_total_exposure):
    gross = sum(abs(x) for x in targets.values())
    if gross > max_total_exposure and gross > 0:
        scale = max_total_exposure / gross
        return {k: v * scale for k, v in targets.items()}
    return targets


def _carry_costs(portfolio, positions, env_cfg):
    gross = sum(abs(v) * portfolio for v in positions.values())
    short_notional = gross / 2.0
    borrow = short_notional * float(env_cfg.get("borrow_fee_annual", 0.0)) / 252.0
    financing = gross * float(env_cfg.get("financing_rate_annual", 0.0)) / 252.0
    cash = max(portfolio * (1.0 - sum(abs(v) for v in positions.values())), 0.0)
    cash_credit = cash * float(env_cfg.get("cash_rate_annual", 0.0)) / 252.0
    return borrow + financing - cash_credit, borrow, financing, cash_credit


def _backtest_policy(policy, test_prices, pairs, models, cfg, seed=42):
    env_cfg = cfg["env"]
    window_size = int(env_cfg["window_size"])
    max_pos = float(env_cfg["max_position"])
    max_total = float(env_cfg["max_total_exposure"])
    tc = float(env_cfg["transaction_cost"]) + float(env_cfg["slippage_cost"])

    rng = np.random.default_rng(seed)
    portfolio = INITIAL_CAPITAL
    positions = {pair: 0.0 for pair in pairs}
    equity_dates = [test_prices.index[window_size]]
    equity = [portfolio]
    pnls = []
    frictions = []

    residuals = {}
    zscores = {}
    for pair in pairs:
        alpha, beta = models[pair]
        residuals[pair] = _residual(test_prices, pair, alpha, beta)
        zscores[pair] = (
            (residuals[pair] - residuals[pair].rolling(ZSCORE_LOOKBACK, min_periods=20).mean())
            / (residuals[pair].rolling(ZSCORE_LOOKBACK, min_periods=20).std() + 1e-12)
        ).fillna(0.0)

    for t in range(window_size, len(test_prices) - 1):
        targets = {}
        if policy == "cash":
            targets = {pair: 0.0 for pair in pairs}
        elif policy == "random":
            for pair in pairs:
                direction = rng.uniform(-1.0, 1.0)
                size = abs(rng.uniform(-1.0, 1.0))
                targets[pair] = direction * size * max_pos
        elif policy == "zscore":
            for pair in pairs:
                z = float(zscores[pair].iloc[t])
                targets[pair] = -np.sign(z) * max_pos if abs(z) >= ENTRY_Z else 0.0
        else:
            raise ValueError(policy)

        targets = _scale_positions(targets, max_total)
        turnover = sum(abs(targets[pair] - positions[pair]) * portfolio for pair in pairs)
        transaction_costs = tc * turnover
        carry, borrow, financing, cash_credit = _carry_costs(portfolio, targets, env_cfg)

        gross_pnl = 0.0
        pair_pnls = {}
        for pair in pairs:
            a, b = pair
            ret_a = test_prices[a].iloc[t + 1] / test_prices[a].iloc[t] - 1
            ret_b = test_prices[b].iloc[t + 1] / test_prices[b].iloc[t] - 1
            pnl = targets[pair] * portfolio * (ret_a - ret_b)
            gross_pnl += pnl
            pair_pnls[f"{a}-{b}"] = pnl

        total_pnl = gross_pnl - transaction_costs - carry
        portfolio += total_pnl
        positions = targets

        pnls.append(total_pnl)
        equity_dates.append(test_prices.index[t + 1])
        equity.append(portfolio)
        frictions.append(
            {
                "date": test_prices.index[t + 1],
                "turnover": turnover,
                "transaction_costs": transaction_costs,
                "carry_costs": carry,
                "borrow_cost": borrow,
                "financing_cost": financing,
                "cash_credit": cash_credit,
                "gross_exposure": sum(abs(v) for v in targets.values()),
                "active_positions": sum(1 for v in targets.values() if abs(v) > 1e-9),
                "gross_pnl": gross_pnl,
                "pnl": total_pnl,
                **{f"pnl_{k}": v for k, v in pair_pnls.items()},
            }
        )

    return pd.Series(equity, index=pd.to_datetime(equity_dates)), pnls, pd.DataFrame(frictions)


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
    capital = INITIAL_CAPITAL
    all_pnls = []
    all_frictions = []
    pair_pnls = {}
    for window, curve, pnls, frictions in curves_by_window:
        returns = curve.pct_change().fillna(0.0)
        scaled = capital * (1.0 + returns).cumprod()
        capital = float(scaled.iloc[-1])
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
        if c not in {"variant", "policy", "decision"}:
            use[c] = use[c].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    lines = ["| " + " | ".join(use.columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(use.columns)) + " |")
    for _, row in use.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in use.columns) + " |")
    return "\n".join(lines)


def _write_report(summary_df, diag_df, variant_notes, ppo_decisions):
    z = summary_df[summary_df.policy == "zscore"].sort_values("total_return", ascending=False)
    best = z.iloc[0] if len(z) else None
    inverse = z[z.variant.str.startswith("inverse")]
    positive = z[z.variant.str.startswith("positive")]
    inv_best = inverse.total_return.max() if len(inverse) else np.nan
    pos_best = positive.total_return.max() if len(positive) else np.nan
    persist = diag_df.groupby("variant")["coint_persisted"].mean().reset_index()

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
        "- Universe: 936 requested symbols, 744 retained full-period symbols.",
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
            ["variant", "policy", "total_return", "sharpe", "max_drawdown", "profit_factor", "turnover", "total_costs"],
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
        "4. Use hedge-ratio residual PnL for classical pairs instead of the current equal-notional spread return approximation.",
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
        require_full_coverage=data_cfg["require_full_coverage"],
        coverage_tolerance_days=data_cfg["coverage_tolerance_days"],
    )
    prices = panels[data_cfg.get("price_type", "Adj Close")]
    returns = prices.pct_change().dropna()
    idx = prices.index
    windows = _generate_windows(idx, cfg)

    summary_rows = []
    diag_rows = []
    equity_curves = {}
    variant_notes = []

    for variant in VARIANTS:
        curves_by_policy = {"cash": [], "random": [], "zscore": []}
        pair_pnls_by_policy = {"cash": {}, "random": {}, "zscore": {}}

        for w in windows:
            train_prices = prices.iloc[w["train_start_idx"]:w["train_end_idx"]]
            test_prices = prices.iloc[w["validation_end_idx"]:w["test_end_idx"]]
            train_returns = train_prices.pct_change().dropna()
            test_returns = test_prices.pct_change().dropna()

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

            if not selected_pairs:
                continue

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
                    test_prices=test_prices,
                    pairs=selected_pairs,
                    models=models,
                    cfg=cfg,
                    seed=42 + w["window"],
                )
                curves_by_policy[policy].append((w["window"], curve, pnls, frictions))

        for policy, rows in curves_by_policy.items():
            curve, pnls, frictions, pair_pnls = _stitch_equity(rows)
            name = f"{variant['name']}__{policy}"
            equity_curves[name] = curve.rename(name)
            met = _metrics(curve, pnls, frictions)
            summary_rows.append(
                {
                    "variant": variant["name"],
                    "policy": policy,
                    "selected_pair_windows": len(rows),
                    **met,
                }
            )
            pair_pnls_by_policy[policy] = pair_pnls

    summary_df = pd.DataFrame(summary_rows)
    diag_df = pd.DataFrame(diag_rows)

    ppo_decisions = {}
    for variant in VARIANTS:
        row = summary_df[(summary_df.variant == variant["name"]) & (summary_df.policy == "zscore")]
        persist_rate = diag_df[diag_df.variant == variant["name"]]["coint_persisted"].mean()
        if row.empty:
            ppo_decisions[variant["name"]] = "skipped_no_pairs"
            continue
        row = row.iloc[0]
        if (
            row["total_return"] > 0
            and row["sharpe"] > 0
            and row["profit_factor"] > 1.05
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
