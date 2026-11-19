from itertools import combinations

import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.stats.multitest import multipletests


def select_pair_models(returns_df, prices_df, config):
    """Select and fit price-level A = intercept + beta * B on one training fold."""
    cfg = config["pairs"]
    min_obs = cfg["min_observations"]
    corr = returns_df.corr(min_periods=min_obs)
    vol = returns_df.std() * np.sqrt(252)
    candidates = {"traditional": [], "inverse": []}
    for a, b in combinations(sorted(returns_df.columns), 2):
        rho = corr.at[a, b]
        if not np.isfinite(rho) or rho == 0 or abs(rho) < cfg["correlation_threshold"]:
            continue
        kind = "traditional" if rho > 0 else "inverse"
        if not cfg["allow_positive" if rho > 0 else "allow_negative"]:
            continue
        lower = cfg.get(f"{kind}_min_annual_vol")
        upper = cfg.get(f"{kind}_max_annual_vol")
        if lower is not None and min(vol[a], vol[b]) < lower:
            continue
        if upper is not None and max(vol[a], vol[b]) > upper:
            continue
        aligned = prices_df[[a, b]].dropna()
        if len(aligned) < min_obs or aligned[b].nunique() < 2:
            continue
        pvalue = None
        if cfg["use_cointegration"]:
            try:
                pvalue = float(coint(aligned[a], aligned[b])[1])
            except (ValueError, np.linalg.LinAlgError):
                pvalue = 1.0
            if not np.isfinite(pvalue):
                pvalue = 1.0
        beta, intercept = np.polyfit(aligned[b], aligned[a], 1)
        candidates[kind].append({
            "pair": (a, b), "type": kind, "correlation": float(rho),
            "observations": len(aligned), "coint_pvalue": pvalue,
            "hedge_ratio": float(beta), "intercept": float(intercept),
        })
    if cfg["use_cointegration"]:
        tested = candidates["traditional"] + candidates["inverse"]
        method = cfg.get("multiple_testing", "none")
        raw = [m["coint_pvalue"] for m in tested]
        adjusted = multipletests(raw, method=method)[1] if tested and method != "none" else raw
        for model, pvalue in zip(tested, adjusted):
            model["screened_tests"] = len(tested)
            model["adjusted_pvalue"] = float(pvalue)
        for kind, rows in candidates.items():
            candidates[kind] = [m for m in rows if m["adjusted_pvalue"] < cfg["cointegration_p_value"]]
    selected = []
    for kind, rows in candidates.items():
        rows.sort(key=lambda row: (-abs(row["correlation"]), row["pair"]))
        selected.extend(rows[:cfg.get(f"max_{kind}_pairs")])
    return selected


def find_strong_pairs(returns_df, prices_df, config):
    selected = select_pair_models(returns_df, prices_df, config)
    return tuple([m["pair"] for m in selected if m["type"] == kind]
                 for kind in ("traditional", "inverse"))
