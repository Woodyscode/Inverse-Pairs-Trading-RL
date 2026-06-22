import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint
from utils.logger import get_logger


def _cointegration_pvalue(prices_df, a, b, min_obs):
    aligned = prices_df[[a, b]].dropna()
    if len(aligned) < min_obs:
        return None
    _, pval, _ = coint(aligned[a], aligned[b])
    return float(pval)


def find_strong_pairs(returns_df, prices_df, config):
    """
    FINAL VERSION:
    
    - Correlation on RETURNS
    - Volatility on RETURNS
    - Cointegration on PRICES (correct for pairs trading)
    """

    log = get_logger("PairSelector")

    pair_cfg = config["pairs"]

    # Shared config
    corr_thresh     = pair_cfg["correlation_threshold"]
    allow_pos       = pair_cfg["allow_positive"]
    allow_neg       = pair_cfg["allow_negative"]

    require_coint   = pair_cfg["use_cointegration"]
    p_value_thresh  = pair_cfg["cointegration_p_value"]
    min_obs         = pair_cfg["min_observations"]

    # Traditional config
    trad_min_vol    = pair_cfg.get("traditional_min_annual_vol", None)
    trad_max_vol    = pair_cfg.get("traditional_max_annual_vol", None)
    max_trad_pairs  = pair_cfg.get("max_traditional_pairs", None)

    # Inverse config
    inv_min_vol     = pair_cfg.get("inverse_min_annual_vol", None)
    inv_max_vol     = pair_cfg.get("inverse_max_annual_vol", None)
    max_inv_pairs   = pair_cfg.get("max_inverse_pairs", None)


    # ============================================================
    # 1) Compute correlation + volatility from RETURNS
    # ============================================================
    log.info("[PAIRS] Computing correlation (returns)...")
    corr = returns_df.corr()

    log.info("[PAIRS] Computing annualized vol (returns)...")
    vol = returns_df.std() * np.sqrt(252)

    cols = returns_df.columns
    n = len(cols)

    log.info(f"[PAIRS] Evaluating {n} assets ({n * (n - 1) // 2} pairs)...")

    traditional_pairs = []
    inverse_pairs = []


    # ============================================================
    # 2) Pair loop
    # ============================================================
    for a, b in combinations(cols, 2):

        rho = corr.loc[a, b]
        if np.isnan(rho):
            continue

        pair_returns = returns_df[[a, b]].dropna()
        if len(pair_returns) < min_obs:
            continue

        # Must exceed correlation threshold
        if abs(rho) < corr_thresh:
            continue

        # =======================================================
        # POSITIVE CORRELATION (Traditional)
        # =======================================================
        if rho > 0 and allow_pos:

            # Volatility screen
            if trad_min_vol is not None:
                if vol[a] < trad_min_vol or vol[b] < trad_min_vol:
                    continue

            if trad_max_vol is not None:
                if vol[a] > trad_max_vol or vol[b] > trad_max_vol:
                    continue

            # Correct cointegration: on PRICES
            if require_coint:
                try:
                    pval = _cointegration_pvalue(prices_df, a, b, min_obs)
                except Exception:
                    continue

                if pval is None or pval >= p_value_thresh:
                    continue

            traditional_pairs.append((a, b))


        # =======================================================
        # NEGATIVE CORRELATION (Inverse Pairs Trading)
        # =======================================================
        elif rho < 0 and allow_neg:

            if rho > -abs(corr_thresh):
                continue

            # Volatility screen (optional)
            if inv_min_vol is not None:
                if vol[a] < inv_min_vol or vol[b] < inv_min_vol:
                    continue
            if inv_max_vol is not None:
                if vol[a] > inv_max_vol or vol[b] > inv_max_vol:
                    continue

            # Cointegration on PRICES
            if require_coint:
                try:
                    pval = _cointegration_pvalue(prices_df, a, b, min_obs)
                except Exception:
                    continue

                if pval is None or pval >= p_value_thresh:
                    continue

            inverse_pairs.append((a, b))


    # ============================================================
    # 3) Apply max limits
    # ============================================================
    if max_trad_pairs is not None:
        traditional_pairs = traditional_pairs[:max_trad_pairs]

    if max_inv_pairs is not None:
        inverse_pairs = inverse_pairs[:max_inv_pairs]


    # ============================================================
    # Logging
    # ============================================================
    log.info(f"[PAIRS] Traditional pairs: {len(traditional_pairs)}")
    log.info(f"[PAIRS] Inverse pairs:     {len(inverse_pairs)}")

    return traditional_pairs, inverse_pairs
