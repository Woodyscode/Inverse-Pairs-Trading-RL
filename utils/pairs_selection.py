import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint
from utils.logger import log


def find_inverse_pairs(
    df,
    corr_thresh=-0.8,
    p_value_thresh=0.05,
    min_obs=300
):
    """
    Identify inverse cointegrated pairs from return matrix.

    Steps:
    1. Negative correlation filter
    2. Cointegration test (Engle-Granger)
    """
    log("[PAIRS] Computing correlation matrix...")
    corr = df.corr()

    n = len(df.columns)
    log(f"[PAIRS] Running pair search on {n} assets (~{n*(n-1)/2:.0f} pairs)")

    selected = []

    for a, b in combinations(df.columns, 2):

        # correlation filter first
        if corr.loc[a, b] > corr_thresh:
            continue

        # minimum length requirement
        if len(df[a]) < min_obs or len(df[b]) < min_obs:
            continue

        try:
            _, p, _ = coint(df[a], df[b])

            if p < p_value_thresh:
                selected.append((a, b))

        except Exception as e:
            log(f"[PAIRS] Error testing {a}-{b}: {e}", level="WARN")

    log(f"[PAIRS] Selected {len(selected)} inverse cointegrated pairs.")
    return selected
