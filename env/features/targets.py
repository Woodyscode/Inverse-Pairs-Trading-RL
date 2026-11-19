import numpy as np


def build_targets(ohlcv, pairs, price_type="Adj Close", pair_models=None):
    """Next-bar PnL per dollar of gross exposure to (one A, minus beta B)."""
    close = ohlcv[price_type]
    targets = {}
    for a, b in pairs:
        model = (pair_models or {}).get((a, b), {})
        beta = model.get("hedge_ratio", 1.0)
        spread = close[a] - beta * close[b]
        gross = close[a] + abs(beta) * close[b]
        targets[a, b] = (spread.diff().shift(-1) / gross).to_numpy(dtype=np.float32)
    return targets
