import numpy as np

def build_targets(ohlcv, pairs, price_type="Adj Close"):
    """
    Create supervised learning targets for each pair:
        y[t] = next-step spread return
    Works well with ElasticNet.
    """

    close = ohlcv[price_type]
    targets = {}

    for (A, B) in pairs:
        spread = close[A] - close[B]
        spread_ret = spread.diff().fillna(0)

        # next-step prediction target
        y = spread_ret.shift(-1).fillna(0).values.astype(np.float32)

        targets[(A, B)] = y

    return targets
