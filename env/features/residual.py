"""Causal log-spread state and purged supervised forecasts for the control study."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBRegressor


def residual_state(prices, volume, train_end):
    """Use an expanding log-price hedge through training, then freeze its slope."""
    log_a, log_b = np.log(prices).to_numpy().T
    a, b = pd.Series(log_a), pd.Series(log_b)
    beta = a.expanding(60).cov(b) / b.expanding(60).var()
    beta.iloc[train_end:] = beta.iloc[train_end - 1]
    rolling_beta = a.rolling(60).cov(b) / b.rolling(60).var()
    relative_volume = np.log((volume + 1) / (volume.rolling(20).mean() + 1)).mean(
        axis=1
    )
    rows = []
    last_z = np.nan
    adf = np.nan
    for t in range(len(a)):
        h = beta.iat[t]
        if t < 119 or not np.isfinite(h):
            rows.append([np.nan] * 8)
            continue
        # Use a single slope over the whole lookback, so refits do not create spread jumps.
        spread = log_a[t - 119 : t + 1] - h * log_b[t - 119 : t + 1]
        recent = spread[-30:]
        z = (recent[-1] - recent.mean()) / max(recent.std(), 1e-9)
        x = spread[:-1] - spread[:-1].mean()
        speed = -np.dot(x, np.diff(spread)) / max(np.dot(x, x), 1e-12)
        if (t - 119) % 20 == 0:
            adf = (
                adfuller(spread, maxlag=1, autolag=None)[1]
                if spread.std() > 1e-9
                else 1.0
            )
        vol = np.std(np.diff(spread[-31:])) / (1 + abs(h))
        rows.append(
            [
                z,
                z - last_z,
                vol,
                speed,
                adf,
                rolling_beta.iat[t],
                h,
                relative_volume.iloc[t],
            ]
        )
        last_z = z
    state = pd.DataFrame(
        rows,
        index=prices.index,
        columns=[
            "z",
            "dz",
            "vol",
            "reversion_speed",
            "adf_p",
            "rolling_beta",
            "beta",
            "relative_volume",
        ],
    )
    # Signal at s, execution at s+1, mark at s+2. Target is the executable basket return.
    rets = prices.pct_change(fill_method=None).shift(-2).to_numpy()
    target = (rets[:, 0] - beta.to_numpy() * rets[:, 1]) / (1 + beta.abs().to_numpy())
    return state, target


def alpha_forecasts(state, target, train_end, first=256, block=63):
    """Expanding-block out-of-fold training forecasts; frozen models for validation/test."""
    forecasts = np.full((len(state), 2), np.nan)
    features = state.to_numpy()
    # Predictions for the first validation order use the last training signal row.
    freeze = train_end - 1
    for start in [*range(first, freeze, block), freeze]:
        known = np.arange(start - 1)  # target at i is observable at i+2 <= start
        known = known[
            np.isfinite(features[known]).all(axis=1) & np.isfinite(target[known])
        ]
        if len(known) < 32:
            raise ValueError("Insufficient matured training labels for alpha")
        end = len(state) if start == freeze else min(start + block, freeze)
        models = (
            make_pipeline(StandardScaler(), Ridge(alpha=10)),
            XGBRegressor(
                n_estimators=40,
                max_depth=2,
                learning_rate=0.05,
                n_jobs=1,
                random_state=42,
                objective="reg:squarederror",
            ),
        )
        for k, model in enumerate(models):
            model.fit(features[known], target[known] * 10000)
            forecasts[start:end, k] = model.predict(features[start:end]) / 10000
    return forecasts
