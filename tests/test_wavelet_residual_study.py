from types import SimpleNamespace

import numpy as np
import pandas as pd
import pywt

from data.wavelet import causal_wavelet_denoise, donoho_johnstone_endpoint
from experiments.forecasting.wavelet_residual_study import transforms


def test_universal_threshold_matches_reference_at_trade_time():
    series = pd.Series(np.random.default_rng(3).normal(size=90))
    result = causal_wavelet_denoise(series, "db4", 1, 64, "donoho_johnstone")
    approx, detail = pywt.dwt(series.iloc[-64:].to_numpy(copy=True), "db4", mode="symmetric")
    mad = np.median(abs(detail - np.median(detail)))
    expected = pywt.idwt(
        approx,
        pywt.threshold(detail, mad / 0.67448975 * np.sqrt(2 * np.log(64)), mode="soft"),
        "db4",
    )
    np.testing.assert_allclose(result.iloc[-1], expected[-1])
    np.testing.assert_allclose(result.iloc[:63], series.iloc[:63])


def test_shrinkage_preserves_constant_and_requires_contiguous_observations():
    windows = np.ones((3, 32))
    windows[1, 10] = np.nan
    windows[2] += np.tile([0.04, -0.04], 16)
    values = donoho_johnstone_endpoint(windows)
    np.testing.assert_allclose(values[0], 1)
    assert np.isnan(values[1])
    assert abs(values[2] - 1) < 0.04


def test_daily_cutoff_and_opening_window_exclude_future_prices():
    rng = np.random.default_rng(42)

    def tape():
        return SimpleNamespace(
            total=rng.normal(0, 0.01, 130),
            open=np.ones(130) * 100,
            price=100 + rng.normal(size=(130, 100)),
            overnight=np.zeros(130),
        )

    tapes = {"A": tape(), "B": tape()}
    row = SimpleNamespace(a="A", b="B", beta=-1.2, intercept=0.0001)
    before = transforms(tapes, row)
    tapes["A"].total[110:] += 0.5
    tapes["B"].price[110, 60:] *= 2
    tapes["B"].price[111:] *= 2
    after = transforms(tapes, row)
    pd.testing.assert_frame_equal(before.iloc[:111], after.iloc[:111])
    assert np.isfinite(before.db4_daily_z.iloc[110])
    assert np.isfinite(before.db4_opening_z.iloc[110])
    tapes["B"].price[110, 40] = np.nan
    assert np.isnan(transforms(tapes, row).db4_opening_residual.iloc[110])
