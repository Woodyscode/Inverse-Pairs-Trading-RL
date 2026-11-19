import numpy as np
import pandas as pd
import pywt


def wavelet_denoise(series, wavelet="db4", level=3):
    """
    Full-series denoising helper kept for offline diagnostics.

    Do not use this for research features unless the full series is strictly
    in-sample. The main pipeline uses causal_wavelet_denoise instead.
    """
    arr = np.asarray(series, dtype=float).copy()
    coeffs = pywt.wavedec(arr, wavelet, level=level)

    if len(coeffs) > 1:
        coeffs[1] = np.zeros_like(coeffs[1])

    reconstructed = pywt.waverec(coeffs, wavelet)

    if len(reconstructed) != len(series):
        reconstructed = reconstructed[: len(series)]

    return pd.Series(reconstructed, index=series.index)


def donoho_johnstone_endpoint(windows, wavelet="db4", level=1):
    """VisuShrink using each complete trailing window's own finest-detail MAD."""
    windows = np.asarray(windows, dtype=float)
    if windows.ndim != 2 or not 1 <= level <= pywt.dwt_max_level(
        windows.shape[1], wavelet
    ):
        raise ValueError("Choose a decomposition level supported by the window length")
    complete = np.isfinite(windows).all(axis=1)
    endpoints = np.full(len(windows), np.nan)
    if complete.any():
        approx, *details = pywt.wavedec(
            windows[complete], wavelet, level=level, mode="symmetric"
        )
        fine = details[-1]
        median = np.median(fine, axis=1, keepdims=True)
        sigma = np.median(abs(fine - median), axis=1, keepdims=True) / 0.67448975
        threshold = sigma * np.sqrt(2 * np.log(windows.shape[1]))
        shrunk = [np.sign(d) * np.maximum(abs(d) - threshold, 0) for d in details]
        endpoints[complete] = pywt.waverec(
            [approx, *shrunk], wavelet, mode="symmetric"
        )[:, windows.shape[1] - 1]
    return endpoints


def causal_wavelet_denoise(
    series, wavelet="db4", level=3, window=64, threshold_method="legacy"
):
    """
    Causal db4 denoising. Each output at t is computed only from the trailing
    window ending at t, and only the final reconstructed value is retained.
    """
    values = pd.Series(series).astype(float)
    if threshold_method == "donoho_johnstone":
        out = values.copy()
        if len(values) >= window:
            windows = np.lib.stride_tricks.sliding_window_view(
                values.to_numpy(), window
            )
            out.iloc[window - 1 :] = donoho_johnstone_endpoint(windows, wavelet, level)
        return out
    if threshold_method != "legacy":
        raise ValueError(f"Unknown wavelet threshold method: {threshold_method}")
    out = pd.Series(index=values.index, dtype=float)

    min_len = max(window, pywt.Wavelet(wavelet).dec_len * (level + 1))

    for i in range(len(values)):
        start = max(0, i - window + 1)
        trailing = values.iloc[start : i + 1].dropna()

        if len(trailing) < min_len:
            out.iloc[i] = values.iloc[i]
            continue

        out.iloc[i] = wavelet_denoise(trailing, wavelet=wavelet, level=level).iloc[-1]

    return out


def denoise_dataframe(df):
    clean = df.copy()
    for col in df.columns:
        clean[col] = wavelet_denoise(df[col])
    return clean


def causal_denoise_dataframe(df, wavelet="db4", level=3, window=64):
    clean = df.copy()
    for col in df.columns:
        clean[col] = causal_wavelet_denoise(
            df[col],
            wavelet=wavelet,
            level=level,
            window=window,
        )
    return clean
