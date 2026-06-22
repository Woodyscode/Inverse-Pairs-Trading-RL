import pywt
import pandas as pd
import numpy as np


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
        reconstructed = reconstructed[:len(series)]

    return pd.Series(reconstructed, index=series.index)


def causal_wavelet_denoise(series, wavelet="db4", level=3, window=64):
    """
    Causal db4 denoising. Each output at t is computed only from the trailing
    window ending at t, and only the final reconstructed value is retained.
    """
    values = pd.Series(series).astype(float)
    out = pd.Series(index=values.index, dtype=float)

    min_len = max(window, pywt.Wavelet(wavelet).dec_len * (level + 1))

    for i in range(len(values)):
        start = max(0, i - window + 1)
        trailing = values.iloc[start:i + 1].dropna()

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
