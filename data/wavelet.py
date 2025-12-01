import pywt
import pandas as pd
import numpy as np


def wavelet_denoise(series, wavelet="db4", level=3):
    """Denoise a single return series using wavelet decomposition.
       Only zero-out the first detail coefficient (D1)."""

    coeffs = pywt.wavedec(series, wavelet, level=level)

    # Zero ONLY the first detail coefficient (D1)
    if len(coeffs) > 1:
        coeffs[1] = np.zeros_like(coeffs[1])

    reconstructed = pywt.waverec(coeffs, wavelet)

    # Align length
    if len(reconstructed) != len(series):
        reconstructed = reconstructed[:len(series)]

    return pd.Series(reconstructed, index=series.index)



def denoise_dataframe(df):
    """Apply wavelet denoising to every column."""
    clean = df.copy()
    for col in df.columns:
        clean[col] = wavelet_denoise(df[col])
    return clean

