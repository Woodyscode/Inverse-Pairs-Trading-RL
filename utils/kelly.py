import numpy as np


def fractional_kelly(expected_return, variance, fraction=0.25):
    """
    Kelly fraction: k = μ / σ²
    We scale it by the fraction (0.1–0.3 typical for hedge funds).
    """
    if variance <= 0:
        return 0

    k = expected_return / variance
    return max(k * fraction, 0)
