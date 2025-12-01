import pandas as pd
import numpy as np


def compute_log_returns(df):
    """
    Compute log returns, which behave better statistically than simple returns.
    """
    log_r = np.log(df / df.shift(1))
    log_r = log_r.replace([np.inf, -np.inf], np.nan).dropna()
    return log_r


def normalize_returns(df):
    """
    Normalize returns for training stability.
    This is optional but helps PPO learn faster.
    """
    return (df - df.mean()) / (df.std() + 1e-8)
