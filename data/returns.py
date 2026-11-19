import pandas as pd
import numpy as np


def compute_log_returns(df, keep_calendar=False):
    """
    Compute log returns, which behave better statistically than simple returns.
    """
    log_r = np.log(df / df.shift(1))
    log_r = log_r.replace([np.inf, -np.inf], np.nan)
    return log_r.iloc[1:] if keep_calendar else log_r.dropna()


def normalize_returns(df):
    """
    Normalize returns for training stability.
    This is optional but helps PPO learn faster.
    """
    past = df.shift().expanding(min_periods=2)
    return (df - past.mean()) / (past.std() + 1e-8)
