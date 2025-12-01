import numpy as np


def sharpe_ratio(returns, freq=252):
    """Annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0
    mean = np.mean(returns)
    std = np.std(returns)
    if std == 0:
        return 0
    return (mean / std) * np.sqrt(freq)


import numpy as np
import pandas as pd


# =============================================================
#                CORE HELPER FUNCTIONS
# =============================================================

def _compute_drawdowns(values):
    """Return max drawdown and duration."""
    peak = np.maximum.accumulate(values)
    drawdowns = (values - peak) / peak

    max_dd = np.min(drawdowns)

    # Drawdown duration
    dd = drawdowns < 0
    duration = 0
    max_duration = 0

    for x in dd:
        if x:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0

    return abs(max_dd), max_duration


def _annualized_return(total_return, years):
    return (1 + total_return) ** (1 / years) - 1


# =============================================================
#                MAIN METRICS FUNCTION
# =============================================================

def compute_all_metrics(
    portfolio_values,
    trade_pnls,
    start_date,
    trading_days=252
):
    """
    Computes institutional-quality metrics for backtesting.
    """

    pv = np.array(portfolio_values, dtype=float)

    # ---------------------------------------------------------
    # Basic returns
    # ---------------------------------------------------------
    total_return = pv[-1] / pv[0] - 1
    years = len(pv) / trading_days
    cagr = _annualized_return(total_return, years)

    # Daily returns
    rets = np.diff(pv) / pv[:-1]
    rets = np.nan_to_num(rets)

    # ---------------------------------------------------------
    # Risk metrics
    # ---------------------------------------------------------
    vol = np.std(rets) * np.sqrt(trading_days)
    sharpe = np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(trading_days)

    downside = rets[rets < 0]
    sortino = (
        np.mean(rets) /
        (np.std(downside) + 1e-9)
        * np.sqrt(trading_days)
    )

    max_dd, dd_duration = _compute_drawdowns(pv)
    calmar = cagr / (max_dd + 1e-9)
    mar = sharpe / (abs(max_dd) + 1e-9)

    # ---------------------------------------------------------
    # PnL Metrics
    # ---------------------------------------------------------
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]

    win_rate = len(wins) / (len(trade_pnls) + 1e-9)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1
    profit_factor = gross_profit / (gross_loss + 1e-9)

    # ---------------------------------------------------------
    # Omega ratio
    # ---------------------------------------------------------
    threshold = 0
    positive = np.sum(rets[rets > threshold] - threshold)
    negative = abs(np.sum(rets[rets < threshold] - threshold))
    omega = positive / (negative + 1e-9)

    # ---------------------------------------------------------
    # Package metrics
    # ---------------------------------------------------------
    metrics = {
        "final_value": pv[-1],
        "total_return": total_return,
        "cagr": cagr,

        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "mar": mar,

        "max_drawdown": max_dd,
        "drawdown_duration_days": dd_duration,
        "volatility": vol,

        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "profit_factor": profit_factor,

        "omega": omega,
    }

    return metrics

