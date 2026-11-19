import numpy as np
import pandas as pd

_EPS = 1e-12


# =====================================================================
#                                HELPERS
# =====================================================================

def _to_np(x):
    return np.asarray(x, dtype=float)


def _safe_std(x):
    x = _to_np(x)
    if len(x) <= 1:
        return 0.0
    return float(x.std() + _EPS)


# =====================================================================
#                         CORE PERFORMANCE METRICS
# =====================================================================

def calculate_cagr(values, years):
    if len(values) < 2:
        return 0.0
    start, end = float(values[0]), float(values[-1])
    if start <= 0:
        return 0.0
    return (end / start) ** (1 / years) - 1


def calculate_volatility(returns):
    return float(np.std(returns) * np.sqrt(252))


def calculate_sharpe(returns, rf=0.0):
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / 252
    vol = excess.std() + _EPS
    return float((excess.mean() / vol) * np.sqrt(252))


def calculate_sortino(returns, rf=0.0):
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / 252
    downside = excess[excess < 0]
    dd = np.sqrt(np.mean(np.minimum(excess, 0) ** 2))
    if dd == 0:
        return np.nan
    return float((excess.mean() / dd) * np.sqrt(252))


# =====================================================================
#                            DRAWDOWN METRICS
# =====================================================================

def calculate_max_drawdown(values):
    values = _to_np(values)
    peaks = np.maximum.accumulate(values)
    dd = (values - peaks) / peaks
    return float(abs(dd.min()))


def calculate_drawdown_duration(values):
    values = _to_np(values)
    peaks = np.maximum.accumulate(values)
    dd = values < peaks

    longest, current = 0, 0
    for flag in dd:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def calculate_ulcer_index(values):
    values = _to_np(values)
    peaks = np.maximum.accumulate(values)
    drawdowns = (values - peaks) / peaks
    return float(np.sqrt(np.mean(drawdowns**2)))


def calculate_cdar(returns, alpha=0.95):
    values = (1 + returns).cumprod()
    peaks = np.maximum.accumulate(values)
    dd = (values - peaks) / peaks

    cutoff = np.quantile(dd, 1 - alpha)
    tail = dd[dd <= cutoff]

    if len(tail) == 0:
        return 0.0
    return float(tail.mean())


def calculate_mar(cagr, max_dd):
    if max_dd == 0:
        return np.inf
    return float(cagr / max_dd)


def calculate_calmar_36m(values, dates):
    df = pd.Series(values, index=dates)
    end = df.index[-1]
    start = end - pd.DateOffset(months=36)
    sub = df[df.index >= start]

    if len(sub) < 60:
        return np.nan

    cagr = (sub.iloc[-1] / sub.iloc[0]) ** (12 / 36) - 1

    peak = sub.cummax()
    dd = (sub - peak) / peak
    max_dd = abs(dd.min())

    if max_dd == 0:
        return np.inf
    return float(cagr / max_dd)


# =====================================================================
#                        TRADE METRICS
# =====================================================================

def calculate_profit_factor(trades):
    if len(trades) == 0:
        return 0.0
    gp = sum(p for p in trades if p > 0)
    gl = abs(sum(p for p in trades if p < 0))
    if gl == 0:
        return np.inf if gp > 0 else np.nan
    return float(gp / gl)


def calculate_win_rate(trades):
    if len(trades) == 0:
        return 0.0
    wins = sum(1 for p in trades if p > 0)
    return wins / len(trades)


def calculate_payoff_ratio(trades):
    wins = [p for p in trades if p > 0]
    losses = [abs(p) for p in trades if p < 0]
    if not wins or not losses:
        return 0.0
    return float(np.mean(wins) / np.mean(losses))


def calculate_max_consecutive_losses(trades):
    streak = longest = 0
    for p in trades:
        if p < 0:
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0
    return longest


def calculate_gain_to_pain(returns):
    total_gain = returns[returns > 0].sum()
    pain = abs(returns[returns < 0].sum()) + _EPS
    return float(total_gain / pain)


def calculate_expectancy(win_rate, payoff_ratio):
    return float(win_rate * payoff_ratio - (1 - win_rate))


# =====================================================================
#                    BENCHMARK / FACTOR METRICS
# =====================================================================

def calculate_alpha_beta(strategy, benchmark):
    if len(strategy) < 5:
        return np.nan, np.nan
    df = pd.DataFrame({"s": strategy, "b": benchmark}).dropna()
    if len(df) < 10:
        return np.nan, np.nan
    cov = np.cov(df["s"], df["b"])[0, 1]
    var = df["b"].var()
    beta = cov / (var + _EPS)
    alpha = (df["s"].mean() - beta * df["b"].mean()) * 252
    return float(alpha), float(beta)


def calculate_tracking_error(strategy, benchmark):
    diff = _to_np(strategy) - _to_np(benchmark)
    return float(np.std(diff) * np.sqrt(252))


def calculate_information_ratio(strategy, benchmark):
    te = calculate_tracking_error(strategy, benchmark)
    if te == 0:
        return np.nan
    excess = _to_np(strategy) - _to_np(benchmark)
    return float(excess.mean() * 252 / te)


def calculate_capture_ratios(strategy, benchmark):
    df = pd.DataFrame({"s": strategy, "b": benchmark}).dropna()
    up = df[df["b"] > 0]
    down = df[df["b"] < 0]

    up_cap = up["s"].mean() / (up["b"].mean() + _EPS) if len(up) else np.nan
    down_cap = down["s"].mean() / (down["b"].mean() + _EPS) if len(down) else np.nan

    # ensure positive down capture ratio
    down_cap = abs(down_cap)

    return float(up_cap), float(down_cap)


# =====================================================================
#                     MONTHLY RETURNS (CALENDAR CORRECT)
# =====================================================================

def calculate_monthly_returns(equity_curve):
    ends = equity_curve.resample("ME").last().dropna()
    starts = ends.shift()
    starts.iloc[0] = equity_curve.iloc[0]
    monthly = ends / starts - 1
    return monthly.round(4).to_dict()


# =====================================================================
#                     MAIN METRIC AGGREGATOR
# =====================================================================

def compute_all_metrics(equity_curve, benchmark_curve, bar_pnls):

    values = equity_curve.values
    dates = equity_curve.index
    returns = equity_curve.pct_change().dropna()

    years = (dates[-1] - dates[0]).total_seconds() / (365.25 * 86400)

    # Core
    cagr = calculate_cagr(values, years)
    vol = calculate_volatility(returns.values)
    sharpe = calculate_sharpe(returns.values)
    sortino = calculate_sortino(returns.values)
    max_dd = calculate_max_drawdown(values)
    dd_duration = calculate_drawdown_duration(values)
    ulcer = calculate_ulcer_index(values)
    cdar = calculate_cdar(returns.values)

    mar = calculate_mar(cagr, max_dd)
    calmar36 = calculate_calmar_36m(values, dates)

    skew = float(returns.skew())
    kurt = float(returns.kurt())
    gpr = calculate_gain_to_pain(returns.values)

    # Monthly
    monthly_returns = calculate_monthly_returns(equity_curve)

    # Per-bar PnL statistics
    win_rate = calculate_win_rate(bar_pnls)
    payoff = calculate_payoff_ratio(bar_pnls)
    expectancy = float(np.mean(bar_pnls)) if len(bar_pnls) else np.nan
    profit_factor = calculate_profit_factor(bar_pnls)
    max_consec_losses = calculate_max_consecutive_losses(bar_pnls)

    # Benchmark
    if benchmark_curve is not None:
        benchmark_returns = benchmark_curve.reindex(dates).pct_change(fill_method=None)
        aligned = pd.concat([returns, benchmark_returns], axis=1, sort=False).dropna()
        strat_r, bench_r = aligned.iloc[:, 0].values, aligned.iloc[:, 1].values

        alpha, beta = calculate_alpha_beta(strat_r, bench_r)
        te = calculate_tracking_error(strat_r, bench_r)
        ir = calculate_information_ratio(strat_r, bench_r)
        up_cap, down_cap = calculate_capture_ratios(strat_r, bench_r)
    else:
        alpha = beta = te = ir = up_cap = down_cap = np.nan

    return {
        "final_value": float(values[-1]),
        "total_return": float(values[-1] / values[0] - 1),
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "drawdown_duration_bars": dd_duration,
        "ulcer_index": ulcer,
        "cdar": cdar,
        "mar": mar,
        "calmar_36m": calmar36,
        "skewness": skew,
        "kurtosis": kurt,
        "gain_to_pain": gpr,
        "monthly_returns": monthly_returns,

        # Per-bar PnL statistics
        "bar_win_rate": win_rate,
        "bar_payoff_ratio": payoff,
        "mean_bar_pnl": expectancy,
        "bar_profit_factor": profit_factor,
        "bar_max_consecutive_losses": max_consec_losses,

        # Benchmark
        "alpha": alpha,
        "beta": beta,
        "tracking_error": te,
        "information_ratio": ir,
        "up_capture_ratio": up_cap,
        "down_capture_ratio": down_cap,
    }
