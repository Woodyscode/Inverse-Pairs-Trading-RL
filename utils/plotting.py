import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# =============================================================================
#                              STYLE SETTINGS
# =============================================================================
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


# =============================================================================
#                          BASIC EQUITY CURVE
# =============================================================================
def plot_equity_curve(equity_curve: pd.Series, title="Strategy Equity Curve"):
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve.index, equity_curve.values, linewidth=2, label="Strategy")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
#                    EQUITY vs. SPY NORMALIZED COMPARISON
# =============================================================================
def plot_equity_vs_spy(strategy_curve: pd.Series,
                       spy_curve: pd.Series,
                       title="Strategy vs SPY Benchmark"):

    df = pd.DataFrame({
        "Strategy": strategy_curve / strategy_curve.iloc[0],
        "SPY": spy_curve / spy_curve.iloc[0]
    }).dropna()

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Strategy"], linewidth=2, label="Strategy")
    plt.plot(df.index, df["SPY"], linewidth=2, linestyle="--", label="SPY Benchmark")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Normalized Value (Start = 1.0)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
#                      DAILY RETURNS DISTRIBUTION
# =============================================================================
def plot_return_distribution(returns: pd.Series,
                             title="Daily Return Distribution"):

    plt.figure(figsize=(10, 5))
    sns.histplot(returns, bins=50, kde=True, stat="density")
    plt.title(title)
    plt.xlabel("Daily Return")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()


# =============================================================================
#                          MONTHLY RETURNS HEATMAP
# =============================================================================
def plot_monthly_heatmap(equity_curve):
    df = equity_curve.pct_change().dropna().to_frame("return")
    df["Year"] = df.index.year
    df["Month"] = df.index.month

    # FIX: pivot requires keyword arguments
    heatmap = df.pivot(
        index="Year",
        columns="Month",
        values="return"
    )

    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap,
        annot=True,
        fmt=".2%",
        cmap="RdYlGn",
        center=0
    )

    plt.title("Monthly Return Heatmap")
    plt.show()



# =============================================================================
#                            ALLOCATION PIE CHART
# =============================================================================
def plot_allocation_pie(allocation_dict):
    if not allocation_dict:
        print("No allocation to plot.")
        return

    keys = list(allocation_dict.keys())
    vals = [abs(v) for v in allocation_dict.values()]

    plt.figure(figsize=(8, 8))
    plt.pie(vals, labels=keys, autopct="%1.1f%%")
    plt.title("Final Allocation (Absolute Exposure)")
    plt.tight_layout()
    plt.show()


# =============================================================================
#                             NEW: DRAWDOWN CURVE
# =============================================================================
def plot_drawdown_curve(equity_curve: pd.Series,
                        title="Drawdown Curve (Underwater Chart)"):

    values = equity_curve.values
    peaks = np.maximum.accumulate(values)
    dd = (values - peaks) / peaks

    plt.figure(figsize=(12, 4))
    plt.fill_between(equity_curve.index, dd, color="red", alpha=0.4)
    plt.title(title)
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.show()


# =============================================================================
#                         NEW: ULCER INDEX PLOT
# =============================================================================
def plot_ulcer_index(equity_curve: pd.Series,
                     title="Ulcer Drawdown Contribution"):

    values = equity_curve.values
    peaks = np.maximum.accumulate(values)
    dd = (values - peaks) / peaks
    squared_dd = dd ** 2

    plt.figure(figsize=(12, 4))
    plt.plot(equity_curve.index, squared_dd, color="purple", label="Squared Drawdown")
    plt.title(title)
    plt.ylabel("Drawdown²")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
#                 NEW: CONDITIONAL DRAWDOWN-AT-RISK (CDaR)
# =============================================================================
def plot_cdar(equity_curve: pd.Series,
              alpha=0.95,
              title="Conditional Drawdown at Risk (CDaR)"):

    values = equity_curve.values
    peaks = np.maximum.accumulate(values)
    dd = (values - peaks) / peaks

    cutoff = np.quantile(dd, 1 - alpha)
    tail = dd[dd <= cutoff]

    plt.figure(figsize=(10, 5))
    sns.histplot(tail, bins=40, color="darkred", kde=True)
    plt.axvline(cutoff, color="black", linestyle="--", label=f"VaR Cutoff ({alpha})")
    plt.title(title)
    plt.xlabel("Drawdown")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
#                       NEW: ROLLING SHARPE / SORTINO
# =============================================================================
def plot_rolling_sharpe(returns: pd.Series,
                        window=63,
                        title="Rolling Sharpe Ratio (3-Month Window)"):

    roll_sharpe = returns.rolling(window).apply(
        lambda x: (x.mean() / (x.std() + 1e-12)) * np.sqrt(252)
    )

    plt.figure(figsize=(12, 4))
    plt.plot(roll_sharpe.index, roll_sharpe.values, color="blue")
    plt.title(title)
    plt.ylabel("Sharpe")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_rolling_sortino(returns: pd.Series,
                         window=63,
                         title="Rolling Sortino Ratio (3-Month Window)"):

    def roll_sort(x):
        downside = x[x < 0]
        dd = downside.std() + 1e-12
        return (x.mean() / dd) * np.sqrt(252)

    roll_s = returns.rolling(window).apply(roll_sort)

    plt.figure(figsize=(12, 4))
    plt.plot(roll_s.index, roll_s.values, color="green")
    plt.title(title)
    plt.ylabel("Sortino")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =============================================================================
#                     NEW: ROLLING REALIZED VOLATILITY
# =============================================================================
def plot_rolling_vol(returns: pd.Series,
                     window=63,
                     title="Rolling Volatility (3-Month Realized)"):

    roll_vol = returns.rolling(window).std() * np.sqrt(252)

    plt.figure(figsize=(12, 4))
    plt.plot(roll_vol.index, roll_vol.values, color="orange")
    plt.title(title)
    plt.ylabel("Volatility")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =============================================================================
#                   NEW: TRADE PNL DISTRIBUTION (DEEPER)
# =============================================================================
def plot_trade_pnl_distribution(trade_pnls,
                                title="Trade PnL Distribution"):

    trade_pnls = np.array(trade_pnls)

    plt.figure(figsize=(10, 5))
    sns.histplot(trade_pnls, bins=40, kde=True, color="steelblue")
    plt.axvline(np.mean(trade_pnls), color="black", linestyle="--", label="Mean PnL")
    plt.title(title)
    plt.xlabel("PnL")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
#                       NEW: GAIN-TO-PAIN RATIO PLOT
# =============================================================================
def plot_gain_to_pain(returns: pd.Series,
                      title="Gain-to-Pain Ratio Breakdown"):

    gains = returns[returns > 0].sum()
    pain = abs(returns[returns < 0].sum())

    plt.figure(figsize=(8, 5))
    plt.bar(["Gains", "Pain"], [gains, pain], color=["green", "red"])
    plt.title(title)
    plt.ylabel("Total")
    plt.tight_layout()
    plt.show()
