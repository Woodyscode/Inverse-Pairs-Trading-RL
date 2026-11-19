import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.plotting import (
    plot_equity_curve,
    plot_equity_vs_spy,
    plot_return_distribution,
    plot_monthly_heatmap,
    plot_allocation_pie,
    plot_drawdown_curve,
    plot_ulcer_index,
    plot_cdar,
    plot_rolling_sharpe,
    plot_rolling_sortino,
    plot_rolling_vol,
    plot_trade_pnl_distribution,
    plot_gain_to_pain,
)

from backtest.metrics import compute_all_metrics

logger = get_logger("Backtester")


# ======================================================================
#                             BACKTESTER
# ======================================================================
class Backtester:

    def __init__(self, agent, env, config):
        self.agent = agent
        self.env = env
        self.config = config

        if not hasattr(env, "price_index"):
            raise ValueError("[Backtester] Environment missing price_index")

        self.price_index = env.price_index


    # ==================================================================
    def run(self, return_curve=False):
        logger.info("Starting backtest evaluation...")

        obs, info = self.env.reset()

        portfolio_vals = [self.env.portfolio_value]
        bar_pnls = []
        friction_rows = []
        start_idx = min(self.env.current_step, len(self.price_index) - 1)
        dates = [self.price_index[start_idx]]

        terminated = False
        truncated = False

        # ===============================================================
        # MAIN LOOP
        # ===============================================================
        while not (terminated or truncated):

            action = self.agent.predict(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Portfolio value
            pv = info.get("portfolio_value", None)
            if pv is None:
                raise RuntimeError("[Backtester] Env missing portfolio_value in info.")

            if info.get("has_transition", True):
                portfolio_vals.append(pv)
                bar_pnls.append(info.get("pnl", 0.0))

                step_idx = min(self.env.current_step, len(self.price_index) - 1)
                dates.append(self.price_index[step_idx])
                friction_rows.append({
                    "date": self.price_index[step_idx],
                    "turnover": info.get("turnover", 0.0),
                    "transaction_costs": info.get("transaction_costs", 0.0),
                    "carry_costs": info.get("carry_costs", 0.0),
                    "borrow_cost": info.get("borrow_cost", 0.0),
                    "financing_cost": info.get("financing_cost", 0.0),
                    "cash_credit": info.get("cash_credit", 0.0),
                    "gross_exposure": info["gross_exposure"],
                    "active_positions": info["active_positions"],
                    "gross_pnl": info["gross_pnl"],
                    "net_pnl": info["pnl"],
                })

        # ===============================================================
        # Equity Curve
        # ===============================================================
        equity_curve = pd.Series(
            portfolio_vals,
            index=pd.to_datetime(dates),
            name="Strategy"
        )

        logger.info(f"Final portfolio value: {equity_curve.iloc[-1]:,.2f}")

        # ===============================================================
        # Benchmark (SPY)
        # ===============================================================
        spy_curve = self._construct_spy_curve(equity_curve)

        # ===============================================================
        # Metrics
        # ===============================================================
        metrics = compute_all_metrics(
            equity_curve=equity_curve,
            benchmark_curve=spy_curve,
            bar_pnls=bar_pnls,
        )

        # ===============================================================
        # Plotting
        # ===============================================================
        self._run_plots(equity_curve, spy_curve, bar_pnls)

        # ===============================================================
        # Print summary
        # ===============================================================
        self._print_metrics(metrics)

        if return_curve:
            return metrics, equity_curve

        pair_pnls = {}
        for pair, values in getattr(self.env, "per_pair_pnls", {}).items():
            pair_key = f"{pair[0]}-{pair[1]}"
            pair_pnls[pair_key] = [float(v) for v in values]

        return {
            "equity_curve": equity_curve,
            "spy_curve": spy_curve,
            "metrics": metrics,
            "bar_pnls": bar_pnls,
            "frictions": pd.DataFrame(friction_rows),
            "pair_pnls": pair_pnls,
        }


    # ==================================================================
    # Construct SPY benchmark curve
    # ==================================================================
    def _construct_spy_curve(self, equity_curve):

        if "benchmark_curve" not in self.config:
            return pd.Series(1.0, index=equity_curve.index)

        spy = (
            self.config["benchmark_curve"]
            .copy()
            .reindex(equity_curve.index)
            .ffill()
        )

        # Reduce DataFrame → Series
        if isinstance(spy, pd.DataFrame):
            spy = spy.iloc[:, 0]

        if spy.iloc[0] <= 0:
            spy = pd.Series(1.0, index=equity_curve.index)
        else:
            spy = spy / spy.iloc[0]

        return spy.astype(float)


    # ==================================================================
    # Plot handling
    # ==================================================================
    def _run_plots(self, equity_curve, spy_curve, bar_pnls):

        cfg = self.config.get("plotting", {})

        if cfg.get("equity_curve", True):
            plot_equity_curve(equity_curve)

        if cfg.get("compare_spy", True):
            plot_equity_vs_spy(equity_curve, spy_curve)

        if cfg.get("return_distribution", True):
            plot_return_distribution(equity_curve.pct_change().dropna())

        if cfg.get("monthly_heatmap", True):
            plot_monthly_heatmap(equity_curve)  # FIXED pivot()

        # Allocation Pie
        if cfg.get("final_allocation", True) and hasattr(self.env, "finalize_positions"):
            alloc = self.env.finalize_positions()
            plot_allocation_pie(alloc)

        # Advanced Plots
        rets = equity_curve.pct_change().dropna()

        if cfg.get("drawdown_curve", True):          plot_drawdown_curve(equity_curve)
        if cfg.get("ulcer_index_curve", True):       plot_ulcer_index(equity_curve)
        if cfg.get("cdar_plot", True):               plot_cdar(equity_curve)
        if cfg.get("rolling_sharpe", True):          plot_rolling_sharpe(rets)
        if cfg.get("rolling_sortino", True):         plot_rolling_sortino(rets)
        if cfg.get("rolling_vol", True):             plot_rolling_vol(rets)
        if cfg.get("trade_pnl_distribution", True):  plot_trade_pnl_distribution(bar_pnls)
        if cfg.get("gain_to_pain", True):            plot_gain_to_pain(rets)


    # ==================================================================
    # Print metrics
    # ==================================================================
    def _print_metrics(self, m):

        logger.info(
            "\n==================== BACKTEST METRICS ====================\n"
            f"Final Value:          {m['final_value']:.2f}\n"
            f"Total Return:         {m['total_return']*100:.2f}%\n"
            f"CAGR:                 {m['cagr']*100:.2f}%\n"
            f"Sharpe Ratio:         {m['sharpe']:.3f}\n"
            f"Sortino Ratio:        {m['sortino']:.3f}\n"
            f"Volatility:           {m['volatility']*100:.2f}%\n"
            f"Max Drawdown:         {m['max_drawdown']*100:.2f}%\n"
            f"MAR:                  {m['mar']:.3f}\n"
            "\n---------------- BENCHMARK METRICS -----------------------\n"
            f"Alpha:                {m['alpha']:.4f}\n"
            f"Beta:                 {m['beta']:.4f}\n"
            f"Tracking Error:       {m['tracking_error']:.4f}\n"
            f"Information Ratio:    {m['information_ratio']:.4f}\n"
            "==========================================================\n"
        )
