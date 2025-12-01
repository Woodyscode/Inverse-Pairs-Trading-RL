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
#                          BACKTESTER CLASS
# ======================================================================
class Backtester:
    """
    Clean institutional backtester.
    Walk-forward safe.
    PPO-safe.
    """

    def __init__(self, agent, env, config):
        self.agent = agent
        self.env = env
        self.config = config

        if not hasattr(env, "price_index"):
            raise AttributeError(
                "[Backtester] Env must define env.price_index (DatetimeIndex)."
            )

        self.price_index = env.price_index


    # ==================================================================
    #                           RUN BACKTEST
    # ==================================================================
    def run(self, return_curve=False):

        logger.info("[BT] Starting backtest...")

        obs, info = self.env.reset()

        portfolio_vals = []
        dates = []
        trade_pnls = []

        terminated, truncated = False, False

        while not (terminated or truncated):

            # ----------------------------------------------------------
            # Prediction
            # ----------------------------------------------------------
            action = self.agent.predict(obs)

            # convert torch → numpy
            if hasattr(action, "detach"):
                action = action.detach().cpu().numpy()
            elif isinstance(action, list):
                action = np.array(action, dtype=np.float32)

            # ensure correct shape
            if not isinstance(action, np.ndarray):
                raise TypeError("[BT] Agent.predict() must return numpy array.")

            # ----------------------------------------------------------
            # Step environment
            # ----------------------------------------------------------
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Collect portfolio value
            pv = float(info.get("portfolio_value", None))
            portfolio_vals.append(pv)

            # Map step → date
            step_idx = min(self.env.current_step, len(self.price_index)-1)
            dates.append(self.price_index[step_idx])

            # Optional trade-level PnL breakdown
            if "pnl_breakdown" in info and isinstance(info["pnl_breakdown"], dict):
                trade_pnls.extend(list(info["pnl_breakdown"].values()))

        # ==================================================================
        #                          EQUITY CURVE
        # ==================================================================
        dates = pd.to_datetime(dates)

        equity_curve = pd.Series(
            data=np.array(portfolio_vals, dtype=float),
            index=dates,
            name="Strategy"
        )

        equity_curve = equity_curve[~equity_curve.index.duplicated()]

        logger.info(f"[BT] Final portfolio value = {equity_curve.iloc[-1]:,.2f}")

        # ==================================================================
        #                              BENCHMARK
        # ==================================================================
        spy_curve = self._construct_spy_curve(equity_curve)

        # ==================================================================
        #                              METRICS
        # ==================================================================
        metrics = compute_all_metrics(
            equity_curve=equity_curve,
            benchmark_curve=spy_curve,
            trade_pnls=trade_pnls,
        )

        # ==================================================================
        #                              PLOTTING
        # ==================================================================
        self._run_plots(equity_curve, spy_curve, trade_pnls)

        # ==================================================================
        #                              SUMMARY
        # ==================================================================
        self._print_metrics(metrics)

        if return_curve:
            return metrics, equity_curve

        return {
            "equity_curve": equity_curve,
            "spy_curve": spy_curve,
            "metrics": metrics,
        }


    # ==================================================================
    #                            SPY CURVE
    # ==================================================================
    def _construct_spy_curve(self, equity_curve):

        if "benchmark_curve" in self.config:
            spy = self.config["benchmark_curve"].copy().reindex(equity_curve.index).ffill()

            if isinstance(spy, pd.DataFrame):
                spy = spy.iloc[:, 0]

            if spy.isna().any():
                spy = spy.fillna(method="ffill")

            if spy.iloc[0] <= 0:
                logger.warning("[BT] SPY invalid start — using flat benchmark.")
                spy = pd.Series(1.0, index=equity_curve.index)
            else:
                spy = spy / spy.iloc[0]

        else:
            spy = pd.Series(1.0, index=equity_curve.index)

        return spy.astype(float)


    # ==================================================================
    #                         PLOTTING SUITE
    # ==================================================================
    def _run_plots(self, equity_curve, spy_curve, trade_pnls):

        cfg = self.config.get("plotting", {})

        rets = equity_curve.pct_change().dropna()

        if cfg.get("equity_curve", False):
            plot_equity_curve(equity_curve)

        if cfg.get("compare_spy", False):
            plot_equity_vs_spy(equity_curve, spy_curve)

        if cfg.get("return_distribution", False):
            plot_return_distribution(rets)

        if cfg.get("monthly_heatmap", False):
            plot_monthly_heatmap(equity_curve)

        if cfg.get("final_allocation", False) and hasattr(self.env, "finalize_positions"):
            plot_allocation_pie(self.env.finalize_positions())

        # --- Advanced ---
        if cfg.get("drawdown_curve", True):
            plot_drawdown_curve(equity_curve)

        if cfg.get("ulcer_index_curve", True):
            plot_ulcer_index(equity_curve)

        if cfg.get("cdar_plot", True):
            plot_cdar(equity_curve)

        if cfg.get("rolling_sharpe", True):
            plot_rolling_sharpe(rets)

        if cfg.get("rolling_sortino", True):
            plot_rolling_sortino(rets)

        if cfg.get("rolling_vol", True):
            plot_rolling_vol(rets)

        if cfg.get("trade_pnl_distribution", True):
            plot_trade_pnl_distribution(trade_pnls)

        if cfg.get("gain_to_pain", True):
            plot_gain_to_pain(rets)


    # ==================================================================
    #                          PRINT METRICS
    # ==================================================================
    def _print_metrics(self, m):

        logger.info(
            "\n=================== BACKTEST PERFORMANCE ===================\n"
            f"Final Value:          {m['final_value']:.2f}\n"
            f"Total Return:         {m['total_return']*100:.2f}%\n"
            f"CAGR:                 {m['cagr']*100:.2f}%\n"
            f"Sharpe Ratio:         {m['sharpe']:.3f}\n"
            f"Sortino Ratio:        {m['sortino']:.3f}\n"
            f"Volatility:           {m['volatility']*100:.2f}%\n"
            f"Max Drawdown:         {m['max_drawdown']*100:.2f}%\n"
            f"MAR Ratio:            {m['mar']:.3f}\n"
            f"Calmar (36m):         {m['calmar_36m']:.3f}\n"
            f"Skewness:             {m['skewness']:.3f}\n"
            f"Kurtosis:             {m['kurtosis']:.3f}\n"
            f"Omega Ratio:          {m['omega']:.3f}\n"
            f"Win Rate:             {m['win_rate']*100:.2f}%\n"
            f"Profit Factor:        {m['profit_factor']:.3f}\n"
            f"Average Win:          {m['avg_win']:.3f}\n"
            f"Average Loss:         {m['avg_loss']:.3f}\n"
            f"Expectancy:           {m['expectancy']:.3f}\n"
            "\n------------------ Benchmark Metrics -----------------------\n"
            f"Alpha:                {m['alpha']:.4f}\n"
            f"Beta:                 {m['beta']:.4f}\n"
            f"Tracking Error:       {m['tracking_error']:.4f}\n"
            f"Information Ratio:    {m['information_ratio']:.4f}\n"
            f"Up Capture Ratio:     {m['up_capture_ratio']:.3f}\n"
            f"Down Capture Ratio:   {m['down_capture_ratio']:.3f}\n"
            "============================================================\n"
        )
