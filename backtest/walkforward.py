import numpy as np
import pandas as pd

from utils.logger import get_logger
from agents.ppo_agent import PPOAgent
from backtest.backtester import Backtester

try:
    from statsmodels.tsa.stattools import coint
except Exception:
    coint = None

try:
    from data.loaders import clean_df
except Exception:
    clean_df = lambda x: x.ffill().bfill()

logger = get_logger("WalkForward")


# =====================================================================
#                      WALK-FORWARD ENGINE (FINAL)
# =====================================================================
class WalkForwardEngine:

    def __init__(self, ohlcv, returns, config, env_builder, price_index):
        # ----------------------- VALIDATE ------------------------
        required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

        for k in required:
            if k not in ohlcv:
                raise KeyError(f"[WF] Missing OHLCV key: {k}")
            if not isinstance(ohlcv[k], pd.DataFrame):
                raise TypeError(f"[WF] OHLCV[{k}] must be a DataFrame.")

        idx_ref = returns.index
        for k in required:
            if not ohlcv[k].index.equals(idx_ref):
                raise ValueError(f"[WF] Index mismatch in OHLCV[{k}]")

        self.full_ohlcv = ohlcv
        self.returns = clean_df(returns.copy())
        self.config = config
        self.env_builder = env_builder

        if env_builder is None:
            raise RuntimeError("[WF] env_builder cannot be None.")

        self.price_index_global = (
            price_index if price_index is not None else self.returns.index
        )

        # WF config
        self.wf_cfg    = config["walkforward"]
        self.pairs_cfg = config["pairs"]
        self.env_cfg   = config["env"]
        self.train_cfg = config["training"]

        self.window_size     = self.env_cfg["window_size"]
        self.initial_capital = self.env_cfg["initial_capital"]


    # =================================================================
    def _slice_ohlcv(self, start, end):
        """Slice OHLCV for each window safely."""
        return {k: clean_df(df.iloc[start:end].copy())
                for k, df in self.full_ohlcv.items()}


    # =================================================================
    def _filter_ohlcv_to_pairs(self, ohlcv_slice, pairs):
        """Keep only tickers appearing in the selected pair list."""

        needed = set([ticker for pair in pairs for ticker in pair])
        filtered = {}

        for key, df in ohlcv_slice.items():
            cols = [c for c in df.columns if c in needed]
            filtered[key] = df[cols].copy()

        logger.info(f"[WF] Window uses tickers: {sorted(list(needed))}")
        return filtered


    # =================================================================
    def _generate_windows(self):

        idx = self.returns.index
        n = len(idx)

        train_days = int(self.wf_cfg["train_years"] * 252)
        test_days  = int(self.wf_cfg["test_months"] * 21)

        min_train = self.wf_cfg["min_train_size"]
        min_test  = self.wf_cfg["min_test_size"]
        mode      = self.wf_cfg.get("mode", "rolling")

        if n < (train_days + test_days + self.window_size):
            logger.warning("[WF] Not enough data for WFO.")
            return

        for start in range(train_days, n - test_days, test_days):

            train_start = 0 if mode == "expanding" else (start - train_days)
            train_end   = start
            test_end    = start + test_days

            train_ret = clean_df(self.returns.iloc[train_start:train_end])
            test_ret  = clean_df(self.returns.iloc[train_end:test_end])

            if len(train_ret) < min_train or len(test_ret) < min_test:
                continue

            ohlcv_train = self._slice_ohlcv(train_start, train_end)
            ohlcv_test  = self._slice_ohlcv(train_end, test_end)

            # ensure window_size validity
            if len(ohlcv_train["Close"]) < self.window_size:
                continue

            yield (
                ohlcv_train,
                ohlcv_test,
                train_ret,
                idx[train_start],
                idx[test_end - 1],
            )


    # =================================================================
    def _select_pairs(self, ohlcv_train, train_ret):
        """Correlation + cointegration based pair selection."""

        trs = train_ret.dropna(axis=1, how="all")

        if trs.shape[1] < 2:
            logger.warning("[WF] Only one ticker available.")
            return []

        trs = trs.sort_index(axis=1)
        close = ohlcv_train["Close"].sort_index(axis=1)

        cfg         = self.pairs_cfg
        corr_th     = cfg["correlation_threshold"]
        max_pairs   = cfg["max_pairs"]
        min_obs     = cfg["min_observations"]
        pval_th     = cfg["cointegration_p_value"]

        corr     = trs.corr()
        tickers  = trs.columns.tolist()
        matches  = []

        for i, a in enumerate(tickers):
            for b in tickers[i+1:]:

                ra, rb = trs[a], trs[b]
                common = ra.index.intersection(rb.index)

                if len(common) < min_obs:
                    continue
                if ra.std() < 1e-12 or rb.std() < 1e-12:
                    continue

                rho = corr.loc[a, b]
                if np.isnan(rho) or rho > corr_th:
                    continue

                # cointegration filter
                if pval_th and coint is not None:
                    try:
                        pa = close[a].loc[common]
                        pb = close[b].loc[common]
                        _, pval, _ = coint(pa, pb)
                        if pval > pval_th:
                            continue
                    except:
                        pass

                matches.append(((a, b), rho))

        if not matches:
            logger.warning("[WF] No valid pairs found.")
            return []

        matches.sort(key=lambda x: x[1])
        selected = [pair for pair, _ in matches[:max_pairs]]

        logger.info(f"[WF] Selected pairs: {selected}")
        return selected


    # =================================================================
    def run(self):
        """Main WFO execution."""

        logger.info("\n======== WALK-FORWARD START ========")

        stitched_equity   = None
        running_capital   = self.initial_capital
        global_trade_pnls = []
        window_summaries  = []
        window_idx = 0

        for (
            ohlcv_train,
            ohlcv_test,
            train_ret,
            start_date,
            end_date,
        ) in self._generate_windows():

            window_idx += 1
            logger.info(f"\n--- WF Window {window_idx}: {start_date.date()} → {end_date.date()} ---")

            # ------------------------------
            # 1. Pair Selection
            # ------------------------------
            pairs = self._select_pairs(ohlcv_train, train_ret)
            if not pairs:
                continue

            # ------------------------------
            # 2. Filter OHLCV to required tickers
            # ------------------------------
            train_ohlcv_filtered = self._filter_ohlcv_to_pairs(ohlcv_train, pairs)
            test_ohlcv_filtered  = self._filter_ohlcv_to_pairs(ohlcv_test,  pairs)

            # ------------------------------
            # 3. Build train env using *train index*
            # ------------------------------
            train_index = train_ohlcv_filtered["Close"].index
            env_train = self.env_builder(
                self.config,
                train_ohlcv_filtered,
                pairs,
                train_index
            )

            # ------------------------------
            # 4. Train PPO agent
            # ------------------------------
            agent = PPOAgent(self.config, env_train)
            agent.train(self.train_cfg["timesteps"])

            # ------------------------------
            # 5. Build test env using *test index*
            # ------------------------------
            test_index = test_ohlcv_filtered["Close"].index
            env_test = self.env_builder(
                self.config,
                test_ohlcv_filtered,
                pairs,
                test_index
            )

            # ------------------------------
            # 6. Backtest
            # ------------------------------
            bt = Backtester(agent=agent, env=env_test, config=self.config)
            results = bt.run()

            eq_curve = results["equity_curve"].copy()
            metrics  = results["metrics"]

            if "trade_pnls" in metrics:
                global_trade_pnls.extend(metrics["trade_pnls"])

            # ------------------------------
            # 7. STITCH EQUITY CURVES
            # ------------------------------
            rets = eq_curve.pct_change().fillna(0)
            scaled = running_capital * (1 + rets).cumprod()

            running_capital = float(scaled.iloc[-1])

            # ✔ ensure correct index alignment
            scaled.index = eq_curve.index

            # ✔ remove duplicate indices
            if stitched_equity is None:
                stitched_equity = scaled
            else:
                scaled = scaled.loc[~scaled.index.isin(stitched_equity.index)]
                stitched_equity = pd.concat([stitched_equity, scaled])

            stitched_equity = stitched_equity[~stitched_equity.index.duplicated()]

            window_summaries.append({
                "window": window_idx,
                "start": start_date,
                "end": end_date,
                "metrics": metrics,
            })

        if stitched_equity is None:
            raise RuntimeError("[WF] No valid WF windows.")

        logger.info("\n======== WALK-FORWARD COMPLETE ========")
        logger.info(f"[WF] OOS Days: {len(stitched_equity)} | "
                    f"Final Value: {stitched_equity.iloc[-1]:,.2f}")

        # ------------------------------
        # Benchmark Curve (SPY)
        # ------------------------------
        spy = self.config["benchmark_curve"].reindex(stitched_equity.index).ffill()
        spy = spy / spy.iloc[0]

        from backtest.metrics import compute_all_metrics
        global_metrics = compute_all_metrics(
            equity_curve=stitched_equity,
            benchmark_curve=spy,
            trade_pnls=global_trade_pnls or None,
        )

        return {
            "equity_curve": stitched_equity,
            "spy_curve": spy,
            "metrics": global_metrics,
            "window_metrics": window_summaries,
        }
