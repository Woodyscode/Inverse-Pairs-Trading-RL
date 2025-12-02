import numpy as np
import pandas as pd

from utils.logger import get_logger
from agents.ppo_agent import PPOAgent
from backtest.backtester import Backtester

try:
    from statsmodels.tsa.stattools import coint
except:
    coint = None

from data.loaders import clean_df

from env.features.advanced import (
    compute_all_advanced_features,
    N_ADVANCED_PAIR_FEATURES,
)
from env.features.targets import build_targets
from env.features.feature_selector import FeatureSelector

logger = get_logger("WalkForward")


# =====================================================================
#                  WALK-FORWARD ENGINE WITH LOGGING
# =====================================================================
class WalkForwardEngine:

    def __init__(self, ohlcv, returns, config, env_builder, price_index):

        logger.info("[WF] Initializing WalkForwardEngine...")

        required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        for k in required:
            if k not in ohlcv:
                raise KeyError(f"[WF] Missing OHLCV key {k}")
            if not isinstance(ohlcv[k], pd.DataFrame):
                raise TypeError(f"[WF] OHLCV[{k}] must be DataFrame")

            if not ohlcv[k].index.equals(returns.index):
                raise ValueError(f"[WF] Index mismatch for {k}")

        logger.info("[WF] OHLCV + returns validated")

        self.full_ohlcv = ohlcv
        self.returns = clean_df(returns.copy())
        self.config = config
        self.env_builder = env_builder

        self.price_index_global = (
            price_index if price_index is not None else self.returns.index
        )

        self.wf_cfg = config["walkforward"]
        self.pairs_cfg = config["pairs"]
        self.env_cfg = config["env"]
        self.train_cfg = config["training"]

        self.window_size = self.env_cfg["window_size"]
        self.initial_capital = self.env_cfg["initial_capital"]

        logger.info("[WF] Initialization complete")


    # =================================================================
    def _slice_ohlcv(self, start, end):
        logger.debug(f"[WF] Slicing OHLCV rows {start}:{end}")
        return {
            k: clean_df(df.iloc[start:end].copy())
            for k, df in self.full_ohlcv.items()
        }


    # =================================================================
    def _filter_ohlcv_to_pairs(self, ohlcv_slice, pairs):
        tickers = {x for p in pairs for x in p}
        logger.info(f"[WF] Filtering OHLCV to tickers: {sorted(tickers)}")

        out = {}
        for key, df in ohlcv_slice.items():
            keep = [c for c in df.columns if c in tickers]
            out[key] = df[keep].copy()

        return out


    # =================================================================
    def _generate_windows(self):

        idx = self.returns.index
        n = len(idx)

        train_days = int(self.wf_cfg["train_years"] * 252)
        test_days  = int(self.wf_cfg["test_months"] * 21)

        min_train = self.wf_cfg["min_train_size"]
        min_test  = self.wf_cfg["min_test_size"]

        mode = self.wf_cfg.get("mode", "rolling")

        logger.info(f"[WF] Generating walk-forward windows...")
        logger.info(f"[WF] train_days={train_days}, test_days={test_days}, mode={mode}")

        for start in range(train_days, n - test_days, test_days):

            train_start = 0 if mode == "expanding" else (start - train_days)
            train_end   = start
            test_end    = start + test_days

            logger.info(f"\n[WF] Window candidate: train={train_start}-{train_end}, test={train_end}-{test_end}")

            tr = clean_df(self.returns.iloc[train_start:train_end])
            te = clean_df(self.returns.iloc[train_end:test_end])

            if len(tr) < min_train or len(te) < min_test:
                logger.warning("[WF] Skipping window: insufficient data")
                continue

            ohl_train = self._slice_ohlcv(train_start, train_end)
            ohl_test  = self._slice_ohlcv(train_end, test_end)

            yield ohl_train, ohl_test, tr, te, idx[train_start], idx[test_end - 1]


    # =================================================================
    def _select_pairs(self, ohl_train, train_ret):

        trs = train_ret.dropna(axis=1, how="all")
        if trs.shape[1] < 2:
            logger.warning("[WF] Only 1 ticker available; skipping")
            return []

        logger.info(f"[WF] Selecting pairs from {trs.shape[1]} tickers")
        logger.info(f"[WF] Tickers: {list(trs.columns)}")

        trs = trs.sort_index(axis=1)
        close = ohl_train["Close"].sort_index(axis=1)

        cfg = self.pairs_cfg
        corr_th   = cfg["correlation_threshold"]
        max_pairs = cfg["max_pairs"]
        min_obs   = cfg["min_observations"]
        p_th      = cfg["cointegration_p_value"]

        corr = trs.corr()
        tickers = trs.columns.tolist()
        candidates = []

        for i, A in enumerate(tickers):
            for B in tickers[i+1:]:

                logger.debug(f"[WF] Testing pair ({A},{B})")

                ra, rb = trs[A], trs[B]
                common = ra.index.intersection(rb.index)

                if len(common) < min_obs:
                    logger.debug(f"[WF] Reject {A}-{B}: insufficient obs ({len(common)} < {min_obs})")
                    continue

                if ra.std() < 1e-12 or rb.std() < 1e-12:
                    logger.debug(f"[WF] Reject {A}-{B}: near-zero std")
                    continue

                rho = corr.loc[A, B]
                if np.isnan(rho) or rho > corr_th:
                    logger.debug(f"[WF] Reject {A}-{B}: corr={rho:.4f} > {corr_th}")
                    continue

                if p_th and coint is not None:
                    try:
                        pa = close[A].loc[common]
                        pb = close[B].loc[common]
                        _, pval, _ = coint(pa, pb)
                        if pval > p_th:
                            logger.debug(f"[WF] Reject {A}-{B}: coint p={pval:.4f} > {p_th}")
                            continue
                    except Exception as e:
                        logger.debug(f"[WF] Coint failed for {A}-{B}: {e}")

                candidates.append(((A, B), rho))

        if not candidates:
            logger.warning("[WF] No valid pairs found")
            return []

        candidates.sort(key=lambda x: x[1])
        selected = [p for p, _ in candidates[:max_pairs]]

        logger.info(f"[WF] Selected pairs: {selected}")

        return selected


    # =================================================================
    def _compute_features_for_window(self, ohlcv, pairs):
        logger.info("[WF] Computing raw features for window...")

        raw = {}
        metrics = {}

        for (A, B) in pairs:
            T = len(ohlcv["Close"][A])
            logger.info(f"[WF] Feature compute: {A}-{B}, T={T}")

            mat = np.zeros((T, N_ADVANCED_PAIR_FEATURES), dtype=np.float32)
            m_list = []
            kal = None

            for t in range(T):
                feats, kal, m = compute_all_advanced_features(
                    ohlcv=ohlcv,
                    A=A,
                    B=B,
                    step=t+1,
                    kalman_state=kal,
                )
                mat[t] = feats
                m_list.append(m)

            raw[(A, B)] = mat
            metrics[(A, B)] = m_list

        logger.info("[WF] Raw features computed")
        return raw, metrics


    # =================================================================
    def _run_elastic_net(self, raw_train, raw_test, pairs, ohl_train):

        logger.info("[WF] Running ElasticNet feature reduction...")

        # Create selector once per window
        selector = FeatureSelector(
            alpha=self.config["elasticnet"]["alpha"],
            l1_ratio=self.config["elasticnet"]["l1_ratio"],
            max_iter=self.config["elasticnet"]["max_iter"],
            auto_k=True,
            k_override=None,
            importance_threshold=0.0,
        )

        train_features = {}
        test_features  = {}
        feature_dims   = []

        # Build all y targets at once
        logger.info("[WF] Building targets...")
        targets_dict = build_targets(ohlcv=ohl_train, pairs=pairs)

        # Process each pair individually
        for (A, B) in pairs:

            X_train = raw_train[(A, B)]
            X_test  = raw_test[(A, B)]
            y_full  = targets_dict[(A, B)]

            T_train = X_train.shape[0]
            y_train = y_full[:T_train]

            logger.info(f"[WF] Pair {A}-{B}: raw dims train={X_train.shape}, test={X_test.shape}")

            # Fit selector for this pair
            selector.fit_pair(
                pair=(A, B),
                X=X_train,
                y=y_train
            )

            # Transform both
            X_train_sel = selector.transform_pair((A, B), X_train)
            X_test_sel  = selector.transform_pair((A, B), X_test)

            train_features[(A, B)] = X_train_sel
            test_features[(A, B)]  = X_test_sel
            feature_dims.append(X_train_sel.shape[1])

            logger.info(
                f"[WF] EN reduced {A}-{B}: {X_train.shape[1]} → {X_train_sel.shape[1]}"
            )

        return train_features, test_features, feature_dims





    # =================================================================
    def run(self):

        logger.info("\n======== WALK-FORWARD START ========")

        stitched_equity = None
        running_capital = self.initial_capital
        global_trade_pnls = []
        window_summaries  = []
        window_idx = 0

        # ----------------------------------------------------------
        # Iterate through WF windows
        # ----------------------------------------------------------
        for (
            ohl_train,
            ohl_test,
            train_ret,
            test_ret,
            start_date,
            end_date,
        ) in self._generate_windows():

            window_idx += 1
            logger.info(f"\n===== WF Window {window_idx}: {start_date.date()} → {end_date.date()} =====")

            logger.info(f"[WF] Train size={len(train_ret)}, Test size={len(test_ret)}")

            # 1. Pair selection
            pairs = self._select_pairs(ohl_train, train_ret)
            if not pairs:
                logger.warning("[WF] No pairs found, skipping window")
                continue

            # 2. Filter OHLCV
            ohl_train = self._filter_ohlcv_to_pairs(ohl_train, pairs)
            ohl_test  = self._filter_ohlcv_to_pairs(ohl_test, pairs)

            # 3. Raw features
            raw_train, metrics_train = self._compute_features_for_window(ohl_train, pairs)
            raw_test,  metrics_test  = self._compute_features_for_window(ohl_test,  pairs)

            # 4. ElasticNet
            f_train, f_test, dims = self._run_elastic_net(
                raw_train, raw_test, pairs, ohl_train
            )

            # 5. Train Env
            logger.info("[WF] Building training environment...")
            env_train = self.env_builder(
                ohlcv=ohl_train,
                pairs=pairs,
                config=self.config,
                price_index=ohl_train["Close"].index,
                features=f_train,
                metrics=metrics_train,
                feature_dims=dims
            )

            agent = PPOAgent(self.config, env_train)
            logger.info("[WF] Training PPO agent...")
            agent.train(self.train_cfg["timesteps"])

            # 6. Test Env
            logger.info("[WF] Building test environment...")
            env_test = self.env_builder(
                ohlcv=ohl_test,
                pairs=pairs,
                config=self.config,
                price_index=ohl_test["Close"].index,
                features=f_test,
                metrics=metrics_test,
                feature_dims=dims
            )

            # 7. Backtest
            logger.info("[WF] Running backtest...")
            bt = Backtester(agent=agent, env=env_test, config=self.config)
            results = bt.run()

            eq_curve = results["equity_curve"]
            metrics  = results["metrics"]

            if "trade_pnls" in results:
                global_trade_pnls.extend(results["trade_pnls"])

            # 8. Stitching
            logger.info("[WF] Stitching equity curve...")
            rets = eq_curve.pct_change().fillna(0)
            scaled = running_capital * (1 + rets).cumprod()

            running_capital = scaled.iloc[-1]
            scaled.index = eq_curve.index

            if stitched_equity is None:
                stitched_equity = scaled
            else:
                new_part = scaled.loc[~scaled.index.isin(stitched_equity.index)]
                stitched_equity = pd.concat([stitched_equity, new_part])

            stitched_equity = stitched_equity[~stitched_equity.index.duplicated()]

            window_summaries.append({
                "window": window_idx,
                "start": start_date,
                "end": end_date,
                "metrics": metrics,
            })

        # ----------------------------------------------------------
        # FINAL METRICS
        # ----------------------------------------------------------
        if stitched_equity is None:
            raise RuntimeError("[WF] No windows executed")

        logger.info("\n======== WALK-FORWARD COMPLETE ========")
        logger.info(f"[WF] Final stitched value: {stitched_equity.iloc[-1]:,.2f}")

        spy = self.config["benchmark_curve"].reindex(stitched_equity.index).ffill()
        spy = spy / spy.iloc[0]

        from backtest.metrics import compute_all_metrics
        global_metrics = compute_all_metrics(
            equity_curve=stitched_equity,
            benchmark_curve=spy,
            trade_pnls=global_trade_pnls,
        )

        return {
            "equity_curve": stitched_equity,
            "spy_curve": spy,
            "metrics": global_metrics,
            "window_metrics": window_summaries,
        }
