import numpy as np
import pandas as pd

from utils.logger import get_logger
from agents.ppo_agent import PPOAgent
from backtest.backtester import Backtester
from backtest.baselines import CashPolicy, RandomPolicy, ZScoreMeanReversionPolicy

from data.loaders import clean_df
from utils.pairs_selection import find_strong_pairs

from env.features.advanced import (
    compute_all_advanced_features,
    N_ADVANCED_PAIR_FEATURES,
)
from env.features.targets import build_targets
from env.features.feature_selector import FeatureSelector

logger = get_logger("WalkForward")


# =====================================================================
#                  WALK-FORWARD ENGINE (FINAL VERSION)
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
        self.feature_context_days = self.wf_cfg.get("feature_context_days", 252)
        self.validation_days = int(self.wf_cfg.get("validation_months", 0) * 21)
        self.price_type = config.get("data", {}).get("price_type", "Adj Close")
        self.selected_pair_history = []

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
    def _concat_ohlcv(self, left, right):
        return {
            key: clean_df(pd.concat([left[key], right[key]], axis=0))
            for key in left
        }


    # =================================================================
    def _generate_windows(self):

        idx = self.returns.index
        n = len(idx)

        train_days = int(self.wf_cfg["train_years"] * 252)
        test_days  = int(self.wf_cfg["test_months"] * 21)

        min_train = self.wf_cfg["min_train_size"]
        min_test  = self.wf_cfg["min_test_size"]

        mode = self.wf_cfg.get("mode", "rolling")
        max_windows = self.wf_cfg.get("max_windows")
        yielded = 0

        logger.info(f"[WF] Generating walk-forward windows...")
        logger.info(f"[WF] train_days={train_days}, test_days={test_days}, mode={mode}")

        for start in range(train_days, n - test_days, test_days):

            train_start = 0 if mode == "expanding" else (start - train_days)
            train_end   = start
            test_end    = start + test_days

            logger.info(
                f"\n[WF] Window candidate: "
                f"train={train_start}-{train_end}, test={train_end}-{test_end}"
            )

            tr = clean_df(self.returns.iloc[train_start:train_end])
            te = clean_df(self.returns.iloc[train_end:test_end])

            if len(tr) < min_train or len(te) < min_test:
                logger.warning("[WF] Skipping window: insufficient data")
                continue

            ohl_train = self._slice_ohlcv(train_start, train_end)
            ohl_test  = self._slice_ohlcv(train_end, test_end)

            yielded += 1
            yield ohl_train, ohl_test, tr, te, idx[train_start], idx[test_end - 1]
            if max_windows is not None and yielded >= int(max_windows):
                break


    # =================================================================
    def _generate_validation_windows(self):
        idx = self.returns.index
        n = len(idx)

        train_days = int(self.wf_cfg["train_years"] * 252)
        val_days = self.validation_days
        test_days = int(self.wf_cfg["test_months"] * 21)

        min_train = self.wf_cfg["min_train_size"]
        min_val = self.wf_cfg.get("min_validation_size", max(20, val_days // 2))
        min_test = self.wf_cfg["min_test_size"]

        mode = self.wf_cfg.get("mode", "rolling")
        max_windows = self.wf_cfg.get("max_windows")
        yielded = 0
        step_days = self.wf_cfg.get("step_days", test_days)

        logger.info("[WF] Generating train/validation/test windows...")
        logger.info(
            f"[WF] train_days={train_days}, val_days={val_days}, "
            f"test_days={test_days}, mode={mode}"
        )

        for start in range(train_days, n - val_days - test_days + 1, step_days):
            train_start = 0 if mode == "expanding" else (start - train_days)
            train_end = start
            val_end = train_end + val_days
            test_end = val_end + test_days

            tr = clean_df(self.returns.iloc[train_start:train_end])
            va = clean_df(self.returns.iloc[train_end:val_end])
            te = clean_df(self.returns.iloc[val_end:test_end])

            if len(tr) < min_train or len(va) < min_val or len(te) < min_test:
                logger.warning("[WF] Skipping validation window: insufficient data")
                continue

            yielded += 1
            yield (
                self._slice_ohlcv(train_start, train_end),
                self._slice_ohlcv(train_end, val_end),
                self._slice_ohlcv(val_end, test_end),
                tr,
                va,
                te,
                idx[train_start],
                idx[train_end],
                idx[val_end],
                idx[test_end - 1],
            )
            if max_windows is not None and yielded >= int(max_windows):
                break


    # =================================================================
    def _compute_features_for_window(self, ohlcv, pairs, context_ohlcv=None):
        logger.info("[WF] Computing raw features for window...")

        raw = {}
        metrics = {}
        context_len = 0

        if context_ohlcv is not None and len(context_ohlcv["Close"]) > 0:
            context_len = len(context_ohlcv["Close"])
            feature_ohlcv = self._concat_ohlcv(context_ohlcv, ohlcv)
        else:
            feature_ohlcv = ohlcv

        for (A, B) in pairs:
            total_T = len(feature_ohlcv["Close"][A])
            target_T = len(ohlcv["Close"][A])
            logger.info(f"[WF] Feature compute: {A}-{B}, T={target_T}, context={context_len}")

            mat_all = np.zeros((total_T, N_ADVANCED_PAIR_FEATURES), dtype=np.float32)
            m_list = []
            kal = None

            for t in range(total_T):
                feats, kal, m = compute_all_advanced_features(
                    ohlcv=feature_ohlcv,
                    A=A,
                    B=B,
                    step=t+1,
                    kalman_state=kal,
                    base_window=self.config.get("features", {}).get("base_window", 30),
                    min_periods=self.config.get("features", {}).get("min_periods", 60),
                    price_type=self.price_type,
                )
                mat_all[t] = feats
                m_list.append(m)

            raw[(A, B)] = mat_all[context_len:]
            metrics[(A, B)] = m_list[context_len:]

            if raw[(A, B)].shape[0] != target_T:
                raise RuntimeError(
                    f"[WF] Feature rows for {A}-{B} do not match target window"
                )

        logger.info("[WF] Raw features computed")
        return raw, metrics


    # =================================================================
    def _fit_transform_feature_sets(self, raw_train, raw_slices, pairs, ohl_train):

        logger.info("[WF] Running ElasticNet feature reduction...")

        selector = FeatureSelector(
            alpha=self.config["elasticnet"]["alpha"],
            l1_ratio=self.config["elasticnet"]["l1_ratio"],
            max_iter=self.config["elasticnet"]["max_iter"],
            auto_k=True,
            k_override=None,
            importance_threshold=0.0,
        )

        train_features = {}
        transformed_slices = {name: {} for name in raw_slices}
        feature_dims = []

        logger.info("[WF] Building targets...")
        targets_dict = build_targets(
            ohlcv=ohl_train,
            pairs=pairs,
            price_type=self.price_type,
        )

        for (A, B) in pairs:

            X_train = raw_train[(A, B)]
            y_full  = targets_dict[(A, B)]

            T_train = X_train.shape[0]
            y_train = y_full[:T_train]

            slice_shapes = {
                name: raw[(A, B)].shape for name, raw in raw_slices.items()
            }
            logger.info(
                f"[WF] Pair {A}-{B}: raw dims train={X_train.shape}, "
                f"slices={slice_shapes}"
            )

            selector.fit_pair((A, B), X_train, y_train)

            X_train_sel = selector.transform_pair((A, B), X_train)

            train_features[(A, B)] = X_train_sel

            for name, raw in raw_slices.items():
                transformed_slices[name][(A, B)] = selector.transform_pair((A, B), raw[(A, B)])

            feature_dims.append(X_train_sel.shape[1])

            logger.info(
                f"[WF] EN reduced {A}-{B}: "
                f"{X_train.shape[1]} → {X_train_sel.shape[1]}"
            )

        return train_features, transformed_slices, feature_dims


    # =================================================================
    def _run_elastic_net(self, raw_train, raw_test, pairs, ohl_train):
        train_features, transformed, dims = self._fit_transform_feature_sets(
            raw_train=raw_train,
            raw_slices={"test": raw_test},
            pairs=pairs,
            ohl_train=ohl_train,
        )
        return train_features, transformed["test"], dims


    # =================================================================
    def _build_env(self, ohlcv, pairs, pair_types, features, metrics, dims):
        return self.env_builder(
            ohlcv=ohlcv,
            pairs=pairs,
            pair_types=pair_types,
            config=self.config,
            price_index=ohlcv["Close"].index,
            features=features,
            metrics=metrics,
            feature_dims=dims,
        )


    # =================================================================
    def _baseline_results(self, ohlcv, pairs, pair_types, features, metrics, dims):
        results = {}

        env_cash = self._build_env(ohlcv, pairs, pair_types, features, metrics, dims)
        results["cash"] = Backtester(
            agent=CashPolicy(env_cash.action_space.shape[0]),
            env=env_cash,
            config=self.config,
        ).run()

        env_random = self._build_env(ohlcv, pairs, pair_types, features, metrics, dims)
        results["random"] = Backtester(
            agent=RandomPolicy(env_random.action_space.shape[0], seed=self.config.get("system", {}).get("seed", 42)),
            env=env_random,
            config=self.config,
        ).run()

        env_z = self._build_env(ohlcv, pairs, pair_types, features, metrics, dims)
        results["zscore_mean_reversion"] = Backtester(
            agent=ZScoreMeanReversionPolicy(env_z),
            env=env_z,
            config=self.config,
        ).run()

        return results


    # =================================================================
    #                            RUN
    # =================================================================
    def run(self):
        if self.validation_days > 0:
            return self._run_with_validation()

        logger.info("\n======== WALK-FORWARD START ========")

        stitched_equity = None
        running_capital = self.initial_capital
        global_trade_pnls = []
        window_summaries  = []
        window_idx = 0

        for (
            ohl_train,
            ohl_test,
            train_ret,
            test_ret,
            start_date,
            end_date,
        ) in self._generate_windows():

            window_idx += 1
            logger.info(
                f"\n===== WF Window {window_idx}: "
                f"{start_date.date()} → {end_date.date()} ====="
            )

            logger.info(
                f"[WF] Train size={len(train_ret)}, Test size={len(test_ret)}"
            )

            # =============================================
            # 1. PAIR SELECTION (traditional + inverse)
            # =============================================
            close_train = ohl_train[self.price_type].copy()
            ret_train   = close_train.pct_change().dropna()

            logger.info("[WF] Selecting pairs...")

            # FIXED CALL — matches your selector
            traditional_pairs, inverse_pairs = find_strong_pairs(
                returns_df=ret_train,
                prices_df=close_train,
                config=self.config
            )

            pairs = traditional_pairs + inverse_pairs

            logger.info(f"[WF] Traditional pairs: {traditional_pairs}")
            logger.info(f"[WF] Inverse pairs:     {inverse_pairs}")
            logger.info(f"[WF] TOTAL pairs used: {len(pairs)}")

            if len(pairs) == 0:
                logger.warning("[WF] No strong pairs found — skipping window")
                continue

            # =============================================
            # 2. Build pair-type dictionary
            # =============================================
            pair_types = {}
            for p in traditional_pairs:
                pair_types[p] = "traditional"
            for p in inverse_pairs:
                pair_types[p] = "inverse"

            self.selected_pair_history.append({
                "window": window_idx,
                "start": str(start_date.date()),
                "end": str(end_date.date()),
                "traditional_pairs": traditional_pairs,
                "inverse_pairs": inverse_pairs,
            })

            # =============================================
            # 3. FILTER OHLCV TO SELECTED PAIRS
            # =============================================
            ohl_train = self._filter_ohlcv_to_pairs(ohl_train, pairs)
            ohl_test  = self._filter_ohlcv_to_pairs(ohl_test,  pairs)

            # =============================================
            # 4. Feature computation
            # =============================================
            context_rows = min(self.feature_context_days, len(ohl_train["Close"]))
            context_ohl = {
                key: df.iloc[-context_rows:].copy()
                for key, df in ohl_train.items()
            }

            raw_train, metrics_train = self._compute_features_for_window(ohl_train, pairs)
            raw_test,  metrics_test  = self._compute_features_for_window(
                ohl_test,
                pairs,
                context_ohlcv=context_ohl,
            )

            for pair, mat in raw_test.items():
                if mat.size and np.allclose(mat, 0):
                    raise RuntimeError(f"[WF] Test features are all zero for pair {pair}")

            # =============================================
            # 5. ElasticNet feature reduction
            # =============================================
            f_train, f_test, dims = self._run_elastic_net(
                raw_train, raw_test, pairs, ohl_train
            )

            # =============================================
            # 6. Train RL on training window
            # =============================================
            env_train = self.env_builder(
                ohlcv=ohl_train,
                pairs=pairs,
                pair_types=pair_types,
                config=self.config,
                price_index=ohl_train["Close"].index,
                features=f_train,
                metrics=metrics_train,
                feature_dims=dims,
            )

            agent = PPOAgent(self.config, env_train)
            agent.train(
                self.config["rl"]["hyperparameters"]["timesteps"],
                log_interval=self.train_cfg.get("log_interval", 10),
                save_dir=self.train_cfg.get("save_dir", "models/ppo"),
            )

            # =============================================
            # 7. Test environment + backtest
            # =============================================
            env_test = self.env_builder(
                ohlcv=ohl_test,
                pairs=pairs,
                pair_types=pair_types,
                config=self.config,
                price_index=ohl_test["Close"].index,
                features=f_test,
                metrics=metrics_test,
                feature_dims=dims,
            )

            bt = Backtester(agent=agent, env=env_test, config=self.config)
            results = bt.run()

            eq_curve = results["equity_curve"]

            if "trade_pnls" in results:
                global_trade_pnls.extend(results["trade_pnls"])

            # =============================================
            # 8. Equity stitching
            # =============================================
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
                "metrics": results["metrics"],
                "pairs": pairs,
            })

        # ======================================================
        # FINAL METRICS
        # ======================================================
        if stitched_equity is None:
            raise RuntimeError("[WF] No WF windows executed")

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
            "selected_pairs": self.selected_pair_history,
        }


    # =================================================================
    def _run_with_validation(self):
        logger.info("\n======== WALK-FORWARD TRAIN/VALIDATION/TEST START ========")

        stitched_equity = None
        running_capital = self.initial_capital
        global_trade_pnls = []
        global_frictions = []
        pair_contributions = {}
        baseline_stitched_equity = {}
        baseline_running_capital = {}
        baseline_trade_pnls = {}
        baseline_frictions = {}
        baseline_pair_contributions = {}
        window_summaries = []
        window_idx = 0

        for (
            ohl_train,
            ohl_val,
            ohl_test,
            train_ret,
            val_ret,
            test_ret,
            train_start_date,
            val_start_date,
            test_start_date,
            end_date,
        ) in self._generate_validation_windows():
            window_idx += 1
            logger.info(
                f"\n===== WF Window {window_idx}: train={train_start_date.date()} "
                f"val={val_start_date.date()} test={test_start_date.date()} "
                f"end={end_date.date()} ====="
            )
            logger.info(
                f"[WF] Sizes train={len(train_ret)}, val={len(val_ret)}, test={len(test_ret)}"
            )

            close_train = ohl_train[self.price_type].copy()
            ret_train = close_train.pct_change().dropna()

            traditional_pairs, inverse_pairs = find_strong_pairs(
                returns_df=ret_train,
                prices_df=close_train,
                config=self.config,
            )
            pairs = traditional_pairs + inverse_pairs

            if len(pairs) == 0:
                logger.warning("[WF] No strong pairs found - skipping validation window")
                continue

            pair_types = {pair: "traditional" for pair in traditional_pairs}
            pair_types.update({pair: "inverse" for pair in inverse_pairs})

            self.selected_pair_history.append({
                "window": window_idx,
                "train_start": str(train_start_date.date()),
                "validation_start": str(val_start_date.date()),
                "test_start": str(test_start_date.date()),
                "end": str(end_date.date()),
                "traditional_pairs": traditional_pairs,
                "inverse_pairs": inverse_pairs,
            })

            ohl_train = self._filter_ohlcv_to_pairs(ohl_train, pairs)
            ohl_val = self._filter_ohlcv_to_pairs(ohl_val, pairs)
            ohl_test = self._filter_ohlcv_to_pairs(ohl_test, pairs)

            context_rows = min(self.feature_context_days, len(ohl_train["Close"]))
            train_context = {
                key: df.iloc[-context_rows:].copy()
                for key, df in ohl_train.items()
            }
            val_context_full = self._concat_ohlcv(ohl_train, ohl_val)
            test_context = {
                key: df.iloc[-context_rows:].copy()
                for key, df in val_context_full.items()
            }

            raw_train, metrics_train = self._compute_features_for_window(ohl_train, pairs)
            raw_val, metrics_val = self._compute_features_for_window(
                ohl_val,
                pairs,
                context_ohlcv=train_context,
            )
            raw_test, metrics_test = self._compute_features_for_window(
                ohl_test,
                pairs,
                context_ohlcv=test_context,
            )

            for name, raw in {"validation": raw_val, "test": raw_test}.items():
                for pair, mat in raw.items():
                    if mat.size and np.allclose(mat, 0):
                        raise RuntimeError(f"[WF] {name} features are all zero for pair {pair}")

            f_train, transformed, dims = self._fit_transform_feature_sets(
                raw_train=raw_train,
                raw_slices={"validation": raw_val, "test": raw_test},
                pairs=pairs,
                ohl_train=ohl_train,
            )
            f_val = transformed["validation"]
            f_test = transformed["test"]

            env_train = self._build_env(
                ohl_train, pairs, pair_types, f_train, metrics_train, dims
            )

            agent = PPOAgent(self.config, env_train)
            agent.train(
                self.config["rl"]["hyperparameters"]["timesteps"],
                log_interval=self.train_cfg.get("log_interval", 10),
                save_dir=self.train_cfg.get("save_dir", "models/ppo"),
            )

            env_val = self._build_env(
                ohl_val, pairs, pair_types, f_val, metrics_val, dims
            )
            validation_results = Backtester(agent=agent, env=env_val, config=self.config).run()

            env_test = self._build_env(
                ohl_test, pairs, pair_types, f_test, metrics_test, dims
            )
            test_results = Backtester(agent=agent, env=env_test, config=self.config).run()

            baseline_results = {}
            if self.config.get("validation", {}).get("run_baselines", True):
                baseline_results = self._baseline_results(
                    ohl_test, pairs, pair_types, f_test, metrics_test, dims
                )

            eq_curve = test_results["equity_curve"]
            global_trade_pnls.extend(test_results.get("trade_pnls", []))
            if not test_results.get("frictions", pd.DataFrame()).empty:
                fr = test_results["frictions"].copy()
                fr["window"] = window_idx
                fr["strategy"] = "ppo_base"
                global_frictions.append(fr)
            for pair_key, pnls in test_results.get("pair_pnls", {}).items():
                pair_contributions[pair_key] = pair_contributions.get(pair_key, 0.0) + float(np.sum(pnls))

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

            baseline_metrics = {}
            for name, baseline in baseline_results.items():
                baseline_metrics[name] = baseline["metrics"]
                b_eq = baseline["equity_curve"]
                baseline_trade_pnls.setdefault(name, []).extend(baseline.get("trade_pnls", []))
                if not baseline.get("frictions", pd.DataFrame()).empty:
                    b_fr = baseline["frictions"].copy()
                    b_fr["window"] = window_idx
                    b_fr["strategy"] = name
                    baseline_frictions.setdefault(name, []).append(b_fr)
                for pair_key, pnls in baseline.get("pair_pnls", {}).items():
                    cur = baseline_pair_contributions.setdefault(name, {})
                    cur[pair_key] = cur.get(pair_key, 0.0) + float(np.sum(pnls))

                b_rets = b_eq.pct_change().fillna(0)
                b_capital = baseline_running_capital.get(name, self.initial_capital)
                b_scaled = b_capital * (1 + b_rets).cumprod()
                baseline_running_capital[name] = b_scaled.iloc[-1]
                b_scaled.index = b_eq.index

                if name not in baseline_stitched_equity:
                    baseline_stitched_equity[name] = b_scaled
                else:
                    new_part = b_scaled.loc[~b_scaled.index.isin(baseline_stitched_equity[name].index)]
                    baseline_stitched_equity[name] = pd.concat([baseline_stitched_equity[name], new_part])

                baseline_stitched_equity[name] = baseline_stitched_equity[name][
                    ~baseline_stitched_equity[name].index.duplicated()
                ]

            window_summaries.append({
                "window": window_idx,
                "train_start": train_start_date,
                "validation_start": val_start_date,
                "test_start": test_start_date,
                "end": end_date,
                "pairs": pairs,
                "validation_metrics": validation_results["metrics"],
                "test_metrics": test_results["metrics"],
                "baseline_metrics": baseline_metrics,
            })

        if stitched_equity is None:
            raise RuntimeError("[WF] No validation WF windows executed")

        spy = self.config["benchmark_curve"].reindex(stitched_equity.index).ffill()
        spy = spy / spy.iloc[0]

        from backtest.metrics import compute_all_metrics
        global_metrics = compute_all_metrics(
            equity_curve=stitched_equity,
            benchmark_curve=spy,
            trade_pnls=global_trade_pnls,
        )

        baseline_global_metrics = {}
        baseline_spy_curves = {}
        for name, curve in baseline_stitched_equity.items():
            b_spy = self.config["benchmark_curve"].reindex(curve.index).ffill()
            b_spy = b_spy / b_spy.iloc[0]
            baseline_spy_curves[name] = b_spy
            baseline_global_metrics[name] = compute_all_metrics(
                equity_curve=curve,
                benchmark_curve=b_spy,
                trade_pnls=baseline_trade_pnls.get(name, []),
            )

        friction_table = (
            pd.concat(global_frictions, ignore_index=True)
            if global_frictions else pd.DataFrame()
        )
        baseline_friction_tables = {
            name: pd.concat(rows, ignore_index=True)
            for name, rows in baseline_frictions.items()
            if rows
        }

        return {
            "equity_curve": stitched_equity,
            "spy_curve": spy,
            "metrics": global_metrics,
            "window_metrics": window_summaries,
            "selected_pairs": self.selected_pair_history,
            "validation_protocol": "rolling_train_validation_test",
            "trade_pnls": global_trade_pnls,
            "frictions": friction_table,
            "pair_contributions": pair_contributions,
            "baseline_equity_curves": baseline_stitched_equity,
            "baseline_spy_curves": baseline_spy_curves,
            "baseline_global_metrics": baseline_global_metrics,
            "baseline_trade_pnls": baseline_trade_pnls,
            "baseline_frictions": baseline_friction_tables,
            "baseline_pair_contributions": baseline_pair_contributions,
        }
