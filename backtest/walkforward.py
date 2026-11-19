from pathlib import Path

import numpy as np
import pandas as pd

from utils.logger import get_logger
from agents.ppo_agent import PPOAgent
from backtest.backtester import Backtester
from backtest.baselines import CashPolicy, RandomPolicy, ZScoreMeanReversionPolicy

from data.wavelet import causal_wavelet_denoise
from utils.pairs_selection import select_pair_models

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
        self.returns = returns
        self.config = config
        self.env_builder = env_builder

        self.price_index_global = (
            price_index if price_index is not None else self.returns.index
        )

        self.wf_cfg = config["walkforward"]
        self.env_cfg = config["env"]
        self.train_cfg = config["training"]

        self.window_size = self.env_cfg["window_size"]
        self.initial_capital = self.env_cfg["initial_capital"]
        self.validation_days = int(self.wf_cfg.get("validation_months", 0) * 21)
        self.price_type = config.get("data", {}).get("price_type", "Adj Close")
        self.selected_pair_history = []
        self.pair_models = {}

        logger.info("[WF] Initialization complete")


    # =================================================================
    def _slice_ohlcv(self, start, end):
        logger.debug(f"[WF] Slicing OHLCV rows {start}:{end}")
        return {
            k: df.iloc[start:end].copy()
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
            key: pd.concat([left[key], right[key]], axis=0)
            for key in left
        }


    # =================================================================
    def _generate_windows(self):

        for train, _, test, tr, _, te, start, _, _, end in self._generate_validation_windows():
            yield train, test, tr, te, start, end

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

        if step_days < test_days:
            raise ValueError("Overlapping test windows cannot be stitched")

        first_fold = max(train_days, self.config["data"].get("min_history", 0))
        for start in range(first_fold, n - val_days - min_test + 1, step_days):
            train_start = 0 if mode == "expanding" else (start - train_days)
            train_end = start
            val_end = train_end + val_days
            test_end = min(n, val_end + test_days)

            tr = self.returns.iloc[train_start:train_end]
            va = self.returns.iloc[train_end:val_end]
            te = self.returns.iloc[val_end:test_end]

            if len(tr) < min_train or (val_days and len(va) < min_val) or len(te) < min_test:
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
            model = self.pair_models.get((A, B), {})
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
                    hedge_ratio=model.get("hedge_ratio", 1.0),
                    intercept=model.get("intercept", 0.0),
                )
                mat_all[t] = feats
                m_list.append(m)

            wavelet = self.config.get("data", {}).get("wavelet", {})
            if wavelet.get("enabled", False):
                prices = feature_ohlcv[self.price_type]
                beta = model.get("hedge_ratio", 1.0)
                gross = prices[A] + abs(beta) * prices[B]
                ret = (prices[A] - beta * prices[B]).diff() / gross.shift()
                signal = causal_wavelet_denoise(
                    ret.fillna(0), wavelet.get("name", "db4"),
                    wavelet.get("level", 3), wavelet.get("window", 64),
                    threshold_method=wavelet.get("threshold_method", "legacy"),
                )
                mat_all = np.column_stack((mat_all, signal.to_numpy(dtype=np.float32)))
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
            importance_threshold=self.config["elasticnet"].get("importance_threshold", 0.0),
        )

        train_features = {}
        transformed_slices = {name: {} for name in raw_slices}
        feature_dims = []

        logger.info("[WF] Building targets...")
        targets_dict = build_targets(
            ohlcv=ohl_train,
            pairs=pairs,
            price_type=self.price_type,
            pair_models=self.pair_models,
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

            lag = self.env_cfg.get("signal_lag_bars", 0)
            warm = self.config.get("features", {}).get("min_periods", 60) - 1
            valid_end = T_train - lag - 1
            if valid_end <= warm:
                raise ValueError("Training fold too short for feature warm-up and prediction horizon")
            selected = selector.fit_pair((A, B), X_train[warm:valid_end], y_train[warm + lag:-1])
            model = self.pair_models[(A, B)]
            model["selected_features"] = selected.tolist()
            model["scaler_mean"] = selector.scalers[(A, B)].mean_[selected].tolist()
            model["scaler_scale"] = selector.scalers[(A, B)].scale_[selected].tolist()

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
    def _build_env(self, ohlcv, pairs, pair_types, features, metrics, dims, start_step=None, capital=None):
        config = self.config
        if capital is not None:
            config = {**config, "env": {**self.env_cfg, "initial_capital": capital}}
        return self.env_builder(
            ohlcv=ohlcv, pairs=pairs, pair_types=pair_types, config=config,
            price_index=ohlcv["Close"].index, features=features, metrics=metrics,
            feature_dims=dims, pair_models=self.pair_models, start_step=start_step,
        )

    def _evaluate_window(self, train, val, test, capital, baseline_capitals, window):
        from backtest.metrics import compute_all_metrics

        train_end = train["Close"].index[-1]
        minimum = self.config["data"].get("min_history", 0)
        eligible = pd.Series(True, index=train["Close"].columns)
        for field, panel in train.items():
            valid = panel.notna().all()
            if field != "Volume":
                valid &= panel.gt(0).all()
            eligible &= valid
            eligible &= self.full_ohlcv[field].loc[:train_end].count() >= minimum
        prices = train[self.price_type].loc[:, eligible]
        models = select_pair_models(prices.pct_change(fill_method=None), prices, self.config)
        self.pair_models = {m["pair"]: m for m in models}
        pairs = list(self.pair_models)
        types = {m["pair"]: m["type"] for m in models}
        benchmark_only = self.config.get("validation", {}).get("benchmark_only", False)
        lag = int(self.env_cfg.get("signal_lag_bars", 0))
        warm = self.config.get("features", {}).get("min_periods", 60)

        def cash_result(index, initial):
            days = index.to_series().diff().dt.days.fillna(0)
            curve = initial * (1 + self.env_cfg.get("cash_rate_annual", 0) * days / 365).cumprod()
            pnls = curve.diff().dropna().tolist()
            return {"equity_curve": curve, "bar_pnls": pnls, "metrics": compute_all_metrics(
                curve, self.config["benchmark_curve"].reindex(index), pnls)}

        if not pairs:
            evaluated = {name: cash_result(test["Close"].index, value)
                         for name, value in baseline_capitals.items()}
            primary = cash_result(test["Close"].index, capital)
            validation = cash_result(val["Close"].index, self.initial_capital) if len(val["Close"]) else {}
            return primary, validation, evaluated, models

        train = self._filter_ohlcv_to_pairs(train, pairs)
        val = self._filter_ohlcv_to_pairs(val, pairs)
        test = self._filter_ohlcv_to_pairs(test, pairs)
        # Compute each chronological prefix once. Evaluation environments receive
        # sufficient preceding rows for signal lag and ADV, then start on fold day one.
        combined = self._concat_ohlcv(self._concat_ohlcv(train, val), test)
        for field, panel in combined.items():
            if panel.isna().any().any() or (field != "Volume" and (panel <= 0).any().any()):
                raise ValueError(f"Missing or invalid {field} quote for a selected pair in fold {window}")
        if benchmark_only:
            raw, metrics = {}, {}
            for pair, model in self.pair_models.items():
                a, b = pair
                levels = combined[self.price_type]
                spread = levels[a] - model["hedge_ratio"] * levels[b] - model["intercept"]
                rolling = spread.rolling(30)
                z = ((spread - rolling.mean()) / (rolling.std() + 1e-9)).fillna(0)
                metrics[pair] = [{"zscore": value} for value in z]
                raw[pair] = z.to_numpy(dtype=np.float32)[:, None]
        else:
            raw, metrics = self._compute_features_for_window(combined, pairs)
        n_train = len(train["Close"])
        n_val = len(val["Close"])
        if not benchmark_only:
            for pair, matrix in raw.items():
                for left, right in ((n_train, n_train + n_val), (n_train + n_val, len(matrix))):
                    if right > left and not np.any(matrix[left:right]):
                        raise ValueError(f"All evaluation features are zero for {pair} in fold {window}")
            raw_train = {p: x[:n_train] for p, x in raw.items()}
            _, transformed, dims = self._fit_transform_feature_sets(
                raw_train, {"all": raw}, pairs, train)
            features = transformed["all"]
        else:
            features, dims = raw, [1] * len(pairs)

        def environment(start, end, initial, training=False):
            context = max(lag, self.env_cfg.get("liquidity", {}).get("adv_window", 20))
            left = max(0, start - context)
            bars = {k: df.iloc[left:end] for k, df in combined.items()}
            feats = {p: x[left:end] for p, x in features.items()}
            details = {p: rows[left:end] for p, rows in metrics.items()}
            first = max(self.window_size, warm - 1 + lag) if training else start - left
            return self._build_env(bars, pairs, types, feats, details, dims, first, initial)

        end = len(combined["Close"])
        split = n_train + n_val
        agent = None
        if not benchmark_only:
            agent = PPOAgent(self.config, environment(0, n_train, self.initial_capital, training=True))
            model_dir = Path(self.train_cfg.get("save_dir", "models/ppo")) / f"window_{window:03d}"
            agent.train(self.config["rl"]["hyperparameters"]["timesteps"],
                        log_interval=self.train_cfg.get("log_interval", 10), save_dir=str(model_dir))

        def evaluate(start, stop, initial, policy=None):
            env = environment(start, stop, initial)
            if policy == "cash":
                actor = CashPolicy(env.action_space.shape[0])
            elif policy == "random":
                actor = RandomPolicy(env.action_space.shape[0], seed=self.config["system"]["seed"])
            elif policy == "zscore_mean_reversion" or agent is None:
                actor = ZScoreMeanReversionPolicy(env)
            else:
                actor = agent
            return Backtester(actor, env, self.config).run()

        validation = evaluate(n_train, split, self.initial_capital) if n_val else {}
        primary = evaluate(split, end, capital)
        evaluated = {name: evaluate(split, end, value, name) for name, value in baseline_capitals.items()}
        return primary, validation, evaluated, models

    def run(self):
        from backtest.metrics import compute_all_metrics
        run_baselines = self.config.get("validation", {}).get("run_baselines", True)
        names = ("cash", "random", "zscore_mean_reversion") if run_baselines else ()
        records = {name: [] for name in ("strategy", *names)}
        capitals = dict.fromkeys(records, self.initial_capital)
        windows = []
        self.selected_pair_history = []
        for window, fold in enumerate(self._generate_validation_windows(), 1):
            train, val, test, _, _, _, start, val_start, test_start, end = fold
            primary, validation, baselines, models = self._evaluate_window(
                train, val, test, capitals["strategy"], {k: capitals[k] for k in names}, window)
            for name, result in {"strategy": primary, **baselines}.items():
                records[name].append(result)
                capitals[name] = result["equity_curve"].iloc[-1]
            summary = {"window": window, "train_start": start, "train_end": train["Close"].index[-1],
                       "validation_start": val_start if self.validation_days else None,
                       "test_start": test_start, "end": end, "pairs": [m["pair"] for m in models]}
            self.selected_pair_history.append({**summary, "models": models,
                **{f"{kind}_pairs": [m["pair"] for m in models if m["type"] == kind]
                   for kind in ("inverse", "traditional")}})
            windows.append({**summary, "validation_metrics": validation.get("metrics", {}),
                            "test_metrics": primary["metrics"], "metrics": primary["metrics"],
                            "baseline_metrics": {k: v["metrics"] for k, v in baselines.items()}})
        if not windows:
            raise RuntimeError("No complete walk-forward windows fit the available calendar")

        def aggregate(rows):
            equity = pd.concat([r["equity_curve"] for r in rows], verify_integrity=True)
            spy = self.config["benchmark_curve"].reindex(equity.index)
            spy = spy / spy.iloc[0]
            pnls = equity.diff().dropna().tolist()
            costs = [r["frictions"].assign(window=i) for i, r in enumerate(rows, 1) if "frictions" in r]
            contributions = {}
            for r in rows:
                for pair, values in r.get("pair_pnls", {}).items():
                    contributions[pair] = contributions.get(pair, 0.0) + sum(values)
            return {"equity_curve": equity, "spy_curve": spy, "bar_pnls": pnls,
                    "frictions": pd.concat(costs, ignore_index=True) if costs else pd.DataFrame(),
                    "pair_contributions": contributions, "metrics": compute_all_metrics(equity, spy, pnls)}

        primary = aggregate(records.pop("strategy"))
        baselines = {name: aggregate(rows) for name, rows in records.items()}
        return {**primary, "window_metrics": windows, "selected_pairs": self.selected_pair_history,
                "validation_protocol": "walk_forward_train_validation_test" if self.validation_days else "walk_forward_train_test",
                **{key: {name: result[field] for name, result in baselines.items()} for key, field in (
                    ("baseline_equity_curves", "equity_curve"), ("baseline_spy_curves", "spy_curve"),
                    ("baseline_global_metrics", "metrics"), ("baseline_bar_pnls", "bar_pnls"),
                    ("baseline_pair_contributions", "pair_contributions"), ("baseline_frictions", "frictions"))}}
