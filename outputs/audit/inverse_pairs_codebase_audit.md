# Inverse Pairs Trading Codebase Audit

Audit date: 2026-06-21

## Executive Summary

Final verdict: **invalid until fixed**.

The repository has the outline of a walk-forward inverse-pairs research pipeline, but the current implementation is not research-valid, not fully reproducible from a clean environment, and not free of major backtest inflation risk. The most serious defects are:

1. The environment has a timing leak: the agent observes features containing the current bar and is then credited with PnL for the return from the previous bar to that same current bar.
2. Transaction costs and slippage are charged as a fraction of realized PnL, not turnover or traded notional, so most trading costs are materially understated and can be zero on large rebalances.
3. Six-month test windows are shorter than the hardcoded 252-step feature warm-up, making all out-of-sample advanced features zero in the default walk-forward test path.
4. The data cleaner forward-fills and backfills before history filtering, creating synthetic pre-inception history and stale post-delisting history.
5. The configured "inverse pairs" study also allows positive-correlation pairs by default.
6. No durable result artifacts, non-empty README claims, validation split, deterministic run manifest, or completed saved outputs were found.

The code does select pairs inside each walk-forward training window, which is a good structural choice. However, the above issues are enough to invalidate any reported performance until corrected and rerun.

## Scope And Files Inspected

Inspected files and functions:

- `main_train.py`: `main`, `build_env`, `load_benchmark`.
- `main_backtest.py`: empty.
- `main_live.py`: live runner wiring.
- `config/config.yaml`: data, pair, PPO, walk-forward, cost, seed settings.
- `config/pairs.yaml`: current ticker universe.
- `data/loaders.py`: `clean_df`, `DataLoader.load_universe`, `DataLoader.load_prices`, `DataLoader.load_single_price`.
- `data/preprocessing.py`: `prepare_data`.
- `data/returns.py`: `compute_log_returns`, `normalize_returns`.
- `data/wavelet.py`: `wavelet_denoise`, `denoise_dataframe`.
- `utils/pairs_selection.py`: `find_strong_pairs`.
- `env/inverse_pairs_env.py`: `InversePairsTradingEnv._get_obs`, `step`, `reset`, `finalize_positions`.
- `env/features/advanced.py`: `compute_all_advanced_features`, rolling helpers, cointegration and ADF helpers.
- `env/features/targets.py`: `build_targets`.
- `env/features/feature_selector.py`: `FeatureSelector`.
- `agents/agent_base.py`: seed handling.
- `agents/ppo_agent.py`: PPO construction, training, prediction, evaluation.
- `backtest/walkforward.py`: `WalkForwardEngine`.
- `backtest/backtester.py`: `Backtester.run`, plotting, benchmark handling.
- `backtest/metrics.py`: metric calculations.
- `utils/stats.py`: unused alternate metrics implementation.
- `live/live_env.py`, `live/portfolio_tracker.py`, `live/alpaca_*`: live-path consistency check.
- `requirements.txt`, `README.md`.

No generated `outputs`, `results`, `models`, or non-empty report files were found before this audit. `README.md` and `main_backtest.py` are empty.

## Data Pipeline

### Where Prices Are Loaded From

Prices are loaded from Yahoo Finance through `yfinance`:

- `data/loaders.py:102` calls `yf.download` for the full universe.
- `data/loaders.py:174` calls `yf.download` for the SPY benchmark.
- The universe comes from `config/pairs.yaml` via `DataLoader.load_universe`.
- Config uses `start_date: "2014-01-01"` and `end_date: "2024-01-01"` in `config/config.yaml:5-6`.

### Timestamp Alignment

The loader returns OHLCV panels with common ticker columns across `Open`, `High`, `Low`, `Close`, `Adj Close`, and `Volume` (`data/loaders.py:136-147`). Because these panels come from one Yahoo download, their row index is normally aligned.

However, alignment quality is weakened by the cleaning method. `clean_df` replaces inf values, then forward-fills and backfills every column (`data/loaders.py:14-24`). This preserves a rectangular panel but can create artificial data where no real quote existed.

### Missing Data, Survivorship Bias, Delisted Assets

Critical issues:

- `clean_df` applies `.ffill().bfill()` before `min_history` filtering. A late-inception ETF with only partial real history can be backfilled to the start of the sample and then pass `min_history` (`data/loaders.py:24`, `data/loaders.py:127-131`).
- Backfill uses future observations to populate earlier timestamps. This is a direct lookahead in the data panel.
- Forward-fill after a ticker stops trading creates stale prices after delisting or suspension.
- The universe is a current static symbol list, not a point-in-time asset universe. Delisted assets, defunct share classes, and historical constituents are not handled.
- Yahoo Finance data is mutable and adjusted retroactively. There is no local raw data snapshot, checksum, or vendor-version manifest.

### Returns And Spreads

Returns:

- `prepare_data` uses `ohlcv["Close"]` at `data/preprocessing.py:44`, despite config declaring `price_type: "Adj Close"` at `config/config.yaml:9`.
- `compute_log_returns` uses `np.log(df / df.shift(1))` in `data/returns.py:5-10`.
- `normalize_returns` uses full-sample mean and standard deviation in `data/returns.py:14-20`.

Spread logic:

- Supervised targets use `spread = close[A] - close[B]` in `env/features/targets.py:13`.
- Environment PnL uses `spread_ret = ret_A - ret_B` in `env/inverse_pairs_env.py:182-187`.
- Feature computation uses `spread = adjA - adjB` in `env/features/advanced.py`.

Concerns:

- Raw price differences are not hedge-ratio adjusted. Cointegration results are not converted into hedge ratios for trading.
- The environment effectively trades `ret_A - ret_B`, while targets and some features use price-level spreads. These are different objects.
- The configured adjusted-close setting is ignored in the main preprocessing and pair-selection path.

## Pair Selection

### Negative-Correlation Filter

`find_strong_pairs` computes correlation on returns (`utils/pairs_selection.py:43-44`) and requires `abs(rho) >= correlation_threshold` (`utils/pairs_selection.py:68`). It then separates positive and negative pairs:

- Positive branch: `rho > 0 and allow_pos` at `utils/pairs_selection.py:74`.
- Negative branch: `rho < 0 and allow_neg` at `utils/pairs_selection.py:107`.

With `correlation_threshold: 0.8`, `allow_positive: true`, and `allow_negative: true` in `config/config.yaml:18-21`, the default study is not inverse-only. It includes traditional positive-correlation pairs too.

For a strict inverse-pairs study, `allow_positive` should be false and the inverse filter should be explicit, for example `rho <= -threshold`.

### Engle-Granger Cointegration

Cointegration uses `statsmodels.tsa.stattools.coint`:

- Imported at `utils/pairs_selection.py:3`.
- Called for positive pairs at `utils/pairs_selection.py:94`.
- Called for negative pairs at `utils/pairs_selection.py:126`.
- Config requires `cointegration_p_value: 0.05` at `config/config.yaml:24`.

Issues:

- It drops NaNs independently for each leg (`utils/pairs_selection.py:92-93`, `utils/pairs_selection.py:124-125`) instead of first aligning both price series on the same timestamp index.
- It uses `Close` prices in the walk-forward selector (`backtest/walkforward.py:246-254`), not the configured adjusted close.
- It does not record p-values, hedge ratios, residual stationarity diagnostics, or selected-pair metadata.
- It does not correct for multiple testing across thousands of pairs.
- Selected pairs are truncated by iteration order if max limits apply, not ranked by correlation strength, p-value, stability, liquidity, or out-of-sample robustness.

### Train-Only Selection And Leakage

The walk-forward engine does select pairs from the training OHLCV slice:

- Windows are generated at `backtest/walkforward.py:85-122`.
- Pair selection uses `ohl_train["Close"]` and `close_train.pct_change().dropna()` at `backtest/walkforward.py:246-254`.

This avoids direct full-sample pair selection in the main walk-forward path. That is one of the stronger parts of the code.

However:

- The underlying OHLCV panel may already contain backfilled future data from the loader.
- The code has no fixed train/validation/test split around 70/15/15. It uses rolling 3-year train and 6-month test windows (`config/config.yaml:79-80`).
- There is no validation set for model or hyperparameter selection.

## Wavelet Processing

### Where db4 Is Used

Wavelets are implemented in `data/wavelet.py`:

- `wavelet_denoise(series, wavelet="db4", level=3)` at `data/wavelet.py:6`.
- `pywt.wavedec` at `data/wavelet.py:10`.
- `pywt.waverec` at `data/wavelet.py:16`.
- Applied column-wise by `denoise_dataframe` at `data/wavelet.py:26-30`.
- Called in `prepare_data` at `data/preprocessing.py:61`.

### Leakage And Actual Use

The wavelet transform is a full-series transform. If those denoised returns are used as research features or selection inputs, it is noncausal and leaks future data.

In the current `main_train.py` to `WalkForwardEngine` path, the denoised `returns` are mostly used for window indexing and validation. Pair selection recomputes returns from `close_train`, and the RL features are computed from OHLCV slices. Therefore, the current walk-forward strategy does not appear to meaningfully use db4-denoised signals for trading. This creates two problems:

- If wavelets are intended to be part of the alpha signal, that claim is not actually implemented in the current walk-forward feature pipeline.
- If denoised returns are later wired into selection or features without changing the implementation, the current full-series wavelet code would introduce lookahead.

## Reinforcement Learning Environment

### Observation Timing

The environment observation is built from `self.features[pair][self.current_step]` (`env/inverse_pairs_env.py:115`). Feature row `t` was computed with `step=t+1` in `WalkForwardEngine._compute_features_for_window` (`backtest/walkforward.py:145`), and `compute_all_advanced_features` slices `ohlcv[k][A].iloc[:step]` (`env/features/advanced.py:211`).

Thus observation at `current_step=t` includes price information through row `t`.

### Step Logic Timing Leak

In `step`, the action is parsed first, positions are immediately overwritten, and PnL is then calculated from `current_step - 1` to `current_step`:

- Action parsed at `env/inverse_pairs_env.py:143-156`.
- Previous and current prices read at `env/inverse_pairs_env.py:177-180`.
- PnL calculated from those prices at `env/inverse_pairs_env.py:182-187`.
- `current_step` increments only after PnL and reward at `env/inverse_pairs_env.py:221`.

This is a critical same-bar lookahead. The agent sees current-bar features and then earns current-bar return.

Correct timing should be one of:

- Observe features through close `t`, choose action, fill at next bar, earn return `t -> t+1`.
- Observe features through close `t-1`, choose action, earn return `t-1 -> t`.

The current implementation mixes these and inflates backtest results.

### Reward

Reward is immediate PnL return minus a position penalty:

- `reward_pnl` in `utils/rewards.py:6-46`.
- Called in `env/inverse_pairs_env.py:208-214`.

The reward itself does not explicitly use future labels. But because environment PnL timing is wrong, reward inherits the same timing leak.

The stated reward context mentions return, Sharpe, and drawdown penalty. The actual reward is pure immediate PnL return minus a fixed `0.01 * position_size` penalty. It does not include Sharpe or drawdown.

### Action Space, Positions, Costs, Slippage, Holding Period

Action space:

- Continuous action `[direction_i, size_i]` per pair (`env/inverse_pairs_env.py:79-84`).
- Position is `direction * abs(size) * max_position` (`env/inverse_pairs_env.py:147-154`).

Position/PnL issues:

- Positions are overwritten every step, not opened and closed as trades.
- There is no average 5-day holding-period mechanism.
- There is no explicit cash ledger, borrow accounting, margin, locate availability, dividends, or financing.
- The PnL formula double-scales by position magnitude: `notional = abs(pos) * portfolio_value`, then `pnl = pos * notional * spread_ret` (`env/inverse_pairs_env.py:156`, `env/inverse_pairs_env.py:187`). A 25% position produces 6.25% effective exposure before spread return, not 25%.
- Costs are calculated as `transaction_cost * abs(total_pnl)` and `slippage_cost * abs(total_pnl)` (`env/inverse_pairs_env.py:195-196`). This is not realistic. Costs should be based on traded notional or turnover per leg. A large rebalance with zero PnL incurs zero cost in the current model.
- Pair type is only appended as an observation flag (`env/inverse_pairs_env.py:119-122`). It does not change execution, hedge ratio, or PnL mechanics.

### Episode Boundaries

Episodes start at `current_step = max(1, self.window)` in `env/inverse_pairs_env.py:240`. With `window_size: 30`, the first action at step 30 receives PnL from step 29 to 30 while seeing features through step 30.

Terminal condition returns a final zero-reward step when `current_step >= T - 2` (`env/inverse_pairs_env.py:133-139`). `Backtester.run` still appends this terminal portfolio value and zero PnL, which can slightly distort bar counts and trade metrics.

## Feature Pipeline And Test-Window Defect

Advanced features use a hardcoded warm-up:

- `if step < 252: return [0] * N_ADVANCED_PAIR_FEATURES` at `env/features/advanced.py:197-198`.

Default walk-forward test windows are six months:

- `test_months: 6` in `config/config.yaml:80`.
- `test_days = int(test_months * 21)` in `backtest/walkforward.py:91`, or about 126 trading days.

Because `_compute_features_for_window` computes test features on the isolated test slice (`backtest/walkforward.py:287`), every test feature row in default six-month windows has `step < 252` and is therefore all zeros. The policy sees only pair-type flags out of sample.

This makes the default walk-forward test unable to validate the feature set or PPO state dependence. It also makes any feature-driven edge claim unsupported.

## Backtest Validity

### Train/Validation/Test Split

The code does not implement the stated 70/15/15 split. It implements rolling walk-forward windows:

- 3 years training (`config/config.yaml:79`).
- 6 months testing (`config/config.yaml:80`).
- Rolling mode by default (`config/config.yaml:87`).

There is no validation set. PPO hyperparameters and feature-selection hyperparameters are fixed in config. The code does not show automatic test-set tuning, but it also does not persist a protocol that would prevent manual test-set iteration.

### Out-Of-Sample Status

Pair selection, feature selection, and PPO training are performed on the train window before testing on the subsequent test window. Structurally, this is intended to be out-of-sample.

But the out-of-sample claim is invalidated by:

- Same-bar environment timing leak.
- Backfilled data from the loader.
- Zeroed test features due the 252-step warm-up.
- Unrealistic costs and execution.

### Metrics

Metrics are computed in `backtest/metrics.py`.

Concerns:

- `Backtester.run` does not include the initial capital as the first equity-curve point. Metrics use `values[0]` as the start (`backtest/metrics.py:300`), so the first step's return is excluded from total return and CAGR.
- `trade_pnls` are per-step PnL observations, not closed trades (`backtest/backtester.py:48-70`). Win rate, payoff ratio, profit factor, and max consecutive losses are therefore bar-level metrics mislabeled as trade metrics.
- Years are approximated as `len(values) / 252` (`backtest/metrics.py:245`), not based on calendar span.
- Monthly returns use first and last observed equity within each month, not prior month-end to month-end.
- Benchmark alpha/beta logic is simple daily return regression without robust alignment checks for missing benchmark dates.

### Unrealistic Assumptions

The current backtest assumes:

- Perfect daily fills.
- No borrow constraints for short ETF/ETN legs.
- No short borrow fees, financing costs, dividend payments, hard-to-borrow constraints, or margin calls.
- No liquidity, ADV, spread, market impact, or order sizing constraints.
- No portfolio-level risk caps beyond a rough exposure clamp.
- No realistic turnover-based transaction costs.
- No stop-outs, stale quote handling, or corporate-action sanity checks.

These are not minor. For a long/short multi-asset ETF strategy, borrow, financing, dividends, and turnover-based costs can materially change results.

## Reproducibility

### Main Runner Scripts

Main intended runner:

```bash
python main_train.py
```

Other runners:

- `main_backtest.py` is empty.
- `main_live.py` exists but is not compatible with current config and agent APIs.

### Seeds

Seeds are partially set:

- `main_train.py` hardcodes `torch.manual_seed(42)`, `np.random.seed(42)`, and `random.seed(42)` at `main_train.py:105-107`.
- `AgentBase.set_seed` sets NumPy and PyTorch seeds at `agents/agent_base.py:66-70`.
- Config contains `system.seed: 42` at `config/config.yaml:172`.

Missing:

- `PPO(..., seed=seed)` is not passed in `agents/ppo_agent.py:68-84`.
- Environment action/observation spaces are not explicitly seeded.
- `DummyVecEnv` reset seed is not controlled.
- PyTorch deterministic settings are not configured.
- Yahoo Finance data is not snapshotted.

### Dependency Gaps

`requirements.txt` is missing dependencies used by the main path:

- `scikit-learn` is required by `env/features/feature_selector.py`.
- `psutil` is required by `main_train.py`.

Therefore a fresh `pip install -r requirements.txt` environment is not sufficient.

### Output Reproducibility

No output persistence was found for:

- Selected pairs per window.
- Pair correlations, p-values, hedge ratios.
- Feature-selection results.
- Equity curves.
- Per-window metrics.
- Final metrics JSON/CSV.
- Config snapshot, git commit, dependency freeze, and data snapshot metadata.

`config.evaluation.save_equity_curve` exists (`config/config.yaml:136`) but is not implemented in `Backtester` or `WalkForwardEngine`.

## Results Audit

No reported result artifacts were found in the repository. There are no existing files under `outputs`, `results`, `models`, or `logs`, and `README.md` is empty. I therefore could not compare code-generated results to README/report claims.

The code path logs metrics from `Backtester._print_metrics`, but does not save them. Even if metrics were observed from a run, they would not be trustworthy until the timing leak, cost model, data cleaning, and zero-test-feature bug are fixed.

Suspicious metric risks:

- Same-bar lookahead can create inflated Sharpe and smooth equity.
- Costs based on PnL rather than turnover can materially overstate returns.
- Per-bar PnL counted as trades can inflate apparent win-rate interpretability.
- Missing borrow/financing/dividend costs can overstate long/short ETF returns.

There is currently no reliable evidence of edge in this repository.

## Major Findings

1. Critical same-bar lookahead exists in the RL environment.
2. Default test features are all zero because the 252-step warm-up exceeds the 126-day test window.
3. Transaction costs and slippage are materially wrong.
4. Loader backfills and forward-fills before history filtering, creating artificial data.
5. Inverse-only research is not enforced by default because positive-correlation pairs are enabled.
6. Cointegration is not converted into tradable hedge-ratio-adjusted spreads.
7. Adjusted-close config is ignored in core preprocessing and pair selection.
8. There is no validation split and no persisted experiment manifest.
9. Requirements are incomplete.
10. No saved results were found.

## Critical Issues

Critical blockers:

- Fix environment timing before trusting any backtest.
- Fix cost model before evaluating profitability.
- Fix data cleaning before selecting pairs.
- Fix test feature generation before claiming out-of-sample PPO feature performance.
- Rerun all results after fixing the above.

## Data Leakage Risks

Confirmed:

- Same-bar observation-to-reward leak in `InversePairsTradingEnv.step`.
- Backfill in `clean_df` uses future data to fill earlier observations.

High risk:

- Full-series normalization and db4 wavelet denoising in `prepare_data` would leak if the resulting returns are used for model inputs or pair selection.
- Manual iteration on test windows is uncontrolled because there is no validation protocol or run manifest.

Not confirmed in the current walk-forward path:

- Direct full-sample pair selection. The selected pairs are chosen inside the train window.

## Backtest Realism Issues

The backtest is not execution-realistic. It lacks:

- Turnover-based costs.
- Per-leg slippage and bid/ask spread.
- Borrow fees and locate constraints.
- Dividend and financing treatment.
- Liquidity/ADV caps.
- Realistic fill timing.
- Hedge-ratio-based dollar allocations.
- Holding-period logic.
- Correct trade accounting.

## Reproducibility Status

Status: not reproducible enough for research claims.

The intended command is:

```bash
pip install -r requirements.txt
python main_train.py
```

But this is incomplete because:

- `requirements.txt` misses `scikit-learn` and `psutil`.
- The run depends on live Yahoo downloads.
- PPO seed is not passed to SB3.
- Results are not saved.
- Config and selected-pair metadata are not persisted.
- Default PPO training is 2,000,000 timesteps per window, making the full run expensive and hard to validate.

## Recommended Fixes Ranked By Priority

P0:

1. Correct the environment step timing so actions earn future returns, not already observed returns.
2. Replace PnL-based transaction costs with turnover-based per-leg costs and slippage.
3. Remove backfill from market data cleaning and perform min-history filtering before any fill.
4. Generate test features with trailing train history as context, or reduce warm-up/window logic so test features are not all zero.

P1:

5. Enforce inverse-only pair selection for the inverse study and persist selected-pair diagnostics.
6. Use adjusted prices consistently where configured.
7. Estimate and trade hedge-ratio-adjusted spreads.
8. Add validation windows or a clear train/validation/test protocol.
9. Save equity curves, metrics, selected pairs, config snapshots, dependency versions, and seeds under a run directory.

P2:

10. Add deterministic seeding for SB3, env spaces, and vector env resets.
11. Add missing dependencies to `requirements.txt`.
12. Add unit tests for data cleaning, pair selection, feature causality, env timing, costs, and metrics.
13. Disable or save plots non-interactively during batch runs.
14. Repair or quarantine the live runner until it matches the trained model and config.

## Final Verdict

**invalid until fixed**.

The repository has a plausible research scaffold, but the current implementation cannot support a valid conclusion about inverse pairs trading edge. The strategy should be rerun only after the timing leak, data-cleaning leak, feature warm-up bug, cost model, adjusted-price consistency, and reproducibility gaps are fixed.
