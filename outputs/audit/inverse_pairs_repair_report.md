# Inverse Pairs Minimal Repair Report

Repair date: 2026-06-21

## Summary

This pass implemented a minimal validation repair, not a strategy optimization. The known lookahead paths from the audit were addressed where they affected the runnable research pipeline:

- RL actions now earn PnL on the future `t -> t+1` interval after the observation at `t`.
- Costs are charged on turnover, not realized PnL.
- Loader backfill was removed from the research path.
- Test-window features now use trailing train context for warm-up and are checked for all-zero matrices.
- Pair selection remains train-window only and now aligns Engle-Granger inputs by timestamp.
- A deterministic local smoke mode was added.

The codebase is now safe enough to run a new engineering validation rerun. It is not yet a final research-valid system because remaining realism items, especially borrow/financing/dividends/liquidity and a formal validation protocol, still need work.

## Files Changed

- `env/inverse_pairs_env.py`
- `backtest/backtester.py`
- `backtest/metrics.py`
- `backtest/walkforward.py`
- `agents/ppo_agent.py`
- `data/loaders.py`
- `data/preprocessing.py`
- `data/wavelet.py`
- `env/features/advanced.py`
- `env/features/feature_selector.py`
- `utils/pairs_selection.py`
- `main_train.py`
- `config/config.yaml`
- `config/smoke_config.yaml`
- `config/smoke_pairs.yaml`
- `requirements.txt`
- `tests/test_env_causality_and_costs.py`
- `tests/test_data_and_wavelet.py`
- `tests/test_features_and_pairs.py`

## Bugs Fixed

1. Same-bar lookahead:
   `InversePairsTradingEnv.step` now observes at `current_step`, applies the action to target positions, and computes PnL from `current_step` to `current_step + 1`.

2. Cost model:
   Transaction cost and slippage are now based on traded notional/turnover:
   `abs(target_position - previous_position) * portfolio_value`.

3. Data backfill:
   `clean_df` no longer uses `bfill`. Market data filtering now counts raw non-null observations before fill and drops rows that still have leading missing values.

4. Data-quality reporting:
   Loader writes field/ticker missingness and kept/dropped metadata. Smoke report path:
   `outputs/audit/smoke_data_quality_report.csv`.

5. Test feature warm-up:
   Test feature computation now prepends trailing train context, computes causal features, and emits only test rows. A guard raises if any selected pair's test matrix is all zero.

6. Causal wavelets:
   Added `causal_wavelet_denoise` and `causal_denoise_dataframe`. Main preprocessing uses causal rolling db4 denoising only when enabled.

7. Pair selection:
   Engle-Granger inputs are timestamp-aligned with `prices_df[[a, b]].dropna()`. Default full-study config is now inverse-only with `allow_positive: false`.

8. Smoke mode:
   `python main_train.py --smoke` uses deterministic synthetic OHLCV data, a tiny universe, plots disabled, local benchmark, and 64 PPO timesteps.

9. Reproducibility:
   PPO receives the configured seed. Smoke run outputs are saved under `outputs/smoke`.

## Tests Added

- `tests/test_env_causality_and_costs.py`
  - Verifies actions cannot profit from already observed bars.
  - Verifies reward/PnL uses the future interval.
  - Verifies costs scale with turnover even when gross PnL is zero.

- `tests/test_data_and_wavelet.py`
  - Verifies `clean_df` never backfills.
  - Verifies leading missing rows are dropped rather than backfilled.
  - Verifies causal wavelet outputs do not change when future data changes.

- `tests/test_features_and_pairs.py`
  - Verifies train, validation-style, and test-style feature windows are nonzero when context is supplied.
  - Verifies pair selection is based on the training window and is unchanged by test-window mutations.

## Verification

All tests:

```bash
python -m pytest tests -q -p no:cacheprovider
```

Result:

```text
8 passed
```

Compile check:

```bash
python -m compileall agents backtest data env utils main_train.py
```

Result: pass.

Smoke pipeline:

```bash
python main_train.py --smoke
```

Result: pass.

Smoke artifacts:

- `outputs/smoke/equity_curve.csv`
- `outputs/smoke/metrics.json`
- `outputs/smoke/window_metrics.json`
- `outputs/smoke/selected_pairs.json`
- `outputs/smoke/config_used.yaml`
- `outputs/smoke/models/ppo/final_model.zip`
- `outputs/audit/smoke_data_quality_report.csv`

## Remaining Limitations

- Smoke mode is only a pipeline validation. It is not evidence of strategy performance.
- The full study still depends on live Yahoo downloads unless a real local data snapshot is added.
- Borrow costs, short locate availability, dividends, financing, liquidity/ADV caps, and market impact are still not modeled.
- The spread/PnL model is still a simple return-spread approximation, not a full hedge-ratio execution model.
- There is still no formal validation split for hyperparameter selection.
- Multiple-testing correction and selected-pair diagnostic persistence are still limited.
- The live-trading path remains inconsistent with the repaired training environment and should not be used as evidence.

## Commands

Run tests:

```bash
python -m pytest tests -q -p no:cacheprovider
```

Run local smoke validation:

```bash
python main_train.py --smoke
```

Run the full study later:

```bash
python main_train.py --config config/config.yaml
```

Before treating a full rerun as research evidence, use a fixed data snapshot and review the remaining limitations above.
