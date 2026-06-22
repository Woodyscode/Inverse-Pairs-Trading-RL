# Inverse Pairs Research Validation Hardening Report

Date: 2026-06-21

## Status

The codebase is now ready for a controlled full expensive validation run from an engineering-protocol standpoint: same-bar lookahead, PnL-based costs, backfill, all-zero test features, and missing validation split controls now have tests or protocol guards.

It is not yet ready for final research claims unless the full run uses a fixed real data snapshot instead of live Yahoo downloads. The local validation run in this report is synthetic and proves pipeline behavior, not edge.

## Exact Validation Protocol

The added rolling protocol is:

1. Build rolling windows as train, validation, and test.
2. Select pairs only on the train window.
3. Fit Engle-Granger and correlation filters only on train prices/returns.
4. Compute train features from train data only.
5. Compute validation/test features with trailing historical context, but emit only validation/test rows.
6. Fit feature selection and scaling only on train features/targets.
7. Transform validation/test with the train-fitted selector/scaler.
8. Train PPO only on train.
9. Evaluate validation before final test.
10. Do not use test for selection or tuning.

Small local validation config:

- `config/validation_smoke_config.yaml`
- Synthetic data, 320 business days.
- Rolling train length: 0.75 years.
- Validation length: 2 months.
- Test length: 2 months.
- PPO timesteps: 64.
- Baselines enabled.

## Files Changed

- `env/inverse_pairs_env.py`
- `backtest/walkforward.py`
- `backtest/backtester.py`
- `backtest/baselines.py`
- `backtest/metrics.py`
- `env/features/feature_selector.py`
- `config/config.yaml`
- `config/smoke_config.yaml`
- `config/validation_smoke_config.yaml`
- `research_validation.py`
- `tests/test_synthetic_research_validation.py`
- `tests/test_features_and_pairs.py`
- `tests/test_env_causality_and_costs.py`

Earlier repaired files remain part of the validation surface:

- `data/loaders.py`
- `data/preprocessing.py`
- `data/wavelet.py`
- `env/features/advanced.py`
- `utils/pairs_selection.py`
- `main_train.py`
- `requirements.txt`

## New Tests Added

Synthetic and protocol tests now cover:

- Known inverse/cointegrated pair passes pair selection.
- Non-cointegrated pair fails selection.
- Random policy on independent synthetic assets is not persistently profitable after costs.
- Manual price path produces the expected PnL and reward.
- Train/validation/test windows are ordered and non-overlapping.
- Feature scaler is fit on train only and reused on test.
- Borrow/financing costs and liquidity clipping are applied.

Full test command:

```bash
python -m pytest tests -q -p no:cacheprovider
```

Result:

```text
15 passed
```

Compile command:

```bash
python -m compileall agents backtest data env utils main_train.py research_validation.py
```

Result: pass.

## Trading Assumptions

Configured assumptions are explicit in config:

- Turnover-based transaction costs.
- Slippage as traded-notional bps/fraction.
- Annual borrow fee on short leg approximation.
- Annual financing rate on gross exposure.
- Optional cash credit.
- Max per-pair and total exposure caps.
- Optional liquidity cap based on traded notional as percent of trailing dollar volume.

Current limitations:

- Short leg is approximated as half of pair notional.
- Hedge-ratio leg sizing is still not fully implemented.
- Dividends and ETF distributions are not modeled.
- Borrow availability/locates are not modeled.
- Market impact is still simple bps/ADV capping, not an order book model.

## Baseline Results

From `outputs/research_validation/summary.json`, first validation-test window:

| Baseline | Final Value | Total Return | Sharpe | Max DD |
|---|---:|---:|---:|---:|
| Cash/no-trade | 100000.00 | 0.0000% | 0.000 | 0.0000% |
| Random policy | 99937.52 | -0.0625% | -1.380 | 0.1891% |
| Z-score mean reversion | 100021.79 | 0.0218% | 3.948 | 0.0050% |

These are synthetic smoke results only. They validate that baselines execute and are saved; they do not establish tradable performance.

## Ablation Results

All local synthetic ablations completed:

| Variant | Status | Final Value | Total Return | Sharpe | Windows |
|---|---|---:|---:|---:|---:|
| Base PPO | pass | 100000.05 | 0.000048% | 4.227 | 2 |
| No-wavelet / PPO wavelets disabled | pass | 100000.05 | 0.000048% | 4.227 | 2 |
| PPO without cointegration filter | pass | 100000.05 | 0.000048% | 4.227 | 2 |
| Strict inverse corr 0.8 | pass | 100000.05 | 0.000048% | 4.227 | 2 |

The ablations are intentionally tiny and nearly identical because the synthetic universe has one dominant inverse pair. They are protocol checks, not performance claims.

## Stress-Test Results

The runner records requested historical regimes:

- 2014-2019
- 2020
- 2021-2022
- 2023-2024

In this local run, all historical regime stress tests were marked `skipped_not_covered_by_local_synthetic_data`. The code path is prepared to run regime slices when a real local 2014-2024 data snapshot is available.

Stress output:

- `outputs/research_validation/summary.json`

## Commands Run

Pytest:

```bash
python -m pytest tests -q -p no:cacheprovider
```

Compile:

```bash
python -m compileall agents backtest data env utils main_train.py research_validation.py
```

Smoke mode:

```bash
python main_train.py --smoke
```

Synthetic validation and small walk-forward validation:

```bash
python research_validation.py --synthetic --small-wf
```

## Output Artifacts

- `outputs/research_validation/summary.json`
- `outputs/research_validation/base/metrics.json`
- `outputs/research_validation/base/window_metrics.json`
- `outputs/research_validation/base/equity_curve.csv`
- `outputs/research_validation/no_wavelet/`
- `outputs/research_validation/ppo_without_cointegration_filter/`
- `outputs/research_validation/strict_inverse_corr_0_8/`
- `outputs/smoke/metrics.json`
- `outputs/smoke/equity_curve.csv`

## Readiness Verdict

Ready for a full expensive validation run: **yes, with conditions**.

Conditions:

- Use a fixed, archived real data snapshot if results will be treated as research evidence.
- Run the full train/validation/test protocol, not the old train/test-only mode.
- Preserve all configs, selected pairs, metrics, equity curves, and data-quality reports.
- Treat full-run results as validation evidence only after comparing PPO against cash, random, z-score, no-wavelet, no-cointegration, and stricter inverse-correlation variants.

Not ready for final strategy-performance claims until hedge-ratio execution, dividends, borrow availability, financing details, and real regime stress tests are completed on fixed historical data.
