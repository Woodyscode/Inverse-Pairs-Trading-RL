# Inverse Pairs Trading Fix Plan

Updated: 2026-06-21 after minimal validation repair pass.

## Completed In This Repair Pass

- [x] Fix environment timing.
  - `env/inverse_pairs_env.py` now computes action PnL over `t -> t+1`, not the already observed `t-1 -> t` bar.
  - Regression tests added in `tests/test_env_causality_and_costs.py`.

- [x] Replace PnL-based costs with turnover-based costs.
  - Costs now apply to entry, exit, and position changes through traded notional.
  - Config now exposes `env.cost_model: "turnover"`.
  - Regression tests verify costs occur even when gross PnL is zero.

- [x] Remove backfill from the research data path.
  - `data/loaders.py` now uses forward-fill only, after raw min-history filtering.
  - Leading missing rows are dropped instead of backfilled.
  - Data-quality reports are written from loader metadata.

- [x] Make wavelet denoising causal when enabled.
  - `data/wavelet.py` now includes `causal_wavelet_denoise` and `causal_denoise_dataframe`.
  - `data/preprocessing.py` uses causal rolling wavelets, not full-sample denoising.
  - Future-mutation regression test added.

- [x] Fix all-zero test features.
  - `backtest/walkforward.py` computes test features with trailing train context and emits only test rows.
  - Feature warm-up is configurable through `features.min_periods`.
  - Guard raises if selected test features are all zero.

- [x] Validate train-only pair selection.
  - `utils/pairs_selection.py` aligns Engle-Granger inputs by timestamp.
  - Full config now defaults to inverse-only (`allow_positive: false`).
  - Regression test verifies test-window mutation does not affect selected pairs.

- [x] Add fast smoke mode.
  - `config/smoke_config.yaml` and `config/smoke_pairs.yaml` added.
  - `main_train.py --smoke` runs deterministic synthetic local data, tiny universe, no plots, no SPY download, and 64 PPO timesteps.
  - Smoke artifacts are saved under `outputs/smoke`.

- [x] Improve reproducibility controls.
  - PPO receives configured seed.
  - TensorBoard logging is optional and disabled by default.
  - Smoke outputs include metrics, equity curve, config snapshot, selected pairs, and model artifact.

- [x] Add validation tests.
  - New tests cover env causality, turnover costs, no backfill, causal wavelets, nonzero features with context, and train-only pair selection.

## Still Open

- [ ] Add a fixed real data snapshot.
  - The full study still uses Yahoo Finance by default. For research-grade reproducibility, add a versioned local data snapshot with checksums.

- [ ] Add formal validation protocol.
  - The code still has train/test walk-forward windows but no separate validation split for hyperparameter or pair-count decisions.

- [ ] Add hedge-ratio-based execution.
  - Current PnL still uses a simple return-spread approximation. Estimate and persist hedge ratios from training data and map pair weights to leg weights.

- [ ] Add realistic short/financing/dividend handling.
  - Borrow fees, locate constraints, financing rates, ETF distributions, and cash drag are still not modeled.

- [ ] Add liquidity and market-impact constraints.
  - Add ADV caps, spread estimates, per-leg notional caps, and market-impact assumptions.

- [ ] Add multiple-testing control for pair selection.
  - Engle-Granger is applied across many candidate pairs. Add FDR or another correction and persist p-values.

- [ ] Persist richer pair and feature diagnostics.
  - Save per-window correlations, p-values, hedge ratios, selected feature IDs, and feature importances.

- [ ] Repair or quarantine live trading path.
  - `main_live.py` and `live/live_env.py` still do not match the repaired training environment and should not be used for validation.

- [ ] Add end-to-end full-study artifact manifest.
  - Record git commit, dependency freeze, data snapshot ID, seed, command line, and runtime environment for full runs.

## Current Commands

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

## Current Status

The codebase is safer for a real rerun than it was after the audit: the known same-bar lookahead, PnL-based costs, backfill, and all-zero test-feature defects have targeted fixes and tests.

It should not yet be called fully research-valid. Treat the next full rerun as a corrected baseline experiment, not final performance evidence, until the remaining open items are addressed.
