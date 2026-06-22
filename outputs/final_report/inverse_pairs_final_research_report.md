# Inverse Pairs Trading Final Research Report

Date: 2026-06-21

## Executive Summary

This project tested whether daily pairs trading could be built around inverse relationships: assets with strong negative correlation, Engle-Granger cointegration, wavelet-denoised features, and a PPO reinforcement-learning agent trading the spread.

The final empirical result is negative. After fixing known leakage paths, adding realistic turnover-based costs, locking a walk-forward protocol, expanding to a 500+ symbol daily universe, and comparing inverse, positive-correlation, and economically constrained pair families, no tested daily formulation produced durable after-cost edge.

The main research conclusion is not that PPO was poorly tuned. The simpler conclusion is more important: the underlying daily spread signals did not show robust out-of-sample mean reversion. PPO should remain paused until a simple baseline, such as z-score mean reversion, demonstrates reliable after-cost edge under the locked validation protocol.

Final verdict: research-valid but no evidence of tradable daily pairs edge.

## Original Hypothesis

The original hypothesis was:

- Multi-asset ETFs and assets contain inverse pairs with stable negative relationships.
- Strong negative correlation, roughly `-0.5` to `-1.0` or stricter `-0.8` to `-1.0`, can identify candidate inverse pairs.
- Engle-Granger cointegration can filter those pairs to spreads that mean-revert.
- Wavelet denoising and decomposition, especially `db4`, can improve signal quality.
- A PPO reinforcement-learning agent can learn to trade the spread better than static entry and exit rules.
- Reward shaping using return, Sharpe, and drawdown penalty can encourage robust risk-adjusted trading.

That hypothesis was tested and rejected for the current daily-bar formulation.

## Methodology

The research pipeline evolved through several stages:

1. Initial inverse-pairs PPO implementation.
2. Codebase audit for leakage, reproducibility, and backtest inflation.
3. Minimal validation repair pass.
4. Research-validation hardening with synthetic tests and walk-forward structure.
5. Broad daily validation on a large fixed yfinance snapshot.
6. Diagnostic analysis of the failed daily run.
7. Inverse-vs-positive pairs premise comparison.
8. Economically constrained pair-family experiment.

The final validation used daily end-of-day data, not minute data. Signals were treated as daily close-based information, with execution modeled on the next tradable bar. No intraday stops, limit-order fills, or minute-level execution claims were made.

## Data Universe And Period

Final daily validation used a fixed local yfinance cache:

- Cache: `outputs/daily_validation/cache/yfinance_daily_500plus_2014_2024.csv`
- Manifest: `outputs/daily_validation/data_manifest.json`
- Source: `yfinance_cache`
- Original source: `yfinance`
- Requested symbols: `936`
- Retained full-period symbols: `744`
- Requested date range: `2014-01-01` to `2024-01-01`
- Actual date range: `2014-01-02` to `2023-12-29`
- Rows per retained symbol: `2516`
- Price type: `Adj Close`
- Cache SHA-256: `90e46104c6f95fc81451f8947916cc60ee54221d6acc2f7fa1961c8650bffe16`

The fixed cache was used to avoid changing results through live yfinance downloads.

## Validation Protocol

The locked validation protocol used rolling train, validation, and test windows:

- Train window: 3 years.
- Validation window: 6 months.
- Test window: 6 months.
- Step size: 126 trading days.
- Windows executed in the daily validation config: 2.
- Pair selection occurred only inside the train window.
- Correlation filters and Engle-Granger tests were fit only on train data.
- Feature selection and scaling were fit only on train data.
- Validation could be used according to the locked protocol.
- Test was held out from pair selection, scaling, and model training.

Daily execution assumptions:

- Signals are computed using information available at or before close `t`.
- Actions affect PnL over the following `t -> t+1` interval.
- Costs are charged on traded notional and turnover, not on realized PnL.
- Configured frictions included transaction cost, slippage, borrow cost, financing cost, cash drag, position caps, and total exposure caps.

Relevant settings from the daily validation configuration:

- Per-pair max position: `10%`.
- Max total exposure: `50%`.
- Transaction cost: `0.0001`.
- Slippage cost: `0.0001`.
- Borrow fee: `0.5%` annualized.
- Financing rate: `2.0%` annualized.
- Cash rate: `0.0%` annualized.

## Codebase Hardening

The initial audit found the project was invalid until fixed. The major issues were same-bar lookahead, PnL-based transaction costs, backfilled market data, invalid all-zero features in some test windows, and expensive/unreproducible live-data full runs.

The repair and hardening passes addressed those issues:

- Fixed RL environment causality so an observation at `t` cannot earn same-bar PnL.
- Changed action execution so rewards are based on price movement after the action is taken.
- Replaced PnL-based costs with turnover/traded-notional costs.
- Applied costs on entry, exit, and position changes.
- Removed backfill from the research data path.
- Added data-quality reporting for missingness, alignment, and dropped symbols.
- Made `Close` versus `Adj Close` usage explicit and consistent through `config["data"]["price_type"]`.
- Added causal `db4` wavelet processing rather than full-series future-aware denoising.
- Added train-context warmup for validation/test feature windows to avoid all-zero advanced features.
- Added synthetic-data tests for pair selection, non-cointegration rejection, manual reward paths, and independent-random assets.
- Added rolling train/validation/test validation logic.
- Added cash, random, z-score, no-wavelet, and no-cointegration baselines.
- Added stress-test and ablation runners.
- Added deterministic smoke mode with tiny PPO timesteps for pipeline validation.

Verification status from the hardening pass:

- Tests passed: `19 passed`.
- Compile check passed.
- Smoke mode passed.
- Synthetic validation passed.
- Small walk-forward validation passed.
- Ablation runner passed.

## Baselines

The final research did not rely on PPO alone. It compared PPO against:

- Cash/no-trade.
- Deterministic random policy.
- Simple z-score mean reversion.
- PPO base.
- PPO without wavelets.
- PPO without cointegration filter.
- Strict inverse-correlation filter.
- Inverse-correlation-only pair selection.
- Inverse-correlation plus cointegration.
- Positive-correlation-only pair selection.
- Positive-correlation plus cointegration.
- Sector/industry constrained positive pairs.
- ETF-family constrained positive pairs.
- Economically constrained ETF families.
- Isolated volatility/inverse/leveraged ETP families.

The project intentionally did not tune PPO after seeing results.

## Final Results

### Broad Daily Inverse-Pairs Validation

The main daily validation results were:

| Strategy | Total Return | Sharpe | Max Drawdown | Profit Factor | Total Costs | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ppo_base` | `0.000377%` | `1.698` | `0.000143%` | `1.432` | `$0.03` | Economically flat, near-zero exposure |
| `cash_no_trade` | `0.000%` | `0.000` | `0.000%` |  | `$0.00` | Baseline |
| `random_policy` | `4.106%` | `1.442` | `1.465%` | `1.334` | `$529.64` | Took exposure and benefited from sampled spread movement |
| `zscore_mean_reversion` | `-15.575%` | `-2.004` | `21.316%` | `0.698` | `$470.44` | Active baseline failed materially |
| `ppo_no_wavelet` | `0.000377%` | `1.698` | `0.000143%` | `1.432` | `$0.03` | Identical to PPO base |
| `ppo_no_cointegration_filter` | `-0.001063%` | `-0.807` | `0.001451%` | `0.834` | `$0.04` | Economically flat |
| `strict_inverse_corr_0_8` | `-0.000372%` | `-1.046` | `0.000390%` | `0.801` | `$0.04` | Economically flat |

The PPO Sharpe is not economically meaningful because PPO almost did not trade. Its total turnover was about `$65.52` on `$100,000` capital, with average gross exposure around `0.000735%` of NAV.

The active z-score baseline traded enough to be informative and lost about `-15.57%`, mainly because selected inverse spreads failed out of sample.

### Daily Diagnostic Findings

The selected inverse pairs passed train-period negative-correlation and Engle-Granger filters, but did not remain stationary or cointegrated in test:

- Daily validation selected equity or equity-like assets paired against volatility ETPs.
- Train negative correlations weakened out of sample in the worst window.
- All six base selected pairs had test Engle-Granger p-values above `0.10`.
- All six base selected pairs had test ADF p-values above `0.10`.
- Average half-life estimates were too slow or unstable for the intended roughly 5-day holding horizon.
- The z-score baseline lost because gross spread PnL was negative before costs, not because costs alone overwhelmed a positive edge.

Random policy outperformed PPO because it took real exposure and happened to benefit from the selected spread movement. That is not evidence of a stable edge.

### Inverse Versus Positive Pairs Premise Comparison

The broad premise comparison tested inverse and positive pair-selection variants under the same walk-forward protocol and frictions.

| Variant | Policy | Total Return | Sharpe | Profit Factor | PPO Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `inverse_corr_only` | z-score | `1.185%` | `0.552` | `1.140` | Skipped |
| `inverse_corr_plus_cointegration` | z-score | `-10.548%` | `-1.834` | `0.614` | Skipped |
| `positive_corr_only` | z-score | `0.181%` | `0.100` | `1.020` | Skipped |
| `positive_corr_plus_cointegration` | z-score | `0.021%` | `0.024` | `1.005` | Skipped |
| `positive_sector_industry_cointegration` | z-score | `-1.663%` | `-1.243` | `0.790` | Skipped |
| `positive_etf_family_cointegration` | z-score | `0.012%` | `0.018` | `1.003` | Skipped |

The best broad result was `inverse_corr_only`, but it had no cointegration persistence and was concentrated in inverse/leveraged ETP relationships. It was not strong enough to justify PPO.

The stronger inverse formulation, `inverse_corr_plus_cointegration`, lost materially.

### Economically Constrained Pair Families

The final pivot tested economically related pairs rather than broad statistical mining.

| Family | Policy | Total Return | Sharpe | Profit Factor | OOS Survival Rate |
| --- | --- | ---: | ---: | ---: | ---: |
| `commodity_related_etf` | z-score | `0.169%` | `0.093` | `1.014` | `0.000` |
| `same_sector_etf` | z-score | `-0.553%` | `-0.412` | `0.932` | `0.000` |
| `index_market_cap_etf` | z-score | `-0.856%` | `-2.666` | `0.583` | `0.000` |
| `volatility_hedge_isolated` | z-score | `-20.934%` | `-1.953` | `0.669` | `0.000` |

Economically constrained pairs did not rescue the daily strategy. The best normal family, commodity-related ETFs, returned only `0.169%` with weak Sharpe and profit factor. No family survived the locked out-of-sample diagnostic definition requiring stationarity, cointegration persistence, and finite half-life.

Volatility, inverse, and leveraged ETPs were isolated from the normal ETF experiment. Their isolated z-score result was strongly negative, which supports treating those products as a separate, higher-risk research domain rather than mixing them with ordinary ETF pairs.

## Why The Strategy Failed

The evidence points to several failure modes.

First, train-period statistical relationships did not persist. The core inverse pair selection logic found relationships that looked plausible in-sample but failed out-of-sample stationarity and cointegration diagnostics.

Second, inverse correlation was not enough. Negative return correlation can come from regime exposure, volatility products, leverage decay, or macro beta rather than a stable residual spread.

Third, Engle-Granger was not sufficient as a one-time train-window filter. Some pairs passed train cointegration but failed in test. Cointegration persistence should be a research target, not assumed from a single in-sample p-value.

Fourth, the intended holding horizon did not match the measured spread behavior. The strategy targeted short average holds, roughly 5 days, but estimated half-lives were often slower, unstable, or invalid.

Fifth, PPO had no validated signal to learn from. The trained PPO policy collapsed to near-zero exposure. Since static z-score mean reversion did not show robust edge, tuning PPO would likely fit noise.

Sixth, wavelet features did not change the active decision path. PPO base and PPO no-wavelet were identical in the broad daily validation because the policy was effectively flat.

Seventh, costs were realistic enough to matter, but they were not the main cause of failure. The z-score baseline lost materially before costs; costs made bad trades worse but did not convert a positive strategy into a negative one.

## What This Teaches

The most important lesson is that research infrastructure must be able to reject a strategy. This project moved from an initially invalid backtest to a stricter validation framework that could produce a credible negative result.

Specific lessons:

- Fix causality before interpreting performance.
- Establish a simple baseline edge before adding reinforcement learning.
- Treat high Sharpe on tiny exposure as meaningless.
- Verify out-of-sample stationarity and cointegration persistence, not just train-period filters.
- Separate instrument classes with different structural behavior, especially volatility and leveraged ETPs.
- Use fixed data snapshots and manifests for reproducibility.
- A random policy beating a trained model is a diagnostic warning, not a trading signal.
- A failed alpha hypothesis can still produce valuable engineering and research process.

## Limitations

This research does not prove that all pairs trading is dead or that no intraday/stat-arb variant can work. It only rejects the tested daily-bar formulations under this codebase and validation protocol.

Known limitations:

- Daily data only; no minute-level execution or intraday spread dynamics.
- yfinance data is convenient but not institutional-grade and may still carry survivorship or historical adjustment limitations.
- The broad daily validation used two walk-forward test windows in the locked config, which is enough for a first benchmark but not enough for final production research.
- Some economically interesting symbols were unavailable under the full-period retained-symbol filter.
- Hedge-ratio residual execution is not fully implemented; the pivot runners use an equal-notional spread approximation.
- ETF distributions, tax effects, locate availability, hard-to-borrow dynamics, and product closure risk are not fully modeled.
- Slippage and financing are simple configurable assumptions, not order-book or broker-specific models.
- PPO hyperparameters were intentionally not tuned after seeing negative baseline results.

## Future Research Directions

Future work should not start by tuning PPO. The next research steps should be:

1. Expand walk-forward windows across the full daily period using a locked commercial or delisting-aware dataset.
2. Add hedge-ratio-weighted residual execution and compare it against equal-notional spread PnL.
3. Require validation-window evidence of stationarity, cointegration persistence, half-life, and spread-volatility stability before final test.
4. Test economically motivated baskets and sector-neutral residuals rather than pair-only mining.
5. Separate ordinary ETFs, single-name equities, leveraged/inverse ETPs, volatility ETPs, and commodity products into separate research universes.
6. Add stricter liquidity, borrow, and short-availability constraints.
7. Explore regime filters only after the base spread process is validated.
8. Move to minute data only if daily baselines show a reason to pay for and model intraday execution.
9. Reintroduce PPO only if a simple baseline has robust after-cost edge across many windows.

## Final Verdict

Final verdict: research-valid but no edge.

The repaired and hardened framework is useful. It loads a fixed daily snapshot, avoids the known leakage paths, applies explicit costs and frictions, runs baselines, produces diagnostics, and can reject invalid strategies.

The strategy itself should not be traded and should not be presented as profitable. Across inverse pairs, positive-correlation pairs, broad statistical mining, and economically constrained pair families, the daily tests found no durable after-cost edge. PPO should remain paused until a simple, transparent baseline demonstrates robust out-of-sample profitability.
