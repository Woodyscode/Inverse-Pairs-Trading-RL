# Daily Validation Diagnostic Report

## Executive Summary

The daily validation produced no evidence of edge for two different reasons:

1. `ppo_base` effectively did not trade. It had 190 tiny turnover events, but only `$65.52` of total turnover on `$100,000` capital and average gross exposure of `0.000735%` of NAV. Its final PnL was `$0.38`. The high Sharpe is meaningless because the equity curve is almost flat.
2. The active baseline, `zscore_mean_reversion`, traded enough but lost because the selected inverse spreads did not mean revert out of sample. Its loss was concentrated in window 1 on equity-vs-volatility ETP pairs. Costs were not the main cause.

The pair selector did select negative-correlation, train-period Engle-Granger passing pairs, but the train signal did not persist out of sample. In window 1, the selected pairs' test correlations weakened from around `-0.61` to `-0.26/-0.30`, and every base-pair test cointegration p-value was above `0.10`. Test ADF p-values also failed stationarity for all base pairs.

The strongest diagnosis is: pair selection is unstable for inverse pairs, and PPO currently collapses to near-zero exposure. The daily run is mechanically valid, but not economically informative as evidence of a tradable edge.

## Data Snapshot

- Source: `yfinance_cache`
- Symbols requested: `936`
- Symbols retained after full-period coverage and history filters: `744`
- Date range: `2014-01-02` to `2023-12-29`
- Price type: `Adj Close`
- Daily validation windows executed: `2`
- Test periods actually traded after env warmup:
  - Window 1: `2017-08-17` to `2018-01-03`
  - Window 2: `2018-02-16` to `2018-07-05`

## Trade Counts And Exposure

| Strategy | Turnover events | Total turnover | Avg gross exposure | Max gross exposure | Active days | Total costs | Final PnL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ppo_base` | 190 | `$65.52` | `0.00000735` | `0.00002518` | 190 | `$0.03` | `$0.38` |
| `ppo_no_wavelet` | 190 | `$65.52` | `0.00000735` | `0.00002518` | 190 | `$0.03` | `$0.38` |
| `ppo_no_cointegration_filter` | 190 | `$53.15` | `0.00001776` | `0.00003348` | 190 | `$0.04` | `-$1.06` |
| `strict_inverse_corr_0_8` | 190 | `$109.17` | `0.00001277` | `0.00004001` | 190 | `$0.04` | `-$0.37` |
| `random_policy` | 190 | `$2,005,313.31` | `0.0753` | `0.1756` | 190 | `$529.64` | `$4,105.62` |
| `zscore_mean_reversion` | 58 | `$797,647.51` | `0.1916` | `0.3000` | 161 | `$470.44` | `-$15,574.94` |
| `cash_no_trade` | 0 | `$0.00` | `0.0000` | `0.0000` | 0 | `$0.00` | `$0.00` |

Interpretation:

- PPO did not trade enough in economic terms. The `active_positions` count is misleading because PPO held tiny nonzero positions in nearly every pair, but the position magnitudes were effectively zero.
- `random_policy` and `zscore_mean_reversion` both expressed real exposure. Their outcomes are therefore more diagnostic of the selected pairs than PPO's near-flat output.
- Transaction, borrow, and financing costs did not kill PPO. PPO simply did not take risk.
- Costs also did not explain the z-score loss: z-score lost about `$15,105` before costs and `$15,575` after costs.

## Pair Counts By Window

Every reported strategy used `3` pairs in each of the two windows. `cash_no_trade`, `random_policy`, and `zscore_mean_reversion` used the same pairs as `ppo_base`.

| Window | PPO base selected pairs |
| ---: | --- |
| 1 | `AME-TVIXF`, `AME-UVXY`, `AME-VIXM` |
| 2 | `ACN-VIXY`, `ADBE-VIXM`, `ADP-VIXM` |

These are all inverse pairs and are mostly equity or equity-like assets paired against volatility ETPs. That is a warning sign: volatility ETPs can have structural decay, reverse splits, path dependency, and regime-specific behavior that can pass in-sample statistical filters without producing a stable tradable spread.

## Pair Selection Diagnostics

The selected `ppo_base` pairs did satisfy the configured train-period inverse-correlation and Engle-Granger filters:

| Pair | Window | Train corr | Train coint p | Test corr | Test coint p | Test ADF p | Test half-life days | Test spread change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `AME-TVIXF` | 1 | `-0.626` | `0.063` | `-0.298` | `0.172` | `0.551` | `19.9` | `+81.7%` |
| `AME-UVXY` | 1 | `-0.634` | `0.063` | `-0.294` | `0.176` | `0.525` | `18.9` | `+79.3%` |
| `AME-VIXM` | 1 | `-0.607` | `0.047` | `-0.263` | `0.215` | `0.935` | `263.0` | `+66.0%` |
| `ACN-VIXY` | 2 | `-0.608` | `0.091` | `-0.709` | `0.178` | `0.120` | `10.8` | `-39.0%` |
| `ADBE-VIXM` | 2 | `-0.524` | `0.011` | `-0.622` | `0.724` | `0.664` | `25.7` | `+36.1%` |
| `ADP-VIXM` | 2 | `-0.586` | `0.044` | `-0.649` | `0.883` | `0.820` | `46.7` | `+15.0%` |

Average half-life:

- Train half-life, excluding invalid estimates: about `150` trading days.
- Test half-life, excluding invalid estimates: about `64` trading days.

This is not a clean 5-day mean-reversion profile. The selected spreads are too slow, unstable, or nonstationary for the intended holding horizon.

## Were Spreads Mean-Reverting Out Of Sample?

No robustly.

- All six `ppo_base` selected pairs had test Engle-Granger p-values above the configured `0.10` threshold.
- All six had test ADF p-values above `0.10`; none passed a conventional out-of-sample stationarity check.
- Window 1 was especially bad: the inverse return correlations weakened sharply and all three spreads moved strongly in one direction during test.
- Window 2 retained stronger negative correlation but still failed test cointegration/ADF on most pairs.

The pair selector found train-only statistical relationships, but those relationships did not survive the test windows.

## Cost And Turnover Decomposition

| Strategy | Gross PnL before costs | Transaction costs | Borrow cost | Financing cost | Total costs | Cost / abs gross PnL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ppo_base` | `$0.40` | `$0.01` | `$0.00` | `$0.01` | `$0.03` | `6.3%` |
| `random_policy` | `$4,635.25` | `$401.06` | `$14.29` | `$114.29` | `$529.64` | `11.4%` |
| `zscore_mean_reversion` | `-$15,104.50` | `$159.53` | `$34.55` | `$276.37` | `$470.44` | `3.1%` |

Costs mattered, but they are not the primary explanation:

- PPO costs were negligible because PPO exposure was negligible.
- Random survived high turnover and costs because the sampled direction exposure happened to benefit from the selected spreads.
- Z-score lost because gross spread PnL was deeply negative, not because costs overwhelmed a small positive edge.

## Window-Level Contribution

| Strategy | Window 1 PnL | Window 2 PnL | Total pattern |
| --- | ---: | ---: | --- |
| `ppo_base` | `+$0.29` | `+$0.09` | Flat in both windows |
| `random_policy` | `+$1,977.92` | `+$2,127.70` | Positive in both windows |
| `zscore_mean_reversion` | `-$16,582.17` | `+$1,007.22` | Loss concentrated in window 1 |
| `ppo_no_cointegration_filter` | `-$0.62` | `-$0.44` | Flat/slightly negative |
| `strict_inverse_corr_0_8` | `-$0.05` | `-$0.32` | Flat/slightly negative |

The z-score result is dominated by window 1. PPO is too flat for window-level performance to be meaningful.

## Pair-Level Contribution

For `ppo_base`, pair-level PnL was economically irrelevant:

| Pair | Window | PPO base PnL |
| --- | ---: | ---: |
| `AME-TVIXF` | 1 | `+$0.33` |
| `AME-UVXY` | 1 | `-$0.25` |
| `AME-VIXM` | 1 | `+$0.22` |
| `ACN-VIXY` | 2 | `+$0.04` |
| `ADBE-VIXM` | 2 | `+$0.04` |
| `ADP-VIXM` | 2 | `+$0.02` |

For `random_policy`, returns were concentrated in a few selected pairs:

| Pair | Window | Random PnL |
| --- | ---: | ---: |
| `AME-TVIXF` | 1 | `+$2,049.06` |
| `AME-UVXY` | 1 | `+$586.09` |
| `AME-VIXM` | 1 | `-$392.37` |
| `ACN-VIXY` | 2 | `+$2,004.71` |
| `ADBE-VIXM` | 2 | `+$233.00` |
| `ADP-VIXM` | 2 | `+$113.50` |

For `zscore_mean_reversion`, losses were also concentrated:

| Pair | Window | Z-score PnL |
| --- | ---: | ---: |
| `AME-TVIXF` | 1 | `-$7,119.33` |
| `AME-UVXY` | 1 | `-$6,006.73` |
| `AME-VIXM` | 1 | `-$3,236.21` |
| `ACN-VIXY` | 2 | `+$1,665.36` |
| `ADBE-VIXM` | 2 | `+$634.57` |
| `ADP-VIXM` | 2 | `-$841.94` |

The z-score baseline lost because it aggressively fought the window 1 spread trend in all three pairs.

## PPO Action And Exposure Distribution

Raw PPO actions were not saved, so an exact action histogram cannot be reconstructed from the current artifacts. The realized position/exposure logs are enough to diagnose the failure:

- `ppo_base` average gross exposure: `0.00000735` of capital, or about `0.000735%`.
- `ppo_base` max gross exposure: `0.00002518`, or about `0.002518%`.
- With three pairs and `max_position = 0.10`, PPO used less than `0.01%` of its available gross pair exposure budget.
- `ppo_no_wavelet` is exactly identical to `ppo_base`, which strongly suggests the trained policy/action scale collapsed to near zero and made feature differences irrelevant.

Conclusion: PPO learned or defaulted into a degenerate flat policy. This can happen because:

- The reward function penalizes position size and costs, while the short training run gives weak evidence for taking risk.
- The continuous action space allows tiny nonzero positions that count as active but are economically flat.
- PPO only trained for `256` timesteps in this cheap benchmark, so this run can validate mechanics but not learning quality.
- The environment does not currently save actions, so policy diagnostics are incomplete.

## Z-Score Entry/Exit Analysis

The z-score baseline policy is simple:

- Enter when `abs(zscore) >= 1.0`.
- Direction is `-sign(zscore)`.
- Size is full configured pair size, so up to `0.10` gross per pair.
- Exit when `abs(zscore) < 1.0`.
- No stop loss.
- No explicit maximum holding period.
- No regime or structural-break filter.

Observed behavior:

- `58` turnover days across `190` bar rows.
- `161` active days.
- Average gross exposure: `0.1916`.
- Maximum gross exposure: `0.3000`, meaning all three pairs could be active at max size.
- Window 1 had only `18` turnover days but `76` active days, so it entered and then held losing positions through persistent spread movement.

The baseline was not killed by overtrading. It was killed by taking full-size mean-reversion bets in spreads that did not mean revert out of sample.

## Why Random Outperformed PPO

`random_policy` outperformed `ppo_base` because it actually took exposure. It averaged `7.5%` gross exposure and turned over about `$2.0M`, while PPO averaged `0.000735%` gross exposure and turned over only `$65.52`.

Random did not prove edge. It got favorable realized signs on the selected equity/volatility ETP spreads in both windows. Its result is a reminder that this two-window sample is small and volatile; random exposure can look good by chance when selected spreads have strong directional movement.

## Why Z-Score Lost About 15.57%

Z-score lost because the selected spreads trended instead of reverting, especially in window 1:

- Window 1 z-score PnL: `-$16,582.17`.
- Window 2 z-score PnL: `+$1,007.22`.
- Window 1 selected spreads had weak test correlations and failed test cointegration/ADF.
- Window 1 test spread changes were roughly `+66%` to `+82%`.

The z-score policy therefore fought a trend with high exposure and no stop/timeout. Costs made the loss worse by about `$470`, but the gross strategy loss was already about `-$15,104`.

## Inverse Correlation Premise

The broad universe had many more positive-correlation candidates than inverse-correlation candidates:

| Window | Total pairs | Corr <= -0.5 | Corr <= -0.8 | Corr >= 0.5 | Corr >= 0.8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 276,396 | 1,630 | 180 | 48,218 | 3,711 |
| 2 | 276,396 | 1,600 | 179 | 44,390 | 3,556 |

This does not prove positive-correlation pairs are better, because positive-correlation spread trading was disabled in this run (`allow_positive: false`, `max_traditional_pairs: 0`). It does show that the inverse-correlation premise is a much narrower subset, and the selected inverse pairs were heavily influenced by volatility ETP behavior.

Inverse correlation alone is a weak pairs-trading premise. It can describe hedging relationships without implying spread convergence. For pairs trading, the tradable premise should be stable residual mean reversion, not just negative return correlation plus an in-sample Engle-Granger pass.

## Root Cause Assessment

Ranked likely causes:

1. Pair selection instability: train-only filters selected negative-correlation and cointegrated pairs, but out-of-sample stationarity failed.
2. PPO degenerate flat policy: PPO produced economically zero exposure, so it cannot show edge or no-edge.
3. Baseline spread definition: the environment and z-score baseline trade `ret_A - ret_B` / raw spread behavior without a robust hedge-ratio/log-spread model.
4. Exit logic: z-score has no stop, max hold, volatility-regime filter, or structural-break filter.
5. Feature pipeline relevance: wavelet denoising does not materially affect the current advanced price-feature path. The identical `ppo_base` and `ppo_no_wavelet` results confirm the wavelet ablation is not informative in this run.
6. Market regime/sample: only two test windows were run. They are useful for smoke validation but insufficient for final evidence.
7. Reward/action design: reward and action scaling allow the agent to minimize risk by shrinking position sizes to near zero.

## Answers To The Specific Questions

1. Did the strategy trade enough?
   - PPO: no. It generated tiny daily position changes but not meaningful notional exposure.
   - Z-score and random: yes, enough to diagnose the pair behavior.

2. Were selected pairs actually negatively correlated and cointegrated?
   - In train: yes, by the configured filters. All `ppo_base` pairs had train correlation below `-0.50` and train Engle-Granger p-value below `0.10`.
   - In test: not reliably. Test cointegration failed for all `ppo_base` pairs.

3. Were spreads mean-reverting out of sample?
   - No. Test ADF p-values failed stationarity for all `ppo_base` pairs, and several spreads trended strongly.

4. Was performance killed by costs, borrow/financing, turnover, or bad exits?
   - PPO: no, it was killed by near-zero exposure.
   - Z-score: mainly bad exits and bad/nonstationary spreads. Costs were secondary.

5. Did PPO avoid trading too much, trade randomly, or learn a degenerate flat policy?
   - It learned or defaulted into a degenerate flat policy. Raw actions were not saved, but realized exposure confirms this.

6. Why did random outperform ppo_base?
   - Random actually took risk and got favorable signs on selected pairs. PPO was nearly flat. This is not evidence that random has edge.

7. Why did zscore_mean_reversion lose about 15.57%?
   - It took full-size mean-reversion trades in window 1 against spreads that kept trending, especially `AME-TVIXF`, `AME-UVXY`, and `AME-VIXM`.

8. Is the issue pair selection, feature construction, reward design, action space, holding period, or market regime?
   - All contribute, but the top issues are pair selection instability and PPO flat-policy collapse. Holding/exit rules hurt z-score. Wavelet feature construction is currently not a meaningful lever.

9. Are returns concentrated in a few windows or pairs?
   - Yes. Z-score losses are concentrated in window 1. Random gains are concentrated in `AME-TVIXF` and `ACN-VIXY`. PPO returns are too small to interpret.

10. Does inverse correlation itself fail as a pairs-trading premise versus positive-correlation spread trading?
   - This run does not test positive-correlation spread trading. It does show inverse correlation by itself is not sufficient. Negative correlation plus in-sample cointegration did not produce stable out-of-sample mean reversion.

## Recommended Next Experiments

1. Add action logging and position logging by pair for PPO. Save raw action direction, raw action size, target position, realized position, reward, and per-pair PnL on every step.
2. Add a no-trade penalty or minimum-position diagnostic run, not as tuning, but to distinguish "agent found no edge" from "agent collapsed because flat policy is reward-optimal."
3. Replace raw price-difference spread diagnostics with hedge-ratio/log-spread diagnostics. Pair selection and z-score baseline should use the same spread definition as trading.
4. Require out-of-sample validation of spread stationarity before test. For example, select on train, choose thresholds on validation, and reject pairs whose validation residual ADF/half-life deteriorates.
5. Add structural-break and product-type filters. Treat leveraged/inverse/volatility ETPs separately or exclude them in a robustness run.
6. Add positive-correlation pairs as a controlled baseline with the same walk-forward protocol.
7. Enforce explicit holding-period rules matching the research premise, such as max hold around 5 trading days, stop loss, and time stop.
8. Run the same daily protocol across more walk-forward windows once diagnostics are instrumented. The current daily run uses only two test windows.
9. Make the wavelet ablation meaningful by confirming where denoised series enter pair selection, feature construction, or observations. Right now `ppo_no_wavelet` matching `ppo_base` suggests the ablation is not testing what it claims.
10. Re-run synthetic controls with the same broad-universe code path, including inverse pairs with known stationary residuals and inverse pairs with structural drift.

## Bottom Line

The daily validation failed to show edge because the active inverse-spread idea did not survive out-of-sample tests, and PPO did not take meaningful exposure. The next work should not tune hyperparameters. It should instrument actions, repair the spread definition, reject unstable spreads on validation data, and compare inverse pairs directly against positive-correlation spread trading.
