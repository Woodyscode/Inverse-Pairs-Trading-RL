# Pairs Premise Comparison Report

## Protocol
- Data: fixed daily yfinance cache from `outputs/daily_validation/cache/yfinance_daily_500plus_2014_2024.csv`.
- Universe: 936 requested symbols, 744 retained full-period symbols.
- Walk-forward: same daily validation protocol, two train/validation/test windows.
- Pair selection: train window only.
- Baselines: cash, deterministic random policy, residual z-score mean reversion.
- Costs/frictions: same transaction, slippage, borrow, financing, and exposure assumptions as daily validation.
- PPO was not run unless z-score passed a predeclared evidence gate: positive return, positive Sharpe, profit factor above 1.05, and at least 50% cointegration persistence.

## Summary
At least one z-score family produced positive return in this two-window pivot, but the evidence is not yet robust enough for tuning.

- Best z-score variant: `inverse_corr_only` with total return `0.0118548`.
- Best inverse z-score total return: `0.0118548`.
- Best positive z-score total return: `0.00181312`.
- PPO decision: inverse_corr_only: skipped_zscore_gate_not_met; inverse_corr_plus_cointegration: skipped_zscore_gate_not_met; positive_corr_only: skipped_zscore_gate_not_met; positive_corr_plus_cointegration: skipped_zscore_gate_not_met; positive_sector_industry_cointegration: skipped_zscore_gate_not_met; positive_etf_family_cointegration: skipped_zscore_gate_not_met.

## Baseline Results
| variant | policy | total_return | sharpe | max_drawdown | profit_factor | turnover | total_costs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| inverse_corr_only | cash | 0 | 0 | 0 |  | 0 | 0 |
| inverse_corr_plus_cointegration | cash | 0 | 0 | 0 |  | 0 | 0 |
| positive_corr_only | cash | 0 | 0 | 0 |  | 0 | 0 |
| positive_corr_plus_cointegration | cash | 0 | 0 | 0 |  | 0 | 0 |
| positive_sector_industry_cointegration | cash | 0 | 0 | 0 |  | 0 | 0 |
| positive_etf_family_cointegration | cash | 0 | 0 | 0 |  | 0 | 0 |
| positive_corr_only | random | 0.0187087 | 1.90581 | 0.0049561 | 1.52669 | 2.21302e+06 | 569 |
| positive_sector_industry_cointegration | random | 0.00374983 | 0.705383 | 0.00786962 | 1.13915 | 2.19315e+06 | 563.898 |
| inverse_corr_only | random | 0.000450096 | 0.0348689 | 0.0285191 | 1.00689 | 2.20089e+06 | 565.881 |
| inverse_corr_plus_cointegration | random | -0.00866241 | -0.327281 | 0.0370842 | 0.930701 | 2.19119e+06 | 563.411 |
| positive_corr_plus_cointegration | random | -0.00913945 | -1.01532 | 0.0158291 | 0.762737 | 2.1932e+06 | 563.929 |
| positive_etf_family_cointegration | random | -0.0103409 | -1.13446 | 0.0170083 | 0.747768 | 2.19286e+06 | 563.84 |
| inverse_corr_only | zscore | 0.0118548 | 0.551985 | 0.019189 | 1.13981 | 960401 | 385.094 |
| positive_corr_only | zscore | 0.00181312 | 0.0998282 | 0.019139 | 1.01975 | 1.26011e+06 | 533.422 |
| positive_corr_plus_cointegration | zscore | 0.000206671 | 0.0242774 | 0.0133489 | 1.00469 | 1.15265e+06 | 450.063 |
| positive_etf_family_cointegration | zscore | 0.000117283 | 0.0175498 | 0.0133489 | 1.00269 | 1.26277e+06 | 478.435 |
| positive_sector_industry_cointegration | zscore | -0.0166326 | -1.24306 | 0.0179495 | 0.79018 | 720496 | 419.255 |
| inverse_corr_plus_cointegration | zscore | -0.105482 | -1.83392 | 0.147894 | 0.613783 | 686106 | 374.924 |

## Cointegration Persistence
| variant | coint_persisted |
| --- | --- |
| inverse_corr_only | 0 |
| inverse_corr_plus_cointegration | 0.666667 |
| positive_corr_only | 0 |
| positive_corr_plus_cointegration | 0.333333 |
| positive_etf_family_cointegration | 0.333333 |
| positive_sector_industry_cointegration | 0.333333 |

## Variant Notes
- `inverse_corr_only` window 1: selected 3 pairs from 1630 candidates; coint candidates evaluated 0.
- `inverse_corr_only` window 2: selected 3 pairs from 1600 candidates; coint candidates evaluated 0.
- `inverse_corr_plus_cointegration` window 1: selected 3 pairs from 1630 candidates; coint candidates evaluated 750.
- `inverse_corr_plus_cointegration` window 2: selected 3 pairs from 1600 candidates; coint candidates evaluated 750.
- `positive_corr_only` window 1: selected 3 pairs from 9750 candidates; coint candidates evaluated 0.
- `positive_corr_only` window 2: selected 3 pairs from 8984 candidates; coint candidates evaluated 0.
- `positive_corr_plus_cointegration` window 1: selected 3 pairs from 9750 candidates; coint candidates evaluated 750.
- `positive_corr_plus_cointegration` window 2: selected 3 pairs from 8984 candidates; coint candidates evaluated 750.
- `positive_sector_industry_cointegration` window 1: selected 3 pairs from 81 candidates; coint candidates evaluated 81.
- `positive_sector_industry_cointegration` window 2: selected 3 pairs from 79 candidates; coint candidates evaluated 79.
- `positive_etf_family_cointegration` window 1: selected 3 pairs from 643 candidates; coint candidates evaluated 643.
- `positive_etf_family_cointegration` window 2: selected 3 pairs from 641 candidates; coint candidates evaluated 641.

## Answers
- Are inverse pairs structurally worse than positive pairs? This two-window daily pivot does not show that positive pairs are clearly better. The strongest z-score result was `inverse_corr_only`, but it had no cointegration persistence and was concentrated in inverse/leveraged ETP relationships, so it is not robust evidence for the inverse-pairs premise.
- Does cointegration persist out of sample? Mostly no. Train Engle-Granger filters did not reliably persist into test; persistence rates are shown above and should be treated as a core gating metric.
- Does z-score work on any family? Only weakly in `inverse_corr_only`; positive-correlation families were near flat or negative after costs. No result is strong enough to justify tuning PPO.
- Is PPO worth testing further? Not yet. PPO should wait until a baseline pair-selection family shows robust after-cost edge across more windows.
- Should the project become classical stat-arb? The next step should be classical stat-arb diagnostics, not PPO. The premise needs a stable residual process before RL adds value.
- Which pair-selection rule has the best out-of-sample evidence? `inverse_corr_only` by z-score total return in this pivot, subject to the small-sample caveat.

## Recommended Next Steps
1. Expand the walk-forward window count before drawing a conclusion.
2. Make cointegration persistence and validation ADF/half-life gates mandatory before test.
3. Exclude or isolate volatility/inverse/leveraged ETPs and compare results separately.
4. Use hedge-ratio residual PnL for classical pairs instead of the current equal-notional spread return approximation.
5. Only rerun PPO if a z-score family remains positive after costs across many windows.
