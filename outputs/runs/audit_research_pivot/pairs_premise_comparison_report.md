# Pairs Premise Comparison Report

## Protocol
- Data: fixed daily yfinance cache from `outputs/daily_validation/cache/yfinance_daily_500plus_2014_2024.csv`.
- Each fold screens the 936 requested symbols using observations through its training cutoff.
- Walk-forward: same daily validation protocol, two train/validation/test windows.
- Pair selection: train window only.
- Baselines: cash, deterministic random policy, residual z-score mean reversion.
- Costs/frictions: same transaction, slippage, borrow, financing, and exposure assumptions as daily validation.
- PPO was not run unless z-score passed a predeclared evidence gate: positive return, positive Sharpe, profit factor above 1.05, and at least 50% cointegration persistence.

## Summary
At least one z-score family produced positive return in this two-window pivot, but the evidence is not yet robust enough for tuning.

- Best z-score variant: `inverse_corr_only` with total return `0.0526025`.
- Best inverse z-score total return: `0.0526025`.
- Best positive z-score total return: `-0.00207133`.
- PPO decision: inverse_corr_only: skipped_zscore_gate_not_met; inverse_corr_plus_cointegration: Missing price for CROC on 2022-05-26 in window 2; positive_corr_only: skipped_zscore_gate_not_met; positive_corr_plus_cointegration: skipped_zscore_gate_not_met; positive_sector_industry_cointegration: skipped_zscore_gate_not_met; positive_etf_family_cointegration: skipped_zscore_gate_not_met.

## Baseline Results
| variant | policy | status | total_return | sharpe | max_drawdown | bar_profit_factor | turnover | total_costs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inverse_corr_only | cash | complete | 0 | 0 | 0 |  | 0 | 0 |
| positive_corr_only | cash | complete | 0 | 0 | 0 |  | 0 | 0 |
| positive_corr_plus_cointegration | cash | complete | 0 | 0 | 0 |  | 0 | 0 |
| positive_sector_industry_cointegration | cash | complete | 0 | 0 | 0 |  | 0 | 0 |
| positive_etf_family_cointegration | cash | complete | 0 | 0 | 0 |  | 0 | 0 |
| inverse_corr_plus_cointegration | cash | Missing price for CROC on 2022-05-26 in window 2 |  |  |  |  |  |  |
| inverse_corr_only | random | complete | 0.00077328 | 0.0551594 | 0.0148627 | 1.0098 | 2.94555e+06 | 757.484 |
| positive_etf_family_cointegration | random | complete | -0.00426638 | -1.29825 | 0.00483082 | 0.757283 | 2.9402e+06 | 755.613 |
| positive_corr_plus_cointegration | random | complete | -0.00695222 | -8.93555 | 0.00716236 | 0.18968 | 2.93419e+06 | 754.016 |
| positive_corr_only | random | complete | -0.00760626 | -27.4772 | 0.00760626 | 0.00144416 | 2.93339e+06 | 753.833 |
| positive_sector_industry_cointegration | random | complete | -0.00890795 | -1.9845 | 0.00892661 | 0.69185 | 2.79522e+06 | 718.945 |
| inverse_corr_plus_cointegration | random | Missing price for CROC on 2022-05-26 in window 2 |  |  |  |  |  |  |
| inverse_corr_only | zscore | complete | 0.0526025 | 1.15563 | 0.0216832 | 1.27231 | 1.15583e+06 | 594.782 |
| positive_sector_industry_cointegration | zscore | complete | -0.00207133 | -0.151594 | 0.010735 | 0.970444 | 1.26725e+06 | 602.069 |
| positive_etf_family_cointegration | zscore | complete | -0.00608527 | -0.835571 | 0.00813067 | 0.831657 | 2.03773e+06 | 689.762 |
| positive_corr_plus_cointegration | zscore | complete | -0.00686977 | -5.09677 | 0.00730617 | 0.367016 | 2.72077e+06 | 795.943 |
| positive_corr_only | zscore | complete | -0.00769333 | -15.4283 | 0.00769333 | 0.0411235 | 3.17525e+06 | 832.139 |
| inverse_corr_plus_cointegration | zscore | Missing price for CROC on 2022-05-26 in window 2 |  |  |  |  |  |  |

## Cointegration Persistence
| variant | coint_persisted |
| --- | --- |
| inverse_corr_only | 0 |
| inverse_corr_plus_cointegration | 0 |
| positive_corr_only | 0.833333 |
| positive_corr_plus_cointegration | 0.833333 |
| positive_etf_family_cointegration | 0.333333 |
| positive_sector_industry_cointegration | 0.166667 |

## Variant Notes
- `inverse_corr_only` window 1: selected 3 pairs from 1845 candidates; coint candidates evaluated 0.
- `inverse_corr_only` window 2: selected 3 pairs from 1767 candidates; coint candidates evaluated 0.
- `inverse_corr_plus_cointegration` window 1: selected 3 pairs from 1845 candidates; coint candidates evaluated 750.
- `inverse_corr_plus_cointegration` window 2: selected 3 pairs from 1767 candidates; coint candidates evaluated 750.
- `positive_corr_only` window 1: selected 3 pairs from 23783 candidates; coint candidates evaluated 0.
- `positive_corr_only` window 2: selected 3 pairs from 23331 candidates; coint candidates evaluated 0.
- `positive_corr_plus_cointegration` window 1: selected 3 pairs from 23783 candidates; coint candidates evaluated 750.
- `positive_corr_plus_cointegration` window 2: selected 3 pairs from 23331 candidates; coint candidates evaluated 750.
- `positive_sector_industry_cointegration` window 1: selected 3 pairs from 153 candidates; coint candidates evaluated 153.
- `positive_sector_industry_cointegration` window 2: selected 3 pairs from 131 candidates; coint candidates evaluated 131.
- `positive_etf_family_cointegration` window 1: selected 3 pairs from 763 candidates; coint candidates evaluated 750.
- `positive_etf_family_cointegration` window 2: selected 3 pairs from 748 candidates; coint candidates evaluated 748.

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
4. Verify hedge-ratio stability across subsequent evaluation windows.
5. Only rerun PPO if a z-score family remains positive after costs across many windows.
