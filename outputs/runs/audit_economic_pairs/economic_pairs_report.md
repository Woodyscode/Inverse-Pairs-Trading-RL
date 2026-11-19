# Economic Pairs Experiment Report

## Protocol
- Data: fixed daily yfinance cache used by daily validation; the script raises if the cache is missing.
- Price type: `Adj Close`.
- Walk-forward: locked daily validation windows from `config/daily_validation_config.yaml`.
- Pair selection: predeclared economic families only; no broad correlation mining and no test-set selection.
- Baselines: cash, deterministic random, and residual z-score mean reversion.
- PPO: not run in this experiment.
- Normal ETFs are evaluated separately from volatility/inverse/leveraged ETPs.
- OOS survival diagnostic: test ADF p < 0.1, train/test Engle-Granger p < 0.1, and test half-life in `(0, 126]` trading days.

## Summary
- Best normal economic z-score family: `index_market_cap_etf` with total return `-0.00691684`.
- Best all-family z-score result: `volatility_hedge_isolated` with total return `0.293278`.
- Economically related pairs versus broad-mined pairs: Broad-mined comparison artifact was not found, so this report only evaluates economic families.

## Baseline Results
| family | family_type | policy | status | total_return | sharpe | max_drawdown | bar_profit_factor | turnover | total_costs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| same_sector_etf | normal_etf | cash | complete | 0 | 0 | 0 |  | 0 | 0 |
| commodity_related_etf | normal_etf | cash | complete | 0 | 0 | 0 |  | 0 | 0 |
| index_market_cap_etf | normal_etf | cash | complete | 0 | 0 | 0 |  | 0 | 0 |
| volatility_hedge_isolated | volatility_inverse_leveraged_isolated | cash | complete | 0 | 0 | 0 |  | 0 | 0 |
| commodity_related_etf | normal_etf | random | complete | 0.0063986 | 0.527061 | 0.00899989 | 1.10264 | 2.67115e+06 | 700.184 |
| index_market_cap_etf | normal_etf | random | complete | -0.00697832 | -6.13602 | 0.00708997 | 0.335839 | 2.65596e+06 | 696.45 |
| same_sector_etf | normal_etf | random | complete | -0.0096266 | -1.43281 | 0.0111968 | 0.780765 | 5.72964e+06 | 1492.08 |
| volatility_hedge_isolated | volatility_inverse_leveraged_isolated | random | complete | -0.0113106 | -0.252892 | 0.056878 | 0.943953 | 6.93416e+06 | 1855.17 |
| volatility_hedge_isolated | volatility_inverse_leveraged_isolated | zscore | complete | 0.293278 | 1.8002 | 0.060675 | 1.44476 | 4.02337e+06 | 1603.94 |
| index_market_cap_etf | normal_etf | zscore | complete | -0.00691684 | -2.37501 | 0.00748722 | 0.636538 | 1.899e+06 | 665.076 |
| commodity_related_etf | normal_etf | zscore | complete | -0.0122722 | -0.392434 | 0.0316693 | 0.933118 | 1.36314e+06 | 680.49 |
| same_sector_etf | normal_etf | zscore | complete | -0.0145097 | -0.802364 | 0.0289581 | 0.875848 | 2.45754e+06 | 1294.35 |

## Spread Diagnostics By Family
| family | family_type | pairs_tested | oos_survival_rate | coint_persistence_rate | median_test_adf_p | median_test_half_life | total_zscore_pair_pnl |
| --- | --- | --- | --- | --- | --- | --- | --- |
| volatility_hedge_isolated | volatility_inverse_leveraged_isolated | 20 | 0 | 0 | 0.296117 | 10.3338 | 30931.8 |
| index_market_cap_etf | normal_etf | 6 | 0.333333 | 0.333333 | 0.253459 | 10.9799 | -26.6076 |
| same_sector_etf | normal_etf | 12 | 0 | 0 | 0.559734 | 24.4344 | -156.612 |
| commodity_related_etf | normal_etf | 6 | 0 | 0 | 0.521191 | 13.6676 | -546.726 |

## Skipped Pairs
| family | window | pair | status |
| --- | --- | --- | --- |
| same_sector_etf | 1 | XLU-VPU | missing_symbols:VPU |
| same_sector_etf | 2 | XLU-VPU | missing_symbols:VPU |
| commodity_related_etf | 1 | URA-URNM | missing_symbols:URNM |
| commodity_related_etf | 1 | CCJ-URA | missing_symbols:CCJ |
| commodity_related_etf | 1 | CCJ-URNM | missing_symbols:CCJ,URNM |
| commodity_related_etf | 2 | URA-URNM | missing_symbols:URNM |
| commodity_related_etf | 2 | CCJ-URA | missing_symbols:CCJ |
| commodity_related_etf | 2 | CCJ-URNM | missing_symbols:CCJ,URNM |
| index_market_cap_etf | 1 | IWM-VB | missing_symbols:VB |
| index_market_cap_etf | 2 | IWM-VB | missing_symbols:VB |
| volatility_hedge_isolated | 1 | VIXY-VXX | missing_symbols:VXX |
| volatility_hedge_isolated | 1 | UVXY-VXX | missing_symbols:VXX |
| volatility_hedge_isolated | 1 | SVXY-VXX | missing_symbols:VXX |
| volatility_hedge_isolated | 1 | VXX-VIXM | missing_symbols:VXX |
| volatility_hedge_isolated | 1 | VXX-TVIXF | missing_symbols:VXX |
| volatility_hedge_isolated | 2 | VIXY-VXX | missing_symbols:VXX |
| volatility_hedge_isolated | 2 | UVXY-VXX | missing_symbols:VXX |
| volatility_hedge_isolated | 2 | SVXY-VXX | missing_symbols:VXX |
| volatility_hedge_isolated | 2 | VXX-VIXM | missing_symbols:VXX |
| volatility_hedge_isolated | 2 | VXX-TVIXF | missing_symbols:VXX |

## Answers
- Do economically related pairs mean-revert better than broad-mined pairs? Broad-mined comparison artifact was not found, so this report only evaluates economic families.
- Which pair families survive out of sample? Use `oos_survival_rate` above as the locked survival definition; individual pair/window outcomes are in `pair_diagnostics.csv`.
- Are volatility/inverse/leveraged ETPs responsible for fake results? Top isolated volatility/inverse/leveraged z-score family returned 29.3278%; top normal family returned -0.6917%. These products are reported separately and are not mixed into normal ETF evidence.
- Is there any pair family worth testing with PPO later? No. No normal economic family met a minimum after-cost baseline gate for PPO.

## Interpretation
- This experiment tests the economic-pair premise only. It does not tune entry thresholds, holding rules, hedge ratios, or PPO.
- Pair PnL uses the signed, training-fitted hedge ratio and gross exposure cap in the PPO research simulator.
