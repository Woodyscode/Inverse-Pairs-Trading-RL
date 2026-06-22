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
- Best normal economic z-score family: `commodity_related_etf` with total return `0.00169248`.
- Best all-family z-score result: `commodity_related_etf` with total return `0.00169248`.
- Economically related pairs versus broad-mined pairs: No clear improvement in this locked run: best normal economic z-score family `commodity_related_etf` returned 0.1692%, versus broad-mined `inverse_corr_only` at 1.1855%.

## Baseline Results
| family | family_type | policy | total_return | sharpe | max_drawdown | profit_factor | turnover | total_costs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| same_sector_etf | normal_etf | cash | 0 | 0 | 0 |  | 0 | 0 |
| commodity_related_etf | normal_etf | cash | 0 | 0 | 0 |  | 0 | 0 |
| index_market_cap_etf | normal_etf | cash | 0 | 0 | 0 |  | 0 | 0 |
| volatility_hedge_isolated | volatility_inverse_leveraged_isolated | cash | 0 | 0 | 0 |  | 0 | 0 |
| commodity_related_etf | normal_etf | random | 0.000350742 | 0.0498819 | 0.00674693 | 1.00805 | 2.04962e+06 | 535.007 |
| index_market_cap_etf | normal_etf | random | -0.00334264 | -2.09304 | 0.00392937 | 0.639441 | 2.05051e+06 | 535.246 |
| same_sector_etf | normal_etf | random | -0.00768811 | -1.19833 | 0.0125463 | 0.80307 | 4.20976e+06 | 1093.96 |
| volatility_hedge_isolated | volatility_inverse_leveraged_isolated | random | -0.102885 | -1.30842 | 0.109122 | 0.71991 | 6.7026e+06 | 1753.4 |
| commodity_related_etf | normal_etf | zscore | 0.00169248 | 0.0931209 | 0.0227152 | 1.0145 | 760576 | 449.864 |
| same_sector_etf | normal_etf | zscore | -0.00553171 | -0.412087 | 0.0108038 | 0.931713 | 1.56838e+06 | 876.134 |
| index_market_cap_etf | normal_etf | zscore | -0.00856412 | -2.66619 | 0.00921978 | 0.583422 | 969012 | 476.222 |
| volatility_hedge_isolated | volatility_inverse_leveraged_isolated | zscore | -0.209341 | -1.95264 | 0.298264 | 0.669159 | 2.35658e+06 | 1101.92 |

## Spread Diagnostics By Family
| family | family_type | pairs_tested | oos_survival_rate | coint_persistence_rate | median_test_adf_p | median_test_half_life | total_zscore_pair_pnl |
| --- | --- | --- | --- | --- | --- | --- | --- |
| commodity_related_etf | normal_etf | 6 | 0 | 0 | 0.738767 | 23.7335 | 619.23 |
| same_sector_etf | normal_etf | 12 | 0 | 0 | 0.593486 | 22.6323 | 322.307 |
| index_market_cap_etf | normal_etf | 6 | 0 | 0 | 0.698485 | 29.9891 | -381.854 |
| volatility_hedge_isolated | volatility_inverse_leveraged_isolated | 20 | 0 | 0 | 0.452279 | 20.0679 | -20863.5 |

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
- Do economically related pairs mean-revert better than broad-mined pairs? No clear improvement in this locked run: best normal economic z-score family `commodity_related_etf` returned 0.1692%, versus broad-mined `inverse_corr_only` at 1.1855%.
- Which pair families survive out of sample? Use `oos_survival_rate` above as the locked survival definition; individual pair/window outcomes are in `pair_diagnostics.csv`.
- Are volatility/inverse/leveraged ETPs responsible for fake results? Top isolated volatility/inverse/leveraged z-score family returned -20.9341%; top normal family returned 0.1692%. These products are reported separately and are not mixed into normal ETF evidence.
- Is there any pair family worth testing with PPO later? No. No normal economic family met a minimum after-cost baseline gate for PPO.

## Interpretation
- This experiment tests the economic-pair premise only. It does not tune entry thresholds, holding rules, hedge ratios, or PPO.
- Pair PnL uses the same equal-notional spread approximation as the prior pivot runner; hedge-ratio-weighted execution remains a separate implementation improvement.
