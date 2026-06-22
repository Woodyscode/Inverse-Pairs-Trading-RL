# Inverse Pairs Daily Validation Report

## Protocol
- Data frequency: daily end-of-day bars only.
- Price type: `Adj Close`.
- Signals use one-bar-lagged features in daily mode; trades are modeled on the next available daily close-to-close interval.
- No intraday stops, limit-order fills, minute-level fills, or intraday execution claims are made.
- Pair selection, feature selection/scaling, and PPO training occur inside train windows only.
- Validation windows are available to the locked protocol; test windows are evaluated after training.
- PPO timesteps are intentionally tiny for this cheap daily benchmark and must not be interpreted as tuned performance.

## Data Snapshot
- Source: `yfinance_cache`
- Original source: `yfinance`
- Cache path: `outputs/daily_validation/cache/yfinance_daily_500plus_2014_2024.csv`
- Cache SHA-256: `90e46104c6f95fc81451f8947916cc60ee54221d6acc2f7fa1961c8650bffe16`
- Download timestamp UTC: `2026-06-21T21:19:27.066783+00:00`
- Symbols requested: 936
- Symbols kept sample: ['A', 'AA', 'AAL', 'AAON', 'AAPL', 'AAXJ', 'ABBV', 'ABT', 'ACGL', 'ACHC', 'ACM', 'ACN', 'ACWI', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AGG', 'AGNC', 'AIA', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB']
- Symbols kept count: 744
- Actual date range: {'start': '2014-01-02T00:00:00', 'end': '2023-12-29T00:00:00'}

## Results
| strategy | status | total_return | cagr | sharpe | max_drawdown | calmar | win_rate | profit_factor | total_trades | turnover | total_costs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ppo_base | pass | 3.77264e-06 | 4.95159e-06 | 1.69835 | 1.43362e-06 | 3.45389 | 0.547368 | 1.43164 | 190 | 65.5227 | 0.0255756 |
| cash_no_trade | pass | 0 | 0 | 0 | 0 |  | 0 |  | 0 | 0 | 0 |
| random_policy | pass | 0.0410562 | 0.0542287 | 1.44242 | 0.0146502 | 3.70157 | 0.468421 | 1.33378 | 190 | 2.00531e+06 | 529.635 |
| zscore_mean_reversion | pass | -0.155749 | -0.199256 | -2.00389 | 0.213164 | -0.934753 | 0.363158 | 0.698123 | 58 | 797648 | 470.445 |
| ppo_no_wavelet | pass | 3.77264e-06 | 4.95159e-06 | 1.69835 | 1.43362e-06 | 3.45389 | 0.547368 | 1.43164 | 190 | 65.5227 | 0.0255756 |
| ppo_no_cointegration_filter | pass | -1.06258e-05 | -1.39463e-05 | -0.806768 | 1.45086e-05 | -0.961241 | 0.505263 | 0.834027 | 190 | 53.1512 | 0.0407669 |
| strict_inverse_corr_0_8 | pass | -3.72132e-06 | -4.88423e-06 | -1.04616 | 3.90216e-06 | -1.25168 | 0.473684 | 0.801152 | 190 | 109.173 | 0.043506 |

## Variant Status
- `ppo_base`: pass 
- `ppo_no_wavelet`: pass 
- `ppo_no_cointegration_filter`: pass 
- `strict_inverse_corr_0_8`: pass 

## Artifacts
- `outputs/daily_validation/data_manifest.json`
- `outputs/daily_validation/config_snapshot.json`
- `outputs/daily_validation/data_quality_report.md`
- `outputs/daily_validation/summary.csv`
- `outputs/daily_validation/summary.json`
- `outputs/daily_validation/equity_curves.csv`
- `outputs/daily_validation/trades.csv`
- `outputs/daily_validation/pair_results.csv`

## Limitations
- This run is a cheap daily-bar validation benchmark, not an expensive final evidence run.
- `trades.csv` is a bar-level turnover/cost event log, not a closed-position trade ledger.
- Average holding period is a turnover-event proxy derived from active-position days divided by turnover events.
- Adjusted close execution is a total-return modeling convention and is not an intraday fill model.

## Verdict
daily benchmark passed mechanically; PPO beat z-score on this cheap run, but this is not final evidence of edge
