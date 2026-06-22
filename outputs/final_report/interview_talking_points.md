# Interview Talking Points

## One-Minute Summary

I built a pairs-trading research framework around an inverse-pairs hypothesis: find negatively correlated assets, filter with Engle-Granger cointegration, generate causal wavelet and spread features, and test whether PPO could trade the spread. The original implementation had backtest-validity issues, so I audited and repaired lookahead, cost modeling, data backfill, feature leakage, and reproducibility. After hardening the framework and running broad daily validation plus several pivots, the empirical result was negative: no tested daily pairs formulation produced durable after-cost edge, so I stopped PPO tuning.

## What The Hypothesis Was

- The original idea was that inverse ETF and asset pairs might form mean-reverting spreads.
- Candidate pairs were selected using negative correlation and Engle-Granger cointegration.
- The strategy used `db4` wavelet features and a PPO agent to trade the spread.
- Expected holding period was short, around several days.

## What I Had To Fix First

- Same-bar lookahead in the environment.
- Transaction costs incorrectly tied to PnL instead of traded notional.
- Backfilled market data that could leak future information.
- Full-series wavelet processing that needed causal treatment.
- Test-window feature initialization that could produce all-zero features.
- Pair selection and feature scaling that needed to be train-window only.
- Reproducibility issues from live downloads and expensive PPO runs.

## How I Validated It

- Created fixed local daily data snapshots with manifests and checksums.
- Used 2014-2023 adjusted daily data from a broad 936-symbol requested universe, with 744 retained full-period symbols.
- Locked a rolling train/validation/test protocol.
- Selected pairs only in train windows.
- Fit scalers and feature selectors only on train data.
- Evaluated against cash, random policy, and z-score mean reversion.
- Added synthetic tests for cointegrated pairs, non-cointegrated pairs, independent random assets, and manual reward paths.
- Added spread diagnostics: ADF p-value, half-life, correlation decay, cointegration persistence, turnover, costs, and pair-level contribution.

## What Failed Empirically

- PPO collapsed to near-zero exposure, so its nominal Sharpe was not economically meaningful.
- The active z-score baseline lost about 15.6% in the broad inverse-pairs daily validation.
- Selected inverse pairs often involved volatility ETPs and failed out-of-sample stationarity and cointegration checks.
- Positive-correlation pair mining did not materially improve results.
- Economically constrained ETF families also failed to show durable after-cost edge.
- Isolated volatility/inverse/leveraged ETPs performed badly and should not be mixed with ordinary ETF pairs.

## Why Random Policy Looked Better Than PPO

Random policy took real exposure while PPO stayed nearly flat. The random result was a lucky or structurally favorable sample of spread movement, not evidence of skill. It was a diagnostic warning that PPO was not learning an economically useful policy.

## Why I Did Not Tune PPO

I intentionally paused PPO because no simple transparent baseline showed robust after-cost edge. If z-score mean reversion cannot make money on the selected spreads, tuning PPO risks fitting noise and turning a research problem into an overfitting exercise.

## What This Demonstrates

- I can audit a quant backtest for leakage and realism.
- I can turn an invalid prototype into a reproducible validation framework.
- I am comfortable reporting negative results honestly.
- I know the difference between a profitable strategy and a useful research process.
- I use baselines and diagnostics before trusting complex models.

## Best Technical Details To Mention

- Fixed action timing so observations at `t` only earn PnL over `t -> t+1`.
- Replaced PnL-based costs with turnover-based costs applied on entries, exits, and position changes.
- Removed backfill and used data-quality manifests to track missingness and retained symbols.
- Made `Adj Close` usage consistent across pair selection, features, PnL, and reports.
- Added causal `db4` wavelet processing to avoid full-series future leakage.
- Added train-only pair selection, train-only scaling, and walk-forward validation.
- Added synthetic regression tests to prove the validation logic catches known failure modes.

## Honest Final Framing

This is not a profitable trading strategy. It is a rigorous quant research framework that rejected the tested hypothesis. The value of the project is in the validation discipline: identifying leakage, enforcing causality, adding baselines, running controlled pivots, and stopping before tuning a complex model on a weak signal.

## Future Work I Would Propose

- Use a delisting-aware institutional dataset.
- Expand walk-forward windows beyond the two-window daily benchmark.
- Implement hedge-ratio-weighted residual execution.
- Add stricter validation gates for stationarity, half-life, and cointegration persistence.
- Test economically motivated baskets rather than single pairs.
- Only move to minute data or PPO if a simple daily or intraday baseline shows robust after-cost edge.
