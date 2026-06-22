# Resume Bullets

- Built and audited a Python research framework for ETF and multi-asset pairs trading, including data snapshotting, walk-forward validation, transaction-cost modeling, statistical pair selection, baseline strategies, and PPO reinforcement-learning experiments.

- Repaired major backtest-validity issues in a quantitative trading codebase, including same-bar lookahead, PnL-based transaction costs, backfilled market data, inconsistent price fields, and non-causal feature generation.

- Designed reproducible daily-bar validation using a fixed yfinance cache of 936 requested symbols, 744 retained full-period assets, and 2014-2023 adjusted close data with checksum-backed data manifests.

- Implemented train-only pair selection and feature scaling, rolling train/validation/test evaluation, synthetic-data regression tests, smoke-mode execution, and baseline comparisons against cash, random policy, and z-score mean reversion.

- Conducted empirical research across inverse-correlation pairs, positive-correlation pairs, cointegration filters, wavelet ablations, and economically constrained ETF families; concluded no tested daily formulation produced durable after-cost edge.

- Demonstrated disciplined model governance by pausing PPO optimization after simple baselines failed, documenting negative alpha findings, and producing research reports focused on leakage control, reproducibility, and out-of-sample validity.

- Added diagnostics for out-of-sample ADF p-values, half-life, correlation decay, cointegration persistence, turnover, costs, exposure, and pair-level contribution to distinguish genuine signal from backtest artifacts.

- Produced final research documentation, validation artifacts, and interview-ready project summaries emphasizing rigorous quant research process rather than unsupported profitability claims.
