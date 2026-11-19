# Inverse-pairs research history and reproduction notes

The project began with a daily inverse-pairs PPO strategy and later added minute-level event studies. The original audit and repair report identify which timing, data and accounting changes separate the early backtests from the results used today. [Original audit](outputs/audit/inverse_pairs_codebase_audit.md) · [Repair report](outputs/audit/inverse_pairs_repair_report.md)

## Gap reopening and the incremental short leg

The gap-reopening study ranked strictly inverse ETFs by their trailing 20-session performance. It bought the falling leader and shorted the rising laggard when a large gap contracted. The first study compared that rule with a leader-only trade sized by trailing volatility and with a causal wavelet filter. The follow-up sized the leader to the pair's estimated factor exposure, reconciled each leg's PnL and reranked candidates after excluding SCO/USO. The leader-only benchmark earned more in both retrospective periods. The forward protocol specifies the next comparison and its scoring date. [Original event study](outputs/runs/inverse_gap_reopening/report.md) · [Matched short-leg study](outputs/runs/inverse_short_leg_review/report.md) · [Forward protocol](config/inverse_short_leg_forward.json)

## Strict inverse-pairs alpha and portfolio control

The matched daily and opening-hour experiment retained the −0.80 correlation threshold and evaluated four holding horizons. Neither validation period qualified a forecast for historical PPO integration. The full report includes the 2026 holdout, the 13-window historical stock and ETF forecast study, the matched historical PPO comparison and the multi-seed synthetic replay. [Research report and reproduction commands](outputs/runs/strict_inverse_alpha_final/report.md)

## Strict cross-asset inverse-pair discovery

The primary research question is whether a deviation from a strong inverse relationship predicts a tradable basket return. Discovery requires daily training return correlation at or below −0.80. The ETF cohort includes currencies, commodities, fixed income, equity indices, sectors and inverse products. Each fold also draws on historical S&P 500 constituents. The [ETF cohort](config/inverse_discovery.yaml) records the benchmark families, intended direction and historical sources.

Run discovery from the repository root with the frozen daily market caches and a new output directory. The first command obtains any missing supplemental ETF snapshots. The second tests the fixed threshold rule for each candidate at one, five and ten sessions. Training estimates the signed hedge and screens relationships. Chronological validation qualifies and ranks rules. The following held-out period measures the result.

```bash
python -m data.fetch_inverse_etfs
python inverse_pair_discovery.py --output outputs/runs/my_inverse_discovery
```

The [discovery report](outputs/runs/inverse_discovery_universe/report.md) and [asset-class breakdown](outputs/runs/inverse_discovery_universe/asset_class_funnel.csv) cover inverse ETFs on a common benchmark, ETF pairs spanning asset classes, different factors within a macro asset class, and stocks or sectors paired with hedge instruments. The [SPY-hedged relative-value study](experiments/relative_value/README.md) has its own runner, results and research question.

### Does an inverse ETF's tracking deviation reverse?

The [tracking study](inverse_tracking_study.py) uses the strict screen to select a pair, then uses its return tracking deviation to decide whether to enter. If IWM rises 1.00% and the fitted relationship predicts RWM will fall 1.00%, an actual RWM return of −0.70% leaves a +0.30 percentage point deviation. A positive extreme prompts a short position in the inverse-tracking basket. With a negative hedge ratio, this means shorting both legs, in proportion to their fitted exposures. The reference ETF approximates the index in the fund mandate. This signal measures a deviation between market returns. Establishing a premium to NAV would require separate NAV data. [RWM daily investment objective](https://www.proshares.com/our-etfs/leveraged-and-inverse/rwm)

The study starts with 16 designed inverse-ETF pairs from the last discovery fold. Training ran from January 7, 2020 to January 5, 2023. Each eligible pair needed correlation at or below −0.80 overall, in each training third, and again in the 2023 validation period before the 2024–2025 test. Each leg needed $10 million median training dollar volume over the final 60 sessions. A training-only unit check compares the two feeds' opening prices under as-traded and split-adjusted conventions and requires the 99th percentile discrepancy to be at most 1%. SQQQ uses split-adjusted minute prices. HYG failed reconciliation, excluding SJB/HYG. All 15 remaining pairs qualified. [Pair calibrations](outputs/runs/inverse_etf_tracking/calibration.csv) · [Feed checks](outputs/runs/inverse_etf_tracking/data_quality.csv)

The prior-close signal measures the daily return residual against a fixed training intercept and slope. The opening-hour signal measures the 09:30–10:30 New York move with the hedge sensitivity adjusted for the overnight drift since the last reset. Each signal uses its own training median and 95th percentile absolute deviation. The primary account takes the largest normalized extreme then available. Entry orders start at 09:31 or 10:31, respectively. The four exit variants receive identical entries and hold fixed quantities. All wait until the longest holding period clears before entering again. The closing window starts ten minutes before the scheduled close. The account targets $25,000 gross per event with $100,000 starting capital. Both legs use synchronized one-minute trade bars, capped at 1% of observed bar and trailing daily dollar volume. Exit orders can liquidate over several bars.

The ledger includes effective splits, cash distributions, two basis points per fill on traded notional, 2% annual gross financing and 0.5% annual short borrow. The reset-adjusted diagnostic compounds predicted daily returns and accounts for partial entry and exit sessions. Portfolio P&L uses observed fund prices. Minute trade bars provide execution proxies. Historical quotes, queue position and borrow locates are unavailable.

| Timing of extreme | Filled entries in 2024–2025 | One hour | Closing window | One session | Five sessions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prior close | 56 | −4.45 bp | −5.50 bp | −5.96 bp | −21.90 bp |
| Opening hour | 48 | −4.20 bp | −4.04 bp | −5.76 bp | −8.70 bp |

Values are total net P&L divided by total gross entry notional across events. All eight primary variants lost money. Only the opening-hour signal held to the closing window had positive gross P&L, about 0.11 bp of gross entry notional. Standalone pair results are exploratory and share dates and reference legs. [Results and 20-session block-bootstrap intervals](outputs/runs/inverse_etf_tracking/results.csv) · [Matched trades](outputs/runs/inverse_etf_tracking/trades.csv) · [Coverage](outputs/runs/inverse_etf_tracking/coverage.csv)

```bash
python -m data.fetch_tracking_minutes
python inverse_tracking_study.py --output outputs/runs/my_inverse_tracking
```

## Post-repair historical check

The November 18, 2026 rerun reprocessed the fixed 2014-2023 Yahoo Finance snapshot for the 340-symbol study universe. The 13 test folds cover July 6, 2017 through December 29, 2023. Four folds qualified for trading and nine held cash. The PPO training budget was 256 steps per eligible fold, with seed 42. All strategies started at $100,000.

| Strategy | Ending equity | Total return | Maximum drawdown |
| --- | ---: | ---: | ---: |
| Cash | $100,000.00 | 0.00% | 0.00% |
| PPO | $99,994.96 | -0.0050% | 0.0086% |
| Seeded random | $98,231.16 | -1.77% | 3.57% |
| Z-score mean reversion | $83,325.70 | -16.67% | 22.77% |

PPO averaged 0.0098% gross exposure over the test intervals. Its result is consistent with holding almost all capital in cash. The separate benchmark run reproduced the z-score equity curve to within $0.000001 on every date. Daily net PnL reconciled to gross PnL less modeled costs for all four strategies. [PPO results](outputs/runs/audit_ppo/metrics.json) / [Benchmark comparison](outputs/runs/audit_ppo/baseline_metrics.json) / [Fold selections](outputs/runs/audit_ppo/selected_pairs.json) / [Run manifest](outputs/runs/audit_ppo/run_manifest.json)

The companion broad-pair study recorded an unavailable CROC price on May 26, 2022. It leaves the affected variant unscored. [Companion study outcomes](outputs/runs/audit_research_pivot/summary.csv)

## Reproducing the earlier PPO experiment

The commands below reproduce the earlier benchmark and PPO workflow. The cross-asset discovery run above uses the threshold baseline alone.

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
python main_backtest.py --config config/config.yaml
python main_train.py --config config/config.yaml
```

The default PPO budget is two million steps per eligible fold. The saved [historical validation configuration](outputs/runs/audit_ppo/input.yaml) uses 256 steps per eligible fold and the existing local Yahoo Finance snapshot. Run it with `python main_train.py --config outputs/runs/audit_ppo/input.yaml`. Each run writes its resolved configuration, data checksum, software versions, source fingerprint, selected pairs, hedge ratios, feature-selection parameters, equity curves and bar-level costs. PPO checkpoints are stored by fold. Choose a separate output and model directory to retain independent runs.

## Evaluation protocol

- Each fold uses three years for fitting, six months for chronological validation and up to six months for testing. The final test fold can be shorter if it meets the configured minimum length. The fixed policy is scored on validation and then test. Cash, random actions and z-score mean reversion use the same execution model.
- Eligibility uses observed history available by the training cutoff. The research panel preserves the market calendar and missing quotes. A selected leg with an unavailable evaluation price stops the run. A fold with no eligible pair holds cash.
- The default study screens for return correlation at or below −0.8. It applies the Holm adjustment to the cointegration tests performed after the correlation and volatility screens, then ranks qualifying pairs by absolute training correlation. The residual is adjusted price A minus the fitted intercept and slope times adjusted price B. A negative slope gives the two legs the same sign when trading the residual.
- Each pair's target is a fraction of equity allocated across the absolute notionals of both legs. Observations through close t−1 determine the target at close t. The position earns the return from t to t+1. Trailing feature history crosses fold boundaries. Feature scaling, selection and hedge estimation use the training fold.
- Commission and slippage apply to traded notional on both legs, including rebalancing and fold-end liquidation. Borrow applies to short notional. Financing applies to gross deployed capital at the configured funding rate. Both accrue over calendar days. ADV caps use prior raw close times volume, with one shared participation budget per symbol.
- The simulation assumes closing-price fills, fractional positions, fixed cost and borrow rates, and availability of shorts except for configured `env.unborrowable_symbols`. Fold-end settlement charges exit costs and assumes the remaining inventory can be closed. Adjusted closes approximate total returns. The supplied static universe and vendor snapshot lack historical membership, historical locates and executable quotes. These assumptions limit the performance conclusions.

The live runner is disabled. Research checkpoints depend on fold-specific pair order, price adjustments, fitted features and the simulator's order conventions.

## Residual-spread portfolio control

The [residual-control study](residual_control_study.py) compares cash, seeded random actions, a z-score rule, Ridge, XGBoost, PPO, and PPO supplied with the Ridge forecast. All seven share the same pair, dates, execution code and trading costs in each comparison. Each fold selects at most one pair for the three-action experiment. A long signal targets dollar weights proportional to `(1, -beta)`. A negative log-price hedge ratio therefore buys both legs. A short signal reverses both weights, and flat requests liquidation.

The market state includes the 30-session residual z-score and its change, residual volatility, a trailing estimate of reversion speed, a trailing ADF diagnostic, hedge estimates and relative volume. The inventory state includes signed exposure, unrealized PnL using average acquisition prices, and holding time. The hedge used for sizing expands during training and freezes at the training cutoff. The rolling hedge feature and stationarity diagnostics continue to update using observed prices. At signal close s, supervised models forecast the attainable simple basket return from close s+1 to s+2. The Ridge forecasts used to train PPO come from expanding blocks fitted only to labels that had matured by the prediction date.

The reward is 100 times the sum of net portfolio return, a negative variance penalty and a negative turnover regularizer. The risk coefficient is 1.0 and the turnover coefficient is 0.00005. The ledger charges actual simulated turnover and carry once. The extra penalties affect learning alone. PPO trains on randomly located historical episodes with a fixed 25% gross target, subject to capacity. Validation selects the checkpoint with the highest net return and retains the earliest tie. The saved diagnostics include sampled-policy entropy, deterministic action frequencies, reward components, optimizer statistics, fill rates and per-fold forecast quality.

To reproduce the 50,000-step historical comparison with the existing raw snapshot, use a fresh output directory.

```bash
python residual_control_study.py --steps 50000 --capacity-mode partial-fill --strict-execution --output outputs/runs/my_residual_study
```

The default full-target screen requires a complete target to fit in one session at the participation limit. Partial-fill mode retains the same inverse-correlation, Holm-adjusted log-cointegration, hedge sign, half-life, subwindow stability and cost screens. It allows smaller fills for lower-capacity pairs. Strict execution sizes target inventory to one session of trailing exit capacity, limits each order using both prior average dollar volume and observed execution-day dollar volume, and reserves the final 20 sessions of each episode for liquidation. A fold fails if final inventory cannot settle within the limit. The `--synthetic-only` option runs the controlled study independently. The `--verify-holdouts 5` option replays saved synthetic policies from the specified output directory on five new continuations, using the original settings.

### What the control study established

In the synthetic market, the log residual has a known autoregressive coefficient of 0.72. Success required each deterministic PPO policy to beat cash and seeded random actions after costs and choose the reversion direction on at least 60% of signals exceeding one stationary standard deviation. The 256-step run failed. With a 50,000-step budget, both policies matched the expected direction on 97.7% of those signals. The 250,000- and 1,000,000-step runs selected the same 20,000-step checkpoints on validation. These are repeated observations of one training seed. On five independently simulated continuations, the frozen policies beat cash and random on every path. [Training budget comparison](outputs/runs/residual_control_50000/synthetic/budget_curve.csv) / [Independent holdouts](outputs/runs/residual_control_50000/synthetic/independent_holdouts.csv)

The earlier continuous-action policy paid a reward penalty of 25 basis points per day at 25% gross exposure. Multiplying its two near-zero action components could also produce a tiny allocation. Across the 500 invested-window intervals, the recorded exposure penalty summed to 16.06 basis points of reward, versus a net portfolio loss of 0.50 basis points. This identifies a substantial incentive to stay near cash. Establishing the contribution of each mechanism would require a controlled ablation. [Original execution ledger](outputs/runs/audit_ppo/strategy_frictions.csv)

Across 13 historical folds, 1,265 pair-window candidates met the negative return-correlation screen. Only ULE/UUP survived Holm-adjusted log-cointegration with a negative hedge slope, in two adjacent folds. Its estimated half-lives were 3.6 and 3.7 sessions. Median-volume entry capacity for the whole account at the 25% gross target was approximately $10,616 and $8,746, respectively. The $100,000 full-target screen therefore held cash throughout. [Candidate diagnostics](outputs/runs/residual_control_50000/screen.csv)

Allowing smaller fills produced 125 scored forecasts in each of the two eligible test folds. Ridge rank correlations were 0.050 and 0.008. XGBoost rank correlations were 0.029 and -0.049. Ridge stayed below the four-basis-point round-trip execution hurdle throughout both windows. [Held-out forecast diagnostics](outputs/runs/residual_control_liquidity_50k/forecast_quality.csv)

The initial partial-fill comparison settled some final positions at up to 2.05 times the daily participation limit. With strict execution throughout training and evaluation, plain PPO, standalone Ridge and cash each returned 0%. PPO with the Ridge forecast requested a non-flat position on half of the 250 eligible decisions and lost 1.37 basis points net across the full walk-forward period. XGBoost lost 3.51 basis points, the z-score rule lost 6.45 and seeded random actions lost 8.89. All historical policies used the same execution constraints and each PPO had 50,000 training steps per eligible fold. One pair in two adjacent folds provides little evidence about performance in other markets. [Historical strategy comparison](outputs/runs/residual_control_liquidity_50k/test_summary.csv) / [Policy diagnostics](outputs/runs/residual_control_liquidity_50k/policy_diagnostics.csv)

The synthetic PPO stability check held the market, training budget (50,000 steps), random baseline and strict execution fixed, and changed the optimizer and episode-sampling seed. Both PPO variants beat cash and random for seeds 7, 42 and 83. Across the six evaluations, agreement with the known reversion direction ranged from 97.7% to 98.2%. The three seeds share one synthetic market path. [Seed 7](outputs/runs/rv_seed_7/synthetic/test_summary.csv) · [Seed 42](outputs/runs/residual_control_liquidity_50k/synthetic/test_summary.csv) · [Seed 83](outputs/runs/rv_seed_83/synthetic/test_summary.csv)

## Daily and one-minute relative-value research

The [separate relative-value experiment](experiments/relative_value/README.md) contains the SPY-hedged and unrestricted stock-pair comparisons.
