# Strict inverse-pairs alpha and portfolio control

## Decision

Keep the cash and fixed-rule benchmarks. The study found inverse relationships and some predictive information in individual periods. No forecast met both the predictive and after-cost validation requirements for historical PPO integration. The decision at both deployment dates was cash, earning 0% under the study's zero cash-rate assumption. Forecasts, counterfactual portfolios and their losing as well as winning trades are saved for both test periods.

Minute granularity changed the trades. It did not establish a repeatable improvement. The largest positive result on the new January–October 2026 test was a one-session linear rule with opening-hour information. It earned $1,886 on $100,000 across 36 trades. Its forecast rank correlation was −0.011, its MSE exceeded the zero-forecast MSE, and its 95% ten-session block-bootstrap interval for account return was −2.08% to +6.93%. The rule lost 0.87% when both legs required confirmed borrow and no locates were available. Choosing it after observing this return would use the test period for model selection.

## What the agent would be trading

An inverse pair has synchronized training-period return correlation at or below −0.80. A tracking residual measures how far today's observed pair return departs from its fitted inverse relationship. If B rises 1%, a slope of −1 predicts that A falls about 1%. If A falls only 0.6%, A finishes about 0.4 percentage points above that prediction, before allowing for the fitted intercept. Reversal bets that this departure subsequently unwinds. Continuation bets that it persists. Both hypotheses use the same strictly qualifying pair. A negative slope gives a long residual positive weights in both legs, proportional to (1, −β). Such a position can retain market exposure.

## Audit and opportunity universe

The [original project audit](../../audit/inverse_pairs_codebase_audit.md) identified backward-looking reward, zero test features, PnL-based fees and future-price backfill. The [repair report](../../audit/inverse_pairs_repair_report.md) describes the corrections. The current experiments use trailing observations, a later execution bar, price changes after execution, split-consistent prices, cash distributions and per-leg traded-notional fees. All amounts in this report are simulated US dollars.

The [archived broad discovery](../inverse_discovery_universe/report.md) screened 422,663 pair-windows using a fixed pre-2014 ETF list and reconstructed historical S&P membership. It included equity, fixed-income, currency and commodity exposures. Of 1,912 strict and liquid pair-windows, 27 joined different ETF asset classes. All 27 failed at least one stability subwindow. The three previously selected rules completed 43 held-out trades and lost money after costs. The original candidate-level rejection ledger remains in that run. The broader SPY-hedged strategy was excluded. SPY itself could qualify as one leg of a strictly inverse pair.

**New daily coverage.** We evaluated zero, linear, Ridge, XGBoost, residual reversal and continuation forecasts over 1, 5 and 10 sessions across the original 13 walk-forward windows. The supplemental study used 471 stable pair-windows after early hedge calibration, comprising 180 inverse ETF, 176 stock or sector hedge, and 115 within-macro-class observations. It used the first two-thirds of each training window for the signed return hedge, then trained the alpha models on the remaining matured labels. Active membership was checked on the signal date, and test deployment required validation correlation at or below −0.80. Forecast labels measure the fixed-unit basket from the next close. Historical stock prices are reconstructed adjusted closes. This supplement measures forecast quality. The archived broad discovery supplies the separate daily execution evidence.

Across the three daily categories, no model had a Holm-significant held-out MSE improvement over zero. For one-session Ridge, held-out rank correlations were −0.018 for inverse ETFs, +0.013 for stock or sector hedges and −0.002 for within-class relationships. [Broad daily forecasts, pairs and metrics](../strict_daily_alpha_coverage/)

**New minute coverage.** The matched experiment used 36 pre-existing ETFs with cached daily and minute histories. The fixed subset included the prior tracking-study instruments plus currency, commodity and fixed-income funds. We screened all 630 combinations per fold. Inclusion required complete calibration returns, negative slope, overall and three subwindow correlations at or below −0.80, price-unit parity and at least $10 million median daily dollar volume per leg over the last 60 sessions. The pooled supervised-training period also had to satisfy the correlation threshold. Deployment reconfirmed trailing-year correlation and liquidity. HYG and YCS failed minute-unit quality. EUO/FXE, for example, had −0.987 calibration correlation in the later fold but just $4.63 million in lower-leg median volume. Its exclusion preserves the cross-asset hypothesis and the liquidity floor.

| Forecast-model fit | Validation | Test | Validation pairs | Test pairs | Forecasts passing both validation endpoints |
|---|---|---|---:|---:|---:|
| 2022–2023, hedge frozen end of 2021 | 2024 | 2025 | 30 | 28 | 0 |
| 2023–2024, hedge frozen end of 2022 | 2025 | Jan 2–Oct 23, 2026 | 29 | 28 | 0 |

The earlier tracking experiment had inspected aggregate 2024–2025 results. The 2026 slice extended beyond that study. One pilot replay of the new protocol preceded a verification run that also enforces the full supervised-training correlation check, limits entry size against current equity after losses and records session-weighted exposures. Neither validation gate changed. The tabulated results use the verified accounting. These are retrospective holdouts. Independent prospective execution remains untested.

## Forecasts, timing and execution

Daily inputs were the lagged return residual and 30-session z-score, five-session residual momentum, residual and reference volatility, signed slope, and the z-score–reference-volatility interaction. The minute arm added the opening-hour drift-adjusted residual, its trailing z-score, realized residual volatility, opening dollar volume relative to the trailing 30 sessions and the opening z-score–volatility interaction. At least 45 synchronized one-minute return increments were required for realized volatility. Both arms used identical complete signal rows and the same intraday hedge and execution. This measures the incremental information from opening-hour features at a fixed decision time. It does not compare two different execution schedules.

The last feature bar closed at 10:30 New York time. The earliest fill used the close of the 10:31 minute bar. We looked up to five bars for a synchronized positive-volume entry. Exits targeted an hour later, or the same intraday time 1, 5 or 10 sessions later. The supervised target was the subsequent gross fixed-unit basket return, including cash distributions, at the first synchronized target print within five bars. Missing target marks were excluded from forecast scoring. The portfolio submitted orders using only the available signal and allowed partial exits to continue until liquidation. An exit failure raises an error. The final horizon plus three sessions were reserved for existing positions. Training labels had to mature before validation, and scored validation labels had to mature before test. A single book could hold one pair at a time. Re-entry began on a later session after both legs were flat.

Linear and Ridge used training-only standardization. Ridge used α = 100. XGBoost used 80 depth-two trees, learning rate 0.04, L2 penalty 10 and seed 42. The two simple rules used opposite signs of the same z-score, with a common training-calibrated return scale. The pooled portfolios chose the pair with the largest absolute forecast above two per-leg execution charges on gross traded notional plus expected calendar-time carry. Risk was constrained by position size and participation. Random selected a pair and one of three directions with a fixed seed. A zero forecast stayed in cash.

Entry gross was capped at the smaller of $25,000 and 25% of current equity. Every leg was limited to 1% of both trailing ADV and realized fill-minute dollar volume. Each fill cost 2 bp on actual absolute leg notional. Financing accrued at 2% annually on gross inventory, with an additional 0.5% on shorts. The primary case assumes borrow is available at that fee. The no-locate replay rejected short orders. Close-to-close marks, split-adjusted units, ex-date distributions, partial liquidations and weekend carry reconcile to every trade's cash PnL. The saved session-weighted exposure distinguishes intraday holdings from a book flat at the closing bell.

## Predictive information and economic significance

Validation required positive improvement in squared forecast error versus zero, positive net returns versus cash, significance on both endpoints after Holm correction and at least eight completed trades. MSE differences were averaged across pairs per date. One-sided tests used ten-lag autocorrelation-consistent variance. The correction covered all portfolio models, signs, granularities, horizons and both endpoints within each deployment decision. The separate comparison of adding minute features also corrected across folds, models, horizons and both endpoints within validation and test. Pair and regime breakdowns are descriptive. No subgroup was promoted into a strategy.

Six specifications improved MSE significantly during 2024 validation, including daily ten-session XGBoost and both one-session Ridge variants. None passed the after-cost test. No specification passed the corrected MSE or economic test in the next validation or in either matched execution holdout. Paired minute-minus-daily tests found no corrected improvement in either forecast error or net return. Some point estimates of IC were positive, including 0.129 rank IC for daily ten-session XGBoost in 2026. That model's 2026 account return was +0.48%. These estimates varied by horizon, model and period.

**Net account returns with opening-hour features, January–October 2026**

| Prespecified rule | One hour | One session | Five sessions | Ten sessions |
|---|---:|---:|---:|---:|
| Linear | +0.23% | +1.89% | −3.44% | −3.34% |
| Ridge | +0.23% | +1.70% | −3.44% | −3.35% |
| XGBoost | +0.18% | −4.93% | +0.48% | −1.16% |
| Residual reversal | −0.44% | +0.60% | −0.66% | +0.61% |
| Residual continuation | −1.18% | −1.85% | +0.00% | −1.25% |
| Cash | 0% | 0% | 0% | 0% |

The one-session linear book earned $2,152 gross, paid $227 execution and $39 combined financing and borrow. Its median entry gross was $15,466, and session-weighted gross exposure averaged 2.73% of equity. Its two largest trades, both SCO/USO, earned $1,546 and $1,303. Those two gains together exceeded the entire portfolio's net profit. The largest positive account result in the 2025 test was only +0.20%, from ten-session daily linear regression. Available fill capacity and the short sample preclude a scalability claim.

## Does PPO improve control?

**Historical controlled comparison.** The earlier corrected 13-window daily experiment already compared cash, fixed Ridge, XGBoost, a z-score rule, random and PPO with and without the Ridge forecast under matched execution. The only selected pair was ULE/UUP, with training correlations of −0.878 and −0.869 in two adjacent windows. Plain PPO and the fixed Ridge rule both stayed flat. PPO with Ridge lost 1.37 bp on the full account, incurred $41.27 in costs, had 0.031% maximum drawdown and turned over 0.242 times account value. Fixed Ridge had zero costs, drawdown and turnover. The paired ten-session block interval for incremental additive return over fixed Ridge was −4.78 to +1.86 bp. Avoiding the losing z-score rule improved outcomes, and cash achieved the same result as plain PPO. [Historical daily ledger and checkpoints](../residual_control_liquidity_50k/) · [Paired comparisons](evidence/historical_ppo_paired.csv)

**Learnability.** We replayed frozen 50,000-step policies trained with three optimizer seeds on the same five independent synthetic continuations. Plain PPO beat both cash and seeded random in all 15 seed/path combinations. Mean cumulative returns by optimizer seed were 25.67%, 25.72% and 24.42%. Mean strong-signal directional agreement exceeded 99% for each seed. There are five independent future paths, with three policies evaluated on each. The saved 256-step, 50,000-step, 250,000-step and one-million-step budget experiments selected the same 20,000-step validation checkpoint for the three larger budgets in their common simulator configuration. [Seed/path results](evidence/synthetic_continuations.csv) · [Budget and training diagnostics](evidence/synthetic_budget_checkpoints.csv)

**New portfolio-control ablation.** We trained two 50,000-step controllers with seed 42 on the same synthetic market. Each saw the causal Ridge forecast, matured forecast-error uncertainty, spread state, inventory, unrealized PnL, position age, cost rates, borrow restrictions, relative capacity, remaining episode horizon and volatility regime. Actions were short, flat, long, maintain units and halve units. Cash earned no trade bonus. Both rewards used subsequent net returns. The penalized arm also charged a gross-exposure variance penalty and a separate turnover regularizer, neither entered in the financial ledger. Checkpoints were selected by validation net return. No reward was fitted to historical test results.

| Synthetic held-out policy | Net return | Maximum drawdown | Turnover × capital |
|---|---:|---:|---:|
| Fixed Ridge | 30.12% | 1.20% | 101.72 |
| PPO, net reward | 25.13% | 1.62% | 92.06 |
| PPO, risk and turnover penalties | 27.16% | 1.10% | 98.35 |

The penalized controller reduced synthetic drawdown and turnover against fixed Ridge on this path. Both controllers earned less than fixed Ridge. The two-arm reward experiment used one training seed. Multi-seed results above apply to the archived three-action policies. Full validation histories include entropy, action frequencies, approximate KL, value loss, exposures, reward components and selected checkpoints. [Reward ablation and checkpoints](../strict_control_rewards/)

**Conclusion at $100,000.** Neither Ridge nor XGBoost produced a repeatable after-cost opportunity for a new historical PPO controller. The archived matched experiment found no incremental benefit over the fixed forecast rule. The counterfactual performance of a newly trained forecast-conditioned PPO on the 2026 minute tape is unmeasured because no forecast qualified for integration. Retain the simpler fixed rule as the research control and cash as the outcome when its opportunity gate fails.

## Scope and reproducibility

These are bar-price fills with modeled costs and participation limits. Historical NBBO, queue position, stock-loan inventory and security-specific borrow fees were unavailable. The 36-fund minute cohort is narrower than the historical daily universe. Reconstructed total returns and corporate actions are vendor snapshots, rather than a vintage history of every vendor revision. Correlated pairs can share legs. Date aggregation and dependent resampling address some of that dependence, but the final holdout covers less than a year. Synthetic returns measure behavior in a known process and have no direct interpretation as expected market returns.

The dated folders contain the full screen, training cutoff, signed hedge, requalification, liquidity and ADV-based gross capacity. Each validation and test folder contains forecasts, actions, executions with signed units and fees, trade results, daily equity, regime and pair diagnostics, no-locate sensitivity and portfolio metrics. Metrics include MSE, mean forecast and outcome, calibration slope, Pearson and rank IC, directional accuracy, annualized and cumulative returns, Sharpe, Sortino, Calmar, drawdown, financing, borrow, exposure, holding time, turnover and capacity. The calibration intercept can be recovered as mean outcome minus slope times mean forecast. Some ratios are undefined for an all-cash book. [Research configuration and raw-input checksums](manifest.json) · [2026 test metrics](2026/test/metrics.csv) · [Pair and regime PnL](evidence/attribution.csv) · [Daily-versus-minute comparisons](evidence/paired_granularity.csv)

From the repository root, use the pinned cached inputs identified in the manifests. The following commands use fresh output names. The fetch step reuses existing files and requires network access for missing files. A later vendor download may differ from the snapshot, so compare input hashes. Model artifacts and reports from previous runs must remain available for archived analyses.

```powershell
python strict_alpha_study.py --fetch
python strict_alpha_study.py --output outputs/test_tmp/strict_alpha_replica
python strict_daily_coverage.py --output outputs/test_tmp/strict_daily_replica
python strict_control_validation.py --replay --output outputs/test_tmp/strict_seed_replica
python strict_control_validation.py --output outputs/test_tmp/strict_reward_replica
python strict_alpha_evidence.py --root outputs/test_tmp/strict_alpha_replica --runs outputs/runs
python -m pytest tests -q -p no:cacheprovider
```

The evidence command reads the specified primary replica and the named archived and supplemental runs under `--runs`. The exact four research scripts used to produce the published artifacts are also preserved under `source/`. The research programs refuse to overwrite populated experiment destinations. Prior corrected runs and the separate relative-value experiment remain available under their original names.

The full correctness suite passed all 44 tests. A second run from the same market snapshots reproduced all 101 primary result files and model artifacts byte for byte. The separately checked manifest records the source version and input checksums.
