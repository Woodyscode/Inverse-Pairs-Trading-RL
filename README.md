# Can strict inverse pairs pay for execution?

This research tests whether a departure from a historically negative return relationship predicts the next tradable move. The second question is whether a PPO agent can manage that position better than a fixed rule. **Neither walk-forward deployment produced a forecast that cleared both the predictive and after-cost validation tests. The selected portfolio held cash.** [Execution research brief](docs/execution_research_review.md) · [Full research report](outputs/runs/strict_inverse_alpha_final/report.md)

## Results

| Experiment | Held-out finding |
|---|---|
| Broad daily stock and ETF universe | Across 13 walk-forward windows and 471 stable pair-windows, no supervised model achieved a multiplicity-adjusted improvement in forecast MSE over zero. |
| Matched daily versus opening-hour forecasts | Of 36 ETFs screened across equity, fixed income, currency and commodities, 28 **pairs** qualified at each test deployment. Adding minute features produced no statistically reliable improvement in forecast error or net returns. |
| Strongest retrospective result | The one-session linear rule using opening-hour data earned **$1,886 on $100,000** across 36 trades in Jan–Oct 2026. The 95% block-bootstrap interval for return was **−2.08% to +6.93%**. Two gains in SCO/USO exceeded the book's entire profit. |
| PPO versus a fixed forecast rule | In the archived matched daily test, plain PPO and fixed Ridge stayed flat. PPO with Ridge lost **1.37 bp**. In synthetic markets with known reversion, plain PPO beat cash and random across three optimizer seeds and five independent continuations. |

The 2026 one-session linear book paid $226.57 in execution costs on $1.13 million of absolute filled notional. Blocking short entries turned its +1.89% result into **−0.87%**. No historical forecast-conditioned PPO was trained on the new minute sample because the validation gate rejected every forecast. [2026 portfolio metrics](outputs/runs/strict_inverse_alpha_final/2026/test/metrics.csv) · [Borrow sensitivity](outputs/runs/strict_inverse_alpha_final/2026/test/no_locates.csv) · [PPO comparison](outputs/runs/strict_inverse_alpha_final/evidence/historical_ppo_paired.csv)

## Research design

1. **Qualify the pair.** Require synchronized training-period return correlation at or below **−0.80**, stability across three training subwindows, a negative signed hedge and at least **$10 million** in trailing median daily dollar volume per leg. Reconfirm correlation and liquidity before deployment. Currency and commodity funds remain in the screen when they fail a threshold, with a recorded rejection reason. [Pair screen](outputs/runs/strict_inverse_alpha_final/2026/qualification.csv) · [Deployment decisions](outputs/runs/strict_inverse_alpha_final/2026/test/deployment.csv)
2. **Forecast the deviation.** Fit the return relationship on historical observations, then test whether the residual reverses or continues. Compare zero, linear, Ridge, XGBoost and fixed trading rules over one hour and 1, 5 and 10 sessions. Daily-only and opening-hour models share signal dates, prices, eligible pairs and execution rules. Overlapping forecast labels must mature before the next evaluation period. [Forecast and replay implementation](strict_alpha_study.py)
3. **Trade both legs.** If the slope is negative, a long residual buys both securities. A short residual needs borrow on both. Orders use a later synchronized minute bar. Each leg is capped at 1% of trailing ADV and realized fill-minute dollar volume, with partial exits until flat. The account pays 2 bp per filled dollar, 2% annualized financing on gross inventory and an additional 0.5% on short inventory. [Per-leg ledger](outputs/runs/strict_inverse_alpha_final/2026/test/executions.parquet) · [Daily equity](outputs/runs/strict_inverse_alpha_final/2026/test/daily_equity.parquet)
4. **Make the deployment decision on validation.** Require improved MSE versus zero, positive net return versus cash, Holm-adjusted significance on both endpoints and at least eight completed validation trades. The two test periods were calendar 2025 and Jan 2–Oct 23, 2026. Earlier research had already inspected aggregate 2024–2025 outcomes. The 2026 period extended beyond that work. [2025 selection](outputs/runs/strict_inverse_alpha_final/2025/validated_alpha.csv) · [2026 selection](outputs/runs/strict_inverse_alpha_final/2026/validated_alpha.csv)

## Verify the published results

The committed ledgers support an offline review. From the repository root in a Python environment, install the four packages used by the reviewer command. The published research environment used CPython 3.14.2.

```shell
python -m pip install numpy==2.4.2 pandas==3.0.1 pyarrow==23.0.1 matplotlib==3.10.8
python -m backtest.execution_review
```

The command checks that signed units close on each leg, fees equal 2 bp of absolute filled notional, and trade PnL ties to daily equity and published account returns in both validation periods and both test periods. Expected output includes four reconciliations, a break-even execution charge of **18.65 bp per filled dollar** for the highlighted book, and **−$963.01** after excluding its two largest gains. To regenerate the cost table, attribution and figure, use a new output directory.

```shell
python -m backtest.execution_review --output outputs/test_tmp/execution_exhibits
```

## Reproduce the research

Use a fresh environment and a new output directory for each replay. The research requirements record the direct dependency versions used for the study. The fetch step requires network access for missing market data. The minute archive is revision-pinned. Daily corporate-action snapshots can change at the vendor, so compare their SHA-256 values to the published manifest before treating a replay as an exact replication. [Research dependencies](requirements-research.txt) · [Input checksums and model settings](outputs/runs/strict_inverse_alpha_final/manifest.json)

```shell
python -m pip install -r requirements-research.txt
python -m pytest tests -q -p no:cacheprovider
python strict_alpha_study.py --fetch
python strict_alpha_study.py --output outputs/test_tmp/strict_alpha_replay
python -m backtest.execution_review --run outputs/test_tmp/strict_alpha_replay
```

The reviewer command reconciles a fresh replay directly. Regenerating its optional figure requires the paired comparisons produced by `strict_alpha_evidence.py`. Commands for the broad historical daily panel, synthetic seed replay, reward ablation and paired statistical analysis are in the [reproduction section of the research report](outputs/runs/strict_inverse_alpha_final/report.md#scope-and-reproducibility).

## Scope and evidence

Execution uses synchronized bar prints and modeled costs. The data contain no historical NBBO, queue position or security-specific locates. The study therefore measures forecast quality and simulated portfolio economics. Measuring implementation shortfall or scalable capacity would require quote and order data. The repository's live-order entry point is disabled.

| For this question | Start here |
|---|---|
| What failed in the original backtest, and how was it corrected? | [Codebase audit](outputs/audit/inverse_pairs_codebase_audit.md) · [Repair report](outputs/audit/inverse_pairs_repair_report.md) |
| Which assets and pairs passed or failed? | [Broad discovery report](outputs/runs/inverse_discovery_universe/report.md) · [Candidate-level rejection ledger](outputs/runs/inverse_discovery_universe/candidate_screen.csv.gz) |
| Did finer data or longer holding periods help? | [Paired granularity tests](outputs/runs/strict_inverse_alpha_final/evidence/paired_granularity.csv) · [2026 forecast and execution results](outputs/runs/strict_inverse_alpha_final/2026/test/) |
| Can PPO learn when a signal exists? | [Independent synthetic continuations](outputs/runs/strict_inverse_alpha_final/evidence/synthetic_continuations.csv) · [Reward ablation](outputs/runs/strict_control_rewards/comparison.csv) |
| Where are the earlier experiments? | [Research history and commands](RESEARCH_HISTORY.md) · [Separate SPY-hedged relative-value study](experiments/relative_value/README.md) |
