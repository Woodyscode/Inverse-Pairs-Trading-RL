# Does shorting the rising laggard add value?

**The short leg made money in 2026, but buying more of the prior leader produced a better result.** In the frozen $100,000 strategy, the short contributed $3,306 net and the long contributed $3,778. A leader-only book sized to approximate the two-leg book's factor exposure earned $7,443 on the same entries. Both books lost money in 2025. Retain the leader pullback as the research benchmark. The incremental results do not justify allocating capital or training PPO on this signal. [Book results](books.csv) · [Leg attribution](legs.csv)

## Matched portfolios

Both instruments had to pass the archived −0.80 return-correlation screen. Yesterday's 20-session leader had to fall, the laggard had to rise, and their indexed gap had to contract without crossing. The frozen rule enters after 10:30 New York time, buys the leader, shorts the laggard in equal dollars, and targets exit the next session. The replay reproduced every published daily dollar PnL in both validation and both test splits. Gross PnL, execution charges, financing, borrow and net PnL reconcile separately from filled units, including partial exits and distributions. [Original rule and eligibility](../inverse_gap_reopening/report.md) · [Simulated fills](fills.parquet) · [Daily marks](daily_pnl.parquet)

| Reused test period and eligible universe | Trades | Frozen pair | Factor-sized leader | Pair scaled to common exposure | Signed hedge |
|---|---:|---:|---:|---:|---:|
| 2025, all | 23 | −0.81% | −0.63% | −0.70% | +0.01% |
| 2025, excluding SCO/USO | 22 | −1.09% | −0.88% | −0.98% | +0.04% |
| 2026 through Oct 23, all | 34 | +7.08% | +7.44% | +6.52% | +1.26% |
| 2026 through Oct 23, excluding SCO/USO | 28 | +4.48% | +3.22% | +3.08% | +0.01% |

Before each entry, estimate both legs' loadings on 60 completed daily returns for SPY, TLT, USO and three equity return differences (QQQ, IWM and DIA less SPY). Choose the long-only leader size that minimizes the historical variance of the modeled factor PnL difference. Financing, gross limits, participation limits, signal times and intended exits are common. Actual fills and partial exits can differ. When the leader hits a cap, the common-exposure comparison scales the pair down to the factor exposure the leader can attain along that projection. A single asset cannot generally match six loadings exactly. Median remaining daily modeled mismatch on the scaled trade was $0.18 in 2025 and $6.11 in 2026. The earlier report used a total-volatility ratio to size its leader benchmark. [Per-entry ratios, sizes and residual risk](exposure_matching.csv)

Across all pairs the frozen pair trailed the factor-sized leader by **0.17 percentage points in 2025** (paired 95% interval −0.88 to +0.47) and **0.36 points in 2026** (−2.64 to +1.84). Excluding SCO/USO before candidate ranking allowed replacement trades. The apparent 2026 advantage of +1.26 points excluding that pair (−0.55 to +3.46) became **−0.14 points** (−0.52 to +0.21) at common attainable exposure. Confidence intervals use 1,999 paired circular resamples of ten sessions. One-sided HAC tests use ten lags with Holm adjustment within each comparison across folds and universes. Validation and test splits are reported separately. Calendar periods and underlying exposures overlap across experiments. [Paired tests](paired.csv)

## What generated the PnL?

| Frozen pair, all candidates | 2025 | 2026 through Oct 23 |
|---|---:|---:|
| Long prior leader, net | −$180 | +$3,778 |
| Short rising laggard, net | −$629 | +$3,306 |
| Total net, before rounding legs | −$808 | +$7,084 |
| Estimated PnL from factor exposures | −$717 | +$7,273 |
| Net after subtracting modeled factor PnL | −$91 | −$189 |

The attribution multiplies pre-entry factor loadings by realized proxy returns between the entry and each partial exit, including distributions. Each proxy mark must be available at or before the fill within five minutes. The 2026 modeled factor PnL exceeded the entire book profit. The proxy set includes USO itself, which matters given the concentration in SCO/USO. Omitted currency, term-structure and other risks, changing intraday betas and sampling error limit interpretation of the residual as alpha. [Legs, loadings, carry and costs](legs.csv)

The signed-hedge diagnostic estimates the slope of the leader on the laggard over the same trailing 60 sessions and enters weights proportional to (1, −slope). Median slopes at entry were −1.00 and −0.72 in the respective test periods. A negative slope calls for buying both legs to offset the inverse relationship. The original opposite-side trade reinforced it. The signed hedge earned about $1,260 in 2026 and $7 after excluding SCO/USO. These are diagnostic outcomes on reused entries. [Portfolio attribution](books.csv)

## Does the laggard add predictive information?

The candidate sample includes large-gap leader pullbacks before imposing the rising-laggard filter. The fixed Ridge benchmark uses the leader's move since the prior close, its literal first-hour return, prior 20-session momentum, trailing 60-session volatility and six market-factor moves since the prior close. Two variants each add one laggard return and its positive-return indicator. One variant measures from the prior close, as in the frozen strategy. The other measures from the actual 9:30 open to the close of the 10:29 bar. Either can differ in sign after an overnight gap.

Both regression targets are gross returns between synchronized marks at or after the intended entry and exit times. Models share complete-case rows and use training-only scaling, fixed Ridge penalty 100 and training labels that mature before validation opens. All eligible opportunities enter the forecast test, including dates when the trading book was occupied. Forecast metrics weight each pair-date equally. Inference averages paired loss differences within each date first, then applies HAC and Holm correction across the 16 test comparisons. Factor proxies may themselves be traded legs, so the question is the value of information beyond these observable controls. [Forecast metrics](forecast_quality.csv) · [2025 predictions](forecasts_2025_test.parquet) · [2026 predictions](forecasts_2026_test.parquet)

| Change in next-session pair forecast MSE versus leader and factors; lower is better | 2025 | 2026 through Oct 23 |
|---|---:|---:|
| Add laggard move since prior close | −0.23% | +10.44% |
| Add laggard move since 9:30 open | +0.18% | −0.76% |
| Pair-date observations / distinct signal dates | 375 / 35 | 252 / 53 |

The literal opening hour improved 2026 pair rank correlation from 0.172 to 0.194 and worsened 2025 MSE. No tested augmentation established an incremental reduction in forecast error after adjustment. The limited number of independent dates restricts power. These diagnostics do not establish that the leader-only effect itself is profitable prospectively.

## Execution evidence

The primary replay charges 2 bp of traded notional per fill, 2% annualized financing on gross positions and an additional 0.5% on short inventory. A separate sensitivity estimates an effective half-spread from the negative first-order covariance of consecutive minute log changes over 20 prior sessions. It replaces the 2 bp charge with the larger of 2 bp and the estimated half-spread. Nonnegative covariance or insufficient observations leave the estimate unidentified; the original 2 bp charge then applies. This fallback covered 77.6% of pair turnover in 2025 and 47.9% in 2026. The resulting pair PnL was −$811 and +$7,058. Minute-close estimates cannot resolve intraminute spreads or market impact. [Cost scenarios](books.csv) · [Roll's original estimator](https://doi.org/10.1111/j.1540-6261.1984.tb03897.x)

The pinned market archive contains trades aggregated into bars. Historical NBBO and broker locate records were unavailable in this study. For the original full-universe book, zero of 124 fills in 2025 or 200 fills in 2026 have quote confirmation, and zero of the 23 and 34 short entries have documentary locate evidence. Realized executable returns remain unknown. Requiring a documented locate from this archive would leave the pair book in cash. [Fill-level coverage](execution_evidence/coverage.csv) · [Pinned minute source](https://huggingface.co/datasets/syncster-data/historical-market-data/tree/eff0dac55d843b3b83caec819efa213021a3e3e5)

## Forward test and reproduction

Protocol commit `904bf57` fixed the evaluation for **November 19 through December 31, 2026**, before its first signal. It uses the 2026 deployment list, original gap cutoffs, two prespecified universes and one endpoint per universe, the paired net difference versus the factor-sized leader. At registration the common snapshot ended October 23. No prospective result is available. The runner requires extended daily and minute histories and the full registered calendar; it refuses interim scoring. The original wind-down reserves the final four sessions for completing exits. A six-week test can yield very few trades. [Frozen protocol](../../../config/inverse_short_leg_forward.json) · [NYSE session calendar](https://www.nyse.com/trade/hours-calendars)

With the [research dependencies](../../../requirements-research.txt) and the [original input snapshots](../strict_inverse_alpha_final/manifest.json), run from the repository root. Each command writes to a new directory.

```shell
python -m pytest tests -q -p no:cacheprovider
python -m experiments.convergence.short_leg_diagnostics --output outputs/test_tmp/short_leg_replay
python -m experiments.convergence.short_leg_execution_evidence --fills outputs/test_tmp/short_leg_replay/fills.parquet --output outputs/test_tmp/short_leg_coverage
```

After the registered interval, supply full-history daily chart JSON and minute Parquet directories on the original data conventions, including corporate actions and the calibration period from 2020. Input checksums are saved with each run.

```shell
python -m experiments.convergence.short_leg_diagnostics --forward-daily PATH_TO_DAILY --forward-minute PATH_TO_MINUTES --output outputs/runs/inverse_short_leg_forward
```

For a fill-level evidence audit, add `--quotes QUOTES.csv --locates LOCATES.csv` to the evidence command. Quotes require `symbol,ts,bid,ask,bid_size,ask_size,price_multiplier`. Locates require `symbol,available_at,valid_until,max_shares,price_multiplier`. Times must carry UTC offsets, quantities are shares, and each multiplier converts vendor price to the ledger's corporate-action basis. Use broker-confirmed account-specific inventory. The audit checks the latest quote strictly before bar end within two seconds, side-specific displayed depth, and the latest locate available by the 10:30 decision that remains valid through the entry. The signed dollar `quote_pnl_change` reprices the same units against the applicable quote; add it to gross PnL and model commission, carry and borrow separately. A missing quote, insufficient depth or unknown locate leaves execution unverified. The audit measures quote coverage on simulated orders and cannot establish queue position, impact, actual fills or borrow recalls. [Evidence reader](../../../experiments/convergence/short_leg_execution_evidence.py) · [Snapshot and source checksums](manifest.json)
