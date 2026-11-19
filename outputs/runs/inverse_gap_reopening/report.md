# Do inverse-pair performance gaps reopen after convergence?

**The trade bought the falling prior leader and shorted the rising laggard.** Across all scorable test events, the prior leader resumed outperforming between executable marks in 182 of 364 cases in 2025 and 117 of 229 through October 23, 2026. Only 25.5% and 31.4%, respectively, fully recovered the previous close's gap by the next target mark. Each observation is a pair and a date. Related ETFs can share the same underlying move. [Event counts](coverage.csv)

## Entry and exit

The experiment reused the archived point-in-time deployment screens for equity, currency, commodity and bond ETFs. There were 28 eligible inverse pairs in each test period. Training return correlation had to be at or below −0.80 overall and across three calibration subwindows. Funding, borrow, fill timestamps, per-leg participation and fee assumptions came from the original strict study. [Archived methodology](../strict_inverse_alpha_final/report.md)

At yesterday's close, rank the two assets by their total returns over the previous 20 sessions. Express the gap as a difference in cumulative log returns from that common starting close. Require its absolute size to reach the pair's training-period 80th percentile. At 10:30 New York time, require the prior leader to be down from the previous close, the prior laggard to be up, and the leader still to be ahead. These conditions make the gap contract without crossing zero. Morning returns include ex-date distributions.

When flat, choose the qualifying pair with the largest prior gap relative to its training cutoff. Buy the leader and short the laggard in equal dollars. Submit after the signal, start filling at the close of the 10:31 bar, and target the same execution time the next session. Each $100,000 book targets at most $25,000 gross, subject to 25% of available equity and 1% of observed minute dollar volume and trailing ADV per leg. Each fill costs 2 bp of traded notional. Financing costs 2% annualized on gross inventory and borrow adds 0.5% on the short. A trade requires a locate on the laggard. Actual stock-loan inventory and bid–ask quotes were unavailable. At entry, the legs offset in dollars but can reinforce the same underlying market exposure.

## Comparisons and results

| Net account return | 2025 (full year) | 2026 (through Oct 23) |
|---|---:|---:|
| Buy falling leader, short rising laggard | −0.81% (23 trades) | +7.08% (34 trades) |
| Buy the leader alone on the same entries | −0.64% | +7.41% |
| Two-leg trade after a leader pullback, without requiring the laggard to rise | −0.46% | +8.66% |
| Two-leg trade with wavelet confirmation | −0.88% | +6.72% |
| Cash | 0% | 0% |

The leader-only benchmark uses the trailing 60-session volatility ratio of the equal-dollar basket to the leader to target comparable risk, subject to gross and liquidity caps. Those caps can prevent exact risk matching. The two books share entry dates, intended holding periods and a common flat-before-reentry requirement. The pullback benchmark trades the same opposite legs, allowing entry when the leader falls and the laggard does not rise. Wavelet confirmation adds a requirement that the filtered gap also move toward zero. All strategies use observed prices for fills and PnL. [Metrics](metrics.csv) · [Trades](trades.csv) · [Per-leg fills](executions.parquet)

The primary rule lost 3.49% in the first validation period and 0.06% in the second. In the 2026 test, it earned $7,338 gross and $7,084 after costs. SCO/USO accounted for $6,471 across 19 of the 34 trades. The pair trade trailed its matched leader-only book by 0.32 percentage points, with a paired 95% ten-session block-bootstrap interval of −2.74 to +1.89 points. The 2026 comparison against cash had a one-sided autocorrelation-adjusted p-value of 0.035 and a Holm-adjusted p-value of 0.279 across the eight prespecified test-period comparisons. The evidence supports retaining the simple leader-pullback benchmark. These calendar periods had already been examined for other signals and are retrospective for this hypothesis. [Paired tests](paired.csv) · [Daily marked equity](daily_equity.parquet)

## Causal wavelet confirmation

At the prior close and at the decision time, take the endpoint of a separate trailing 64-observation window of relative log wealth. Apply a one-level db4 transform, preserving the approximation coefficients. Estimate noise from the finest detail using median absolute deviation divided by 0.67449. Soft-threshold the detail coefficients at the Donoho–Johnstone universal threshold, σ̂√(2 ln 64). A sufficiently large price shock can still be signal, and thresholding can delay a turn. Here confirmation filtered out trades and reduced held-out net return in both periods. [Wavelet implementation](../../../data/wavelet.py)

A companion Ridge study tested the same denoising method on the earlier return-tracking hypothesis. It used 64 daily residuals or the final 32 opening-hour minutes, with raw and three-observation moving-average controls on identical valid rows. For daily inputs in 2026 the filtered book gained 2.81% versus 0.33% raw, but the paired interval included zero and no adjusted predictive or economic comparison established an improvement. Its findings concern the tracking residual, a different trade from the gap-reopening rule above. [Residual study results](../inverse_residual_dj/paired.csv) · [Coverage and signal attenuation](../inverse_residual_dj/coverage.csv)

## Reproduction

From the repository root, restore the market snapshots specified in the archived manifest, install the research dependencies, and use new output directories. The runners verify the market-input checksums before simulation. The portfolio metrics count completed trades. The event outcomes include every scorable contraction, even when another position occupied the book. Gap reopening measures the original leader's relative return from the synchronized entry mark to the next target mark. Full recovery compares the target mark to the pre-contraction close using the original anchor.

```shell
python strict_alpha_study.py --fetch
python inverse_convergence_study.py --output outputs/test_tmp/gap_reopening_replica
python wavelet_residual_study.py --output outputs/test_tmp/dj_residual_replica
python -m pytest tests -q -p no:cacheprovider
```
