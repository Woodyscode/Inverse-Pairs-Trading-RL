# Strict inverse-pairs trading research

When a historically inverse ETF pair moves closer together, does buying the falling prior leader and shorting the rising laggard pay? In two retrospective tests, the leader alone earned more on the same entries after sizing it to approximate the pair's factor exposure. **The short leg has yet to demonstrate incremental alpha.** [Read the attribution report](outputs/runs/inverse_short_leg_review/report.md)

| Simulated net return on $100,000 | 2025 | 2026 through Oct 23 |
|---|---:|---:|
| Completed pair trades | 23 | 34 |
| Buy the prior leader and short the laggard | −0.81% | +7.08% |
| Buy the factor-matched leader alone | −0.63% | +7.44% |
| **Two-leg advantage** | **−0.17 pp** | **−0.36 pp** |
| Paired 95% interval for the advantage | −0.88 to +0.47 pp | −2.64 to +1.84 pp |

Differences use unrounded returns. In 2026 the short earned $3,306 and the long earned $3,778. The six-factor model attributed $7,273 to market exposures, leaving a −$189 residual. That estimate depends on the factors, including the oil proxy USO. The simulation used minute bars, per-leg costs and liquidity limits. Historical quotes and broker locates were unavailable. [Check the portfolio totals offline](outputs/README.md#verify-the-headline-result-offline) · [Paired comparisons](outputs/runs/inverse_short_leg_review/paired.csv)

## The experiment

The universe spans equity, fixed-income, currency and commodity ETFs. Eligible pairs must have training-period return correlation at or below **−0.80** and pass liquidity and stability screens. At 10:30 a.m. New York time, the rule looks for a previously large 20-session performance gap that has narrowed with the leader still ahead. It buys the falling leader and shorts the rising laggard in equal dollars, targeting an exit the next session. Buying an ETF and shorting its inverse can reinforce exposure to the same market move. The leader-only benchmark measures how much the second leg adds. [Signal definition and execution rules](outputs/runs/inverse_gap_reopening/report.md)

The comparison is frozen for **November 19 through December 31, 2026**, across the full eligible universe and with SCO/USO excluded before ranking. The leader pullback remains the benchmark, and historical PPO training on this signal is deferred. [Forward protocol](config/inverse_short_leg_forward.json)

## Explore the repository

| To review | Start here |
|---|---|
| Results, fills and statistical comparisons | [Published results by research question](outputs/README.md) |
| Signal implementation, portfolio replay and reproduction commands | [Research code by study](experiments/README.md) |
| Defects in the original backtest and their corrections | [Codebase audit](outputs/audit/inverse_pairs_codebase_audit.md) · [Repair report](outputs/audit/inverse_pairs_repair_report.md) |
| Earlier daily, minute, wavelet and PPO experiments | [Research history](RESEARCH_HISTORY.md) |
