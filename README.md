# Strict inverse-pairs trading research

Does a short position in a rising ETF add value when its historically inverse partner pulls back? In two retrospective periods, buying the falling prior leader alone earned more than buying the leader and shorting the laggard on the same entries. **The short leg has yet to demonstrate incremental alpha.** [Attribution report](outputs/runs/inverse_short_leg_review/report.md) · [Paired tests](outputs/runs/inverse_short_leg_review/paired.csv) · [Frozen forward protocol](config/inverse_short_leg_forward.json)

## The matched trade

The universe includes equity, fixed-income, currency and commodity ETFs. Each pair must have a training-period return correlation of **−0.80 or lower**, pass three stability subwindows and meet liquidity requirements. Rank the two ETFs by their prior 20-session total returns. If a historically large gap narrows by 10:30 a.m. New York time, with the prior leader down, the laggard up and the leader still ahead, buy the leader and short the laggard in equal dollars. The trade targets an exit the following session. [Qualification ledger](outputs/runs/strict_inverse_alpha_final/2026/qualification.csv) · [Signal and execution rules](outputs/runs/inverse_gap_reopening/report.md)

Buying an index ETF and shorting its inverse can double up on the same market direction. To test what the second instrument contributes, the benchmark buys the leader on the same dates and uses the preceding 60 sessions to approximate the pair's exposure to six equity, Treasury and oil factors. Both portfolios face the same gross and liquidity limits. [Position sizes and remaining factor mismatch](outputs/runs/inverse_short_leg_review/exposure_matching.csv)

| Simulated net return on $100,000 | 2025 | 2026 through Oct 23 |
|---|---:|---:|
| Completed pair trades | 23 | 34 |
| Buy prior leader, short rising laggard | −0.81% | +7.08% |
| Buy prior leader alone, sized to estimated factor exposure | −0.63% | +7.44% |
| **Two-leg advantage** | **−0.17 pp** | **−0.36 pp** |
| Paired 95% block-bootstrap interval for the advantage | −0.88 to +0.47 pp | −2.64 to +1.84 pp |

Differences use unrounded returns. In 2026 the short earned **$3,306** and the long earned **$3,778** after modeled costs. Estimated factor PnL of $7,273 left a **−$189** residual for the pair. Its magnitude depends on the factor model, which includes USO. Excluding SCO/USO before ranking and scaling both books to attainable factor exposure left the pair behind by 0.14 percentage points. The intervals permit both a gain and a loss from adding the short. [Portfolio results](outputs/runs/inverse_short_leg_review/books.csv) · [Long and short attribution](outputs/runs/inverse_short_leg_review/legs.csv)

### Execution and inference limits

Orders use later, synchronized minute-bar prices. The $100,000 account targets up to $25,000 gross per trade. Each leg is capped at 1% of both trailing average daily dollar volume and observed fill-minute dollar volume. The simulation charges 2 bp per filled dollar, 2% annual financing on gross inventory and an additional 0.5% on short inventory. Historical bid/ask quotes and account-specific locates were unavailable for all 57 short entries in these retrospective tests. A bar-based spread sensitivity and a fill-level coverage ledger quantify what can be checked. [Simulated fills](outputs/runs/inverse_short_leg_review/fills.parquet) · [Quote and locate coverage](outputs/runs/inverse_short_leg_review/execution_evidence/coverage.csv)

The prospective test runs **November 19 through December 31, 2026**. The registered endpoint is the paired difference in daily net PnL, across all eligible pairs and after excluding SCO/USO before ranking. Scoring waits for the complete period. With 23 and 34 completed historical trades, a six-week window may leave the economic question unresolved. The leader pullback remains the benchmark and historical PPO training on this signal is deferred. [Registered protocol at commit 904bf57](https://github.com/Woodyscode/Inverse-Pairs-Trading-RL/blob/904bf57/config/inverse_short_leg_forward.json)

## Related experiments

| Research question | Finding and evidence |
|---|---|
| Does the gap reopen the next session? | It reopened in 50.0% of scorable 2025 events and 51.1% in 2026. Wavelet confirmation reduced net returns in both periods. [Event study and Donoho–Johnstone filter](outputs/runs/inverse_gap_reopening/report.md) |
| Does the laggard's opening hour improve the forecast? | Adding its 9:30–10:30 return to the leader and market-factor model increased pair forecast error by 0.18% in 2025 and reduced it by 0.76% in 2026. No improvement survived adjustment for multiple tests. [Forecast diagnostics](outputs/runs/inverse_short_leg_review/forecast_quality.csv) |
| Do inverse-return tracking errors predict tradable returns? | Daily and opening-hour features were tested on the same qualified pairs at four horizons. Neither deployment selected a forecast that cleared both the predictive and after-cost validation gates. The selected book held cash. [Forecasting study](outputs/runs/strict_inverse_alpha_final/report.md) · [Execution cost review](docs/execution_research_review.md) |
| Can PPO improve portfolio control? | In synthetic mean-reverting markets PPO learned across three training seeds. In the archived daily comparison, fixed Ridge and plain PPO stayed flat, and PPO with Ridge lost 1.37 bp. [Synthetic continuations](outputs/runs/strict_inverse_alpha_final/evidence/synthetic_continuations.csv) · [Historical comparison](outputs/runs/strict_inverse_alpha_final/evidence/historical_ppo_paired.csv) |

## Verify and reproduce

**Review the headline result offline.** The CSV and Parquet evidence is committed. From the repository root, the following Python snippet prints trade counts and net dollar PnL for the two books behind the headline table. It uses the standard library.

```python
import csv
from pathlib import Path

books = Path("outputs/runs/inverse_short_leg_review/books.csv")
with books.open(newline="", encoding="utf-8") as ledger:
    for row in csv.DictReader(ledger):
        if (row["split"], row["scope"]) == ("test", "all") and row["book"] in {"pair", "factor_leader"}:
            print(row["fold"], row["book"], row["trades"], round(float(row["net"]), 2))
```

**Replay from market data.** The published research environment used CPython 3.14.2. Install the pinned dependencies in a fresh environment. The fetch command needs network access for missing snapshots. The replay verifies market checksums and writes to a new output directory. Later vendor revisions to the daily data can prevent exact replication. [Research dependencies](requirements-research.txt) · [Source and snapshot checksums](outputs/runs/inverse_short_leg_review/manifest.json)

```shell
python -m pip install -r requirements-research.txt
python strict_alpha_study.py --fetch
python short_leg_diagnostics.py --output outputs/test_tmp/short_leg_replay
python short_leg_execution_evidence.py --fills outputs/test_tmp/short_leg_replay/fills.parquet --output outputs/test_tmp/short_leg_coverage
python -m pytest tests -q -p no:cacheprovider
```

The earlier forecast study has a separate offline ledger checker, `python -m backtest.execution_review`. The research history includes the daily universe, synthetic PPO experiments and the separate SPY-hedged relative-value study. [Forecast study reproduction](outputs/runs/strict_inverse_alpha_final/report.md#scope-and-reproducibility) · [Research history](RESEARCH_HISTORY.md)

## Codebase audit

The original audit identified timing leakage, missing out-of-sample feature history, future-data backfilling and costs charged on PnL. The repair report documents the corrections and rerun. The post-repair ledgers above are the basis for the reported findings. [Project audit](outputs/audit/inverse_pairs_codebase_audit.md) · [Repair report](outputs/audit/inverse_pairs_repair_report.md)
