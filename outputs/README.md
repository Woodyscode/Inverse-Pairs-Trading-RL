# Published research results

For the current conclusion, read the [short-leg attribution report](runs/inverse_short_leg_review/report.md). The table points to the evidence behind each experiment. Dollar amounts and returns in these ledgers are simulated.

| Question | Results | Data to inspect |
|---|---|---|
| Did shorting the rising laggard add value? | [Matched portfolio analysis](runs/inverse_short_leg_review/report.md) | [Portfolio PnL](runs/inverse_short_leg_review/books.csv) · [Individual legs](runs/inverse_short_leg_review/legs.csv) · [Paired intervals](runs/inverse_short_leg_review/paired.csv) |
| Did the gap reopen after convergence? | [Gap-reopening event study](runs/inverse_gap_reopening/report.md) | [Event counts](runs/inverse_gap_reopening/coverage.csv) · [Completed trades](runs/inverse_gap_reopening/trades.csv) |
| Which inverse pairs qualified? | [Cross-asset discovery](runs/inverse_discovery_universe/report.md) | [Candidate-level screens and rejections](runs/inverse_discovery_universe/candidate_screen.csv.gz) |
| Did finer data or alternative forecasts help? | [Daily and minute study](runs/strict_inverse_alpha_final/report.md) · [Execution cost review](../docs/execution_research_review.md) | [Paired daily and minute results](runs/strict_inverse_alpha_final/evidence/paired_granularity.csv) · [Wavelet comparison](runs/inverse_residual_dj/paired.csv) |
| Did PPO add value? | [Historical policy comparison](runs/strict_inverse_alpha_final/evidence/historical_ppo_paired.csv) | [Independent synthetic continuations](runs/strict_inverse_alpha_final/evidence/synthetic_continuations.csv) · [Reward specifications](runs/strict_control_rewards/comparison.csv) |

The [original audit](audit/inverse_pairs_codebase_audit.md) and [repair report](audit/inverse_pairs_repair_report.md) document why the early backtest was discarded. The [research history](../RESEARCH_HISTORY.md) records earlier iterations. The reproduction examples write local replays to `outputs/test_tmp/`. Published studies live under `outputs/runs/`.


## Verify the headline result offline

The CSV and Parquet evidence is committed. From the repository root, the following Python snippet prints trade counts and net dollar PnL for the two books behind the headline table. It uses the standard library.

```python
import csv
from pathlib import Path

books = Path("outputs/runs/inverse_short_leg_review/books.csv")
with books.open(newline="", encoding="utf-8") as ledger:
    for row in csv.DictReader(ledger):
        if (row["split"], row["scope"]) == ("test", "all") and row["book"] in {"pair", "factor_leader"}:
            print(row["fold"], row["book"], row["trades"], round(float(row["net"]), 2))
```
