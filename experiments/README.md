# Research code

Start with [the convergence study](convergence/inverse_convergence_study.py) for the entry signal and [the short-leg replay](convergence/short_leg_diagnostics.py) for the comparison with the leader-only portfolio. The [results guide](../outputs/README.md) links each question to its evidence.

| Directory | What runs here |
|---|---|
| [convergence](convergence/) | Detect narrowing performance gaps, replay the matched portfolios, attribute PnL to each leg and audit quote and locate coverage. |
| [discovery](discovery/) | Screen cross-asset pairs for strict negative correlation and test deviations from their fitted inverse-return relationship. |
| [forecasting](forecasting/) | Compare daily and minute inputs, forecast horizons, linear models, XGBoost and causal wavelet filters. Collect paired comparisons. |
| [portfolio_control](portfolio_control/) | Train and evaluate PPO and fixed policies on synthetic spreads and historical data. |
| [early_studies](early_studies/) | Run the post-audit validation, daily diagnostics and early strategy comparisons. |
| [relative_value](relative_value/) | Explore the separate SPY-hedged and cointegrated-stock hypotheses. |

## Replay the short-leg comparison

Run from the repository root. The recorded environment used CPython 3.14.2. Install the pinned dependencies into a fresh environment and choose new output directories. The fetch step can access the network for missing market data. The replay checks the input hashes and will reject a vendor revision that differs from the published snapshot. [Dependencies](../requirements-research.txt) · [Snapshot manifest](../outputs/runs/inverse_short_leg_review/manifest.json)

```shell
python -m pip install -r requirements-research.txt
python -m experiments.forecasting.strict_alpha_study --fetch
python -m experiments.convergence.short_leg_diagnostics --output outputs/test_tmp/short_leg_replay
python -m experiments.convergence.short_leg_execution_evidence --fills outputs/test_tmp/short_leg_replay/fills.parquet --output outputs/test_tmp/short_leg_coverage
python -m pytest tests -q -p no:cacheprovider
```

The [attribution report](../outputs/runs/inverse_short_leg_review/report.md#forward-test-and-reproduction) gives the commands for scoring the full prospective period and auditing broker evidence. The [research history](../RESEARCH_HISTORY.md) covers earlier workflows, including the root training and backtest entry points. To follow older commands that invoke a study directly from the root, see the [last checkout with the original layout](https://github.com/Woodyscode/Inverse-Pairs-Trading-RL/tree/833ce20). Each published run records its own source hashes.
