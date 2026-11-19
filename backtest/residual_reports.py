"""Summarize predictive accuracy and separate cash-only dates from policy decisions."""

from pathlib import Path

import numpy as np
import pandas as pd


def summarize_study(root):
    root = Path(root)
    bars = pd.read_csv(root / "bars.csv")
    reports = []
    keys = ["split", "policy"]
    if "window" in bars:
        keys.append("window")
    for group, records in bars.groupby(keys):
        row = dict(zip(keys, group))
        decisions = records.loc[records.capacity.notna()]
        prior = decisions.equity - decisions.gross_pnl + decisions.costs
        net = (decisions.gross_pnl - decisions.costs) / prior
        np.testing.assert_allclose(net, decisions.net_return, atol=1e-12)
        np.testing.assert_allclose(
            100 * (net - decisions.risk_penalty - decisions.turnover_penalty),
            decisions.reward,
            atol=1e-12,
        )
        row.update(
            decisions=len(decisions),
            bars=len(records),
            max_exit_cap_multiple=decisions["exit_adv_multiple"].max()
            if "exit_adv_multiple" in decisions
            else np.nan,
            net_return=(1 + records.net_return).prod() - 1,
            gross_return_sum=(decisions.gross_pnl / prior).sum(),
            cost_return_sum=(decisions.costs / prior).sum(),
            risk_penalty_sum=decisions.risk_penalty.sum(),
            turnover_penalty_sum=decisions.turnover_penalty.sum(),
            short_freq=(decisions.action == -1).mean(),
            flat_freq=(decisions.action == 0).mean(),
            long_freq=(decisions.action == 1).mean(),
            held_flat_freq=(decisions.position == 0).mean(),
            gross_exposure=decisions.gross.mean(),
            policy_entropy=decisions.entropy.mean(),
            liquidation_notional=(
                decisions.turnover * prior - decisions.filled_notional
            ).sum(),
        )
        reports.append(row)
    pd.DataFrame(reports).to_csv(root / "policy_diagnostics.csv", index=False)
    paths = list(root.glob("window_*/forecasts.csv"))
    if (root / "forecasts.csv").exists():
        paths.append(root / "forecasts.csv")
    predictions = []
    for path in paths:
        df = pd.read_csv(path)
        for (split, model), group in df.groupby(["split", "model"]):
            y, p = group.realized, group.prediction
            predictions.append(
                {
                    "fold": path.parent.name,
                    "split": split,
                    "model": model,
                    "n": len(y),
                    "pearson": p.corr(y),
                    "rank_ic": p.corr(y, method="spearman"),
                    "directional_accuracy": (np.sign(p) == np.sign(y)).mean(),
                    "rmse": np.sqrt(np.mean((p - y) ** 2)),
                    "zero_forecast_rmse": np.sqrt(np.mean(y**2)),
                }
            )
    if predictions:
        pd.DataFrame(predictions).to_csv(root / "forecast_quality.csv", index=False)
