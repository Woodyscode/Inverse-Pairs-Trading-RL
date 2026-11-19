"""Paired granularity comparisons, archived PPO comparisons and accounting checks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from strict_alpha_study import CAPITAL, FEE, SEED, hac_p, holm


def block_interval(series):
    x = np.asarray(series)
    rng = np.random.default_rng(SEED)
    starts = rng.integers(len(x), size=(1999, (len(x) + 9) // 10))
    ix = ((starts[:, :, None] + np.arange(10)) % len(x)).reshape(1999, -1)[:, : len(x)]
    return np.quantile(x[ix].sum(axis=1), [0.025, 0.975])


def summarize(root, runs):
    dest = root / "evidence"
    dest.mkdir(exist_ok=False)
    comparisons = []
    strata = []
    for year in (2025, 2026):
        for split in ("validation", "test"):
            folder = root / str(year) / split
            equity = pd.read_parquet(folder / "daily_equity.parquet")
            fills = pd.read_parquet(folder / "executions.parquet")
            trades = pd.read_parquet(folder / "trades.parquet")
            np.testing.assert_allclose(fills.fee, abs(fills.notional) * FEE)
            g = ["model", "horizon"]
            fees = fills.groupby(g).fee.sum()
            pnl = trades.groupby(g).net.sum()
            marks = equity.groupby(g).pnl.sum()
            np.testing.assert_allclose(
                fees, trades.groupby(g).execution.sum(), atol=1e-7
            )
            np.testing.assert_allclose(
                marks, pnl.reindex(marks.index, fill_value=0), atol=1e-7
            )
            totals = (
                trades.groupby([*g, "a", "b", "regime"])
                .agg(
                    trades=("net", "size"),
                    net=("net", "sum"),
                    gross=("gross", "sum"),
                    execution=("execution", "sum"),
                    borrow=("borrow", "sum"),
                    financing=("funding", "sum"),
                    median_size=("capacity", "median"),
                )
                .reset_index()
            )
            strata.append(totals.assign(fold=year, split=split))
            for h in (0, 1, 5, 10):
                pred = pd.read_parquet(folder / f"forecasts_{h}.parquet").dropna(
                    subset="label"
                )
                returns = (
                    equity[equity.horizon.eq(h)].pivot(
                        index="date", columns="model", values="pnl"
                    )
                    / CAPITAL
                )
                for model in ["linear", "ridge", "xgboost", "continuation", "reversal"]:
                    a, b = model + "_minute", model + "_daily"
                    paired = returns[a] - returns[b]
                    gain = (
                        ((pred.label - pred[b]) ** 2 - (pred.label - pred[a]) ** 2)
                        .groupby(pred.date)
                        .mean()
                    )
                    low, high = block_interval(paired)
                    comparisons.append(
                        dict(
                            fold=year,
                            split=split,
                            model=model,
                            horizon=h,
                            net_difference=paired.sum(),
                            lower=low,
                            upper=high,
                            net_p=hac_p(paired),
                            mse_gain=gain.mean(),
                            mse_p=hac_p(gain),
                        )
                    )
    pd.concat(strata).to_csv(dest / "attribution.csv", index=False)
    paired = pd.DataFrame(comparisons)
    for split, indices in paired.groupby("split").groups.items():
        p = paired.loc[indices, ["mse_p", "net_p"]].to_numpy()
        paired.loc[indices, ["mse_holm", "net_holm"]] = holm(p.ravel()).reshape(p.shape)
    paired.to_csv(dest / "paired_granularity.csv", index=False)

    legacy = runs / "residual_control_liquidity_50k"
    ledger = pd.read_csv(legacy / "bars.csv").query("split=='test'")
    returns = ledger.pivot(index="date", columns="policy", values="net_return")
    matched = []
    for agent, benchmark in [
        ("ppo_ridge", "ridge"),
        ("ppo", "cash"),
        ("ppo_ridge", "xgboost"),
        ("ppo", "zscore"),
    ]:
        diff = returns[agent] - returns[benchmark]
        lo, hi = block_interval(diff)
        matched.append(
            dict(
                agent=agent,
                benchmark=benchmark,
                additive_net_difference=diff.sum(),
                lower=lo,
                upper=hi,
                p=hac_p(diff),
                sessions=len(diff),
            )
        )
    matched = pd.DataFrame(matched)
    matched["holm"] = holm(matched.p)
    matched.to_csv(dest / "historical_ppo_paired.csv", index=False)
    synthetic = []
    for f in (runs / "strict_alpha_synthetic").glob(
        "*/synthetic/independent_holdouts.csv"
    ):
        synthetic.append(pd.read_csv(f).assign(optimizer_seed=int(f.parents[1].name)))
    pd.concat(synthetic).to_csv(dest / "synthetic_continuations.csv", index=False)
    budgets = []
    for name in [
        "residual_control_256",
        "residual_control_50000",
        "residual_control_250k",
        "residual_control_1m",
        "residual_control_liquidity_50k",
    ]:
        for policy in ["ppo", "ppo_ridge"]:
            source = runs / name / "synthetic" / (policy + ".validation.csv")
            frame = pd.read_csv(source)
            row = frame[frame.selected_checkpoint].iloc[0].to_dict()
            budgets.append(
                dict(run=name, policy=policy, budget=frame.steps.max(), **row)
            )
    pd.DataFrame(budgets).to_csv(dest / "synthetic_budget_checkpoints.csv", index=False)
    inputs = [
        Path(__file__),
        Path("strict_control_validation.py"),
        Path("env/spread_control.py"),
        Path("env/features/residual.py"),
        Path("residual_control_study.py"),
        legacy / "manifest.json",
        legacy / "bars.csv",
        runs / "strict_daily_alpha_coverage" / "manifest.json",
        *(runs / "strict_control_rewards").glob("*"),
        *(runs / "strict_alpha_synthetic").glob("*/*/*.zip"),
    ]
    (dest / "provenance.json").write_text(
        json.dumps(
            {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=Path("outputs/runs"))
    args = parser.parse_args()
    summarize(args.root, args.runs)
