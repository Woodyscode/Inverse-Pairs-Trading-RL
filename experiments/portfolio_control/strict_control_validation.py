"""Synthetic portfolio-control reward ablation and replay of archived PPO seeds."""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torch

from env.spread_control import SpreadControlEnv
from experiments.portfolio_control.residual_control_study import (
    evaluate,
    make_data,
    replicate_synthetic,
    stats,
    synthetic,
    train_policy,
)


def run(output, replay):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=False)
    runs = Path("outputs/runs")
    sources = {7: "rv_seed_7", 42: "residual_control_liquidity_50k", 83: "rv_seed_83"}
    cfg = json.loads((runs / sources[42] / "manifest.json").read_text())["settings"]
    if replay:
        for seed, folder in sources.items():
            dest = output / str(seed) / "synthetic"
            dest.mkdir(parents=True)
            for name in ("ppo.zip", "ppo_ridge.zip"):
                shutil.copy2(runs / folder / "synthetic" / name, dest / name)
            shutil.copy2(runs / folder / "manifest.json", dest.parent / "manifest.json")
            replicate_synthetic(dest, 42, cfg, 100000, 5)
        return
    prices, volume, _, truth = synthetic(42)
    data = make_data(prices, volume, prices * volume, 1400, [], truth)
    error = pd.Series(data["target"] - data["alpha"][:, 0]).shift(2)
    # Warm-up uses the observed volatility when cross-fitted predictions are first available.
    data["forecast_uncertainty"] = (
        error.rolling(63, min_periods=20)
        .std()
        .fillna(pd.Series(data["vol"]))
        .to_numpy()
    )
    vol = pd.Series(data["vol"])
    data["regime"] = (vol > vol.shift().rolling(126).median()).astype(float).to_numpy()
    records = []
    for reward, risk, turn in [("net", 0.0, 0.0), ("risk_turnover", 1.0, 0.00005)]:
        settings = dict(cfg, portfolio_control=True, lambda_risk=risk, lambda_turn=turn)
        model = train_policy(
            data, 1400, 1652, settings, 100000, True, 42, 50000, output / reward
        )
        for split, start, end in [
            ("validation", 1400, 1652),
            ("synthetic_test", 1652, len(prices)),
        ]:
            for name in ["cash", "ridge", "ppo_ridge"]:
                env = SpreadControlEnv(data, start, end, settings, 100000, alpha=True)
                bars = evaluate(env, name, 42, model if name == "ppo_ridge" else None)
                bars.to_parquet(
                    output / f"{reward}_{split}_{name}.parquet", index=False
                )
                records.append(
                    dict(
                        reward=reward,
                        split=split,
                        policy=name,
                        **stats(bars),
                        hold_frequency=(bars.action_id == 3).mean(),
                        reduce_frequency=(bars.action_id == 4).mean(),
                    )
                )
    pd.DataFrame(records).to_csv(output / "comparison.csv", index=False)
    (output / "settings.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "train_end": 1400,
                "validation_end": 1652,
                "steps": 50000,
                "source_settings": cfg,
                "actions": ["short", "flat", "long", "hold", "halve"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    run(args.output, args.replay)
