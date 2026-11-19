"""Synthetic learnability gate and chronological PPO position-management comparison."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, coint

from backtest.residual_reports import summarize_study
from backtest.walkforward import WalkForwardEngine
from data.loaders import DataLoader
from data.preprocessing import prepare_data
from env.features.residual import alpha_forecasts, residual_state
from env.spread_control import SpreadControlEnv

POLICIES = ("cash", "random", "zscore", "ridge", "xgboost", "ppo", "ppo_ridge")
FIRST_SIGNAL = 256


def synthetic(seed):
    rng = np.random.default_rng(seed)
    n = 2400
    b = np.log(100) + np.cumsum(rng.normal(0, 0.018, n))
    residual = np.zeros(n)
    for t in range(1, n):
        residual[t] = 0.72 * residual[t - 1] + rng.normal(0, 0.013)
    prices = np.exp(np.column_stack((2 * np.log(100) - b + residual, b)))
    index = pd.bdate_range("2000-01-03", periods=n)
    prices = pd.DataFrame(prices, index=index, columns=["A", "B"])
    volume = pd.DataFrame(
        rng.lognormal(14, 0.2, (n, 2)), index=index, columns=prices.columns
    )
    return prices, volume, prices * volume, residual


def screen(train, dollars, cfg, window, capacity_mode):
    pairs = cfg["pairs"]
    if len(train) < pairs["min_observations"]:
        return pd.DataFrame(), None
    logs = np.log(train)
    corr = train.pct_change(fill_method=None).corr()
    rows = []
    for i, a in enumerate(train):
        for b in train.columns[i + 1 :][
            corr[a].iloc[i + 1 :].to_numpy() <= -pairs["correlation_threshold"]
        ]:
            x, y = logs[b].to_numpy(), logs[a].to_numpy()
            h, intercept = np.polyfit(x, y, 1)
            spread = y - h * x - intercept
            phi = np.polyfit(spread[:-1], spread[1:], 1)[0]
            half = -np.log(2) / np.log(phi) if 0 < phi < 1 else np.inf
            stable = all(
                np.cov(x[idx], y[idx])[0, 1] < 0
                for idx in np.array_split(np.arange(len(x)), 3)
            )
            adv = dollars[[a, b]].median().to_numpy()
            weights = np.abs([1.0, h]) / (1 + abs(h))
            capacity = (
                np.min(adv / np.maximum(weights, 1e-12))
                * cfg["env"]["liquidity"]["max_trade_pct_adv"]
                / cfg["env"]["max_position"]
            )
            rate = cfg["env"]["transaction_cost"] + cfg["env"]["slippage_cost"]
            cost = (
                2 * rate
                + (
                    cfg["env"]["financing_rate_annual"]
                    + cfg["env"]["borrow_fee_annual"]
                )
                * half
                / 252
            )
            rows.append(
                {
                    "window": window,
                    "a": a,
                    "b": b,
                    "rho": corr.at[a, b],
                    "beta": h,
                    "eg_p": coint(y, x)[1],
                    "adf_p": adfuller(spread)[1],
                    "half_life": half,
                    "stable": stable,
                    "capacity": capacity,
                    "displacement": spread.std() / (1 + abs(h)),
                    "estimated_round_trip_cost": cost,
                }
            )
    if not rows:
        return pd.DataFrame(), None
    frame = pd.DataFrame(rows)
    frame["holm_p"] = multipletests(frame.eg_p.fillna(1), method="holm")[1]
    frame["stationary"] = (frame.holm_p < pairs["cointegration_p_value"]) & (
        frame.beta < 0
    )
    frame["half_life_ok"] = frame.half_life.between(1, 63)
    frame["liquid"] = frame.capacity >= cfg["env"]["initial_capital"]
    frame["cost_ok"] = frame.displacement > frame.estimated_round_trip_cost
    criteria = ["stationary", "half_life_ok", "stable", "cost_ok"]
    if capacity_mode == "full-target":
        criteria.append("liquid")
    chosen = frame[frame[criteria].all(axis=1) & frame.capacity.gt(0)]
    chosen = chosen.sort_values(["holm_p", "rho", "a", "b"])
    pair = tuple(chosen.iloc[0][["a", "b"]]) if len(chosen) else None
    return frame, pair


def make_data(prices, volume, dollars, train_end, blocked, truth=None):
    if (
        not np.isfinite(prices).all().all()
        or (prices <= 0).any().any()
        or volume.isna().any().any()
        or dollars.isna().any().any()
    ):
        raise ValueError(
            f"Unavailable selected-pair quote or volume for {prices.columns.tolist()}"
        )
    state, target = residual_state(prices, volume, train_end)
    forecasts = alpha_forecasts(state, target, train_end, FIRST_SIGNAL)
    features = state.to_numpy().copy()
    features[:, 2] *= 100
    if not np.isfinite(features[FIRST_SIGNAL:]).all():
        raise ValueError("Non-finite residual state after warm-up")
    return {
        "prices": prices.to_numpy(),
        "index": prices.index,
        "vol": state.vol.to_numpy(),
        "beta": state.beta.to_numpy(),
        "features": np.clip(features, -10, 10),
        "z": state.z.to_numpy(),
        "alpha": forecasts,
        "target": target,
        "truth": truth,
        "adv": dollars.rolling(20).mean().to_numpy(),
        "dollars": dollars.to_numpy(),
        "blocked": prices.columns.isin(blocked),
        "years": np.diff(prices.index.to_numpy()).astype("timedelta64[D]").astype(int)
        / 365,
    }


def evaluate(env, policy, seed, model=None):
    rng = np.random.default_rng(seed)
    obs, _ = env.reset()
    records = []
    while True:
        t = env.t
        s = t - 1
        ent = np.nan
        if model is not None:
            with torch.no_grad():
                dist = model.policy.get_distribution(
                    model.policy.obs_to_tensor(obs)[0]
                ).distribution
                action, ent = dist.probs.argmax().item(), dist.entropy().item()
        elif policy == "cash":
            action = 1
        elif policy == "random":
            action = rng.integers(3)
        elif policy == "zscore":
            z = env.data["z"][s]
            action = 1 - int(np.sign(z)) if abs(z) >= 1 else 1
        else:
            forecast = env.data["alpha"][s, int(policy == "xgboost")]
            action = (
                1 + int(np.sign(forecast))
                if abs(forecast) > 2 * env.settings["cost_rate"]
                else 1
            )
        obs, _, done, _, info = env.step(action)
        info.update(date=env.data["index"][t + 1], entropy=ent, policy=policy)
        if env.data["truth"] is not None:
            residual = env.data["truth"][s]
            strong = abs(residual) > 0.013 / np.sqrt(1 - 0.72**2)
            info["direction_match"] = (
                float(info["action"] == -np.sign(residual)) if strong else np.nan
            )
        records.append(info)
        if done:
            return pd.DataFrame(records)


def stats(df):
    r = df.net_return
    decisions = df.loc[df.capacity.notna()]
    eq = np.r_[1.0, (1 + r).cumprod().to_numpy()]
    return {
        "net_return": eq[-1] - 1,
        "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0,
        "max_drawdown": (1 - eq / np.maximum.accumulate(eq)).max(),
        "gross_exposure": df.gross.mean(),
        "turnover": df.turnover.sum(),
        "costs": df.costs.sum(),
        "short_freq": (decisions.action == -1).mean(),
        "flat_freq": (decisions.action == 0).mean(),
        "long_freq": (decisions.action == 1).mean(),
        "entropy": df.entropy.mean(),
        "capacity_p10": df.capacity.quantile(0.1),
        "fill_rate": df.filled_notional.sum() / df.requested_notional.sum()
        if df.requested_notional.sum()
        else np.nan,
        "direction_match": df.direction_match.mean()
        if "direction_match" in df
        else np.nan,
    }


def train_policy(data, train_end, val_end, cfg, capital, alpha, seed, steps, path):
    def build():
        return SpreadControlEnv(
            data, FIRST_SIGNAL + 1, train_end, cfg, capital, alpha, training=True
        )

    # Choose rollout lengths dividing the specified budget exactly.
    rollout = 32 if steps == 256 else 125
    if steps % (8 * rollout):
        raise ValueError("Use 256 or a multiple of 1000 steps")
    vec = DummyVecEnv([build for _ in range(8)])
    model = PPO(
        "MlpPolicy",
        vec,
        seed=seed,
        device="cpu",
        n_steps=rollout,
        batch_size=64 if steps == 256 else 100,
        n_epochs=4,
        gamma=0.97,
        ent_coef=0.005,
        policy_kwargs={"net_arch": [32, 32]},
        verbose=0,
    )
    validation = SpreadControlEnv(data, train_end, val_end, cfg, capital, alpha)
    log = []
    best_step = 0
    best = -np.inf
    while model.num_timesteps < steps:
        model.learn(min(10000, steps - model.num_timesteps), reset_num_timesteps=False)
        bars = evaluate(validation, "ppo_ridge" if alpha else "ppo", seed, model)
        result = stats(bars)
        log.append(
            dict(
                steps=model.num_timesteps,
                **result,
                **{
                    k: float(v)
                    for k, v in model.logger.name_to_value.items()
                    if k.startswith("train/")
                },
            )
        )
        if result["net_return"] > best:
            best_step = model.num_timesteps
            best = result["net_return"]
            model.save(str(path))
    vec.close()
    for row in log:
        row["selected_checkpoint"] = row["steps"] == best_step
    pd.DataFrame(log).to_csv(path.with_suffix(".validation.csv"), index=False)
    return PPO.load(str(path), device="cpu")


def run_fold(
    data,
    train_end,
    val_end,
    cfg,
    capitals,
    seed,
    steps,
    root,
    initial_capital,
    optimizer_seed=None,
):
    root.mkdir(parents=True, exist_ok=True)
    evaluated = []
    forecasts = []
    for split, start, end in (
        ("validation", train_end, val_end),
        ("test", val_end, len(data["index"])),
    ):
        sig = slice(start - 1, end - 2)
        labels = data["target"][sig]
        for k, name in enumerate(("ridge", "xgboost")):
            pred = data["alpha"][sig, k]
            forecasts.append(
                pd.DataFrame(
                    {
                        "date": data["index"][start + 1 : end],
                        "split": split,
                        "model": name,
                        "prediction": pred,
                        "realized": labels,
                    }
                )
            )
    pd.concat(forecasts).to_csv(root / "forecasts.csv", index=False)
    for policy in POLICIES:
        alpha = policy == "ppo_ridge"
        model = (
            train_policy(
                data,
                train_end,
                val_end,
                cfg,
                initial_capital,
                alpha,
                seed if optimizer_seed is None else optimizer_seed,
                steps,
                root / policy,
            )
            if policy.startswith("ppo")
            else None
        )
        for split, start, end in (
            ("validation", train_end, val_end),
            ("test", val_end, len(data["index"])),
        ):
            capital = capitals[policy] if split == "test" else initial_capital
            env = SpreadControlEnv(data, start, end, cfg, capital, alpha)
            bars = evaluate(env, policy, seed, model)
            # Include the flat opening mark on the common trading calendar.
            opening = cash_bars(data["index"][start : start + 1], capital, policy)
            bars = pd.concat([opening, bars], ignore_index=True)
            bars["split"] = split
            evaluated.append(bars)
            if split == "test":
                capitals[policy] = bars.equity.iloc[-1]
    return pd.concat(evaluated, ignore_index=True)


def cash_bars(index, capital, policy):
    return pd.DataFrame(
        {
            "date": index,
            "policy": policy,
            "equity": capital,
            "net_return": 0.0,
            "gross_pnl": 0.0,
            "costs": 0.0,
            "turnover": 0.0,
            "gross": 0.0,
            "risk_penalty": 0.0,
            "turnover_penalty": 0.0,
            "action": 0,
            "position": 0,
            "reward": 0.0,
            "fill": 1.0,
            "exit_notional": 0.0,
            "exit_adv_multiple": 0.0,
            "requested_notional": 0.0,
            "filled_notional": 0.0,
            "capacity": np.nan,
            "entropy": np.nan,
        }
    )


def save_results(frames, root):
    bars = pd.concat(frames, ignore_index=True)
    bars.to_csv(root / "bars.csv", index=False)
    summary = []
    for (split, policy), frame in bars.groupby(["split", "policy"]):
        if split == "test":
            summary.append(dict(policy=policy, **stats(frame)))
    summary = pd.DataFrame(summary).set_index("policy")
    summary.to_csv(root / "test_summary.csv")
    summarize_study(root)
    return summary


def replicate_synthetic(root, seed, settings, capital, count):
    """Evaluate frozen policies on independent continuations from the same validation endpoint."""
    train_end, test_start = 1400, 1652
    prices, volumes, _, residual = synthetic(seed)
    n = len(prices) - test_start
    models = {
        name: PPO.load(root / name, device="cpu") for name in ("ppo", "ppo_ridge")
    }
    rows = []
    for k in range(1, count + 1):
        continuation_seed = seed + 1000 + k
        future, future_volume, _, shocks = synthetic(continuation_seed)
        b = np.log(prices.iloc[test_start - 1, 1]) + np.log(
            future.iloc[1 : n + 1, 1].to_numpy() / future.iloc[0, 1]
        )
        spread = (
            0.72 ** np.arange(1, n + 1) * residual[test_start - 1] + shocks[1 : n + 1]
        )
        sample = prices.copy()
        sample.iloc[test_start:] = np.exp(
            np.column_stack((2 * np.log(100) - b + spread, b))
        )
        volume = volumes.copy()
        volume.iloc[test_start:] = future_volume.iloc[1 : n + 1].to_numpy()
        truth = np.r_[residual[:test_start], spread]
        data = make_data(sample, volume, sample * volume, train_end, [], truth)
        for name in POLICIES:
            env = SpreadControlEnv(
                data, test_start, len(sample), settings, capital, name == "ppo_ridge"
            )
            bars = evaluate(env, name, continuation_seed, models.get(name))
            rows.append(
                dict(continuation_seed=continuation_seed, policy=name, **stats(bars))
            )
    pd.DataFrame(rows).to_csv(root / "independent_holdouts.csv", index=False)


def main(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    if not args.verify_holdouts and any(root.iterdir()):
        raise FileExistsError(
            f"Use a new output directory to retain earlier checkpoints: {root}"
        )
    config = yaml.safe_load(args.config.read_text())
    e = config["env"]
    if e.get("signal_lag_bars", 1) != 1 or not e["liquidity"]["enabled"]:
        raise ValueError(
            "The residual study requires a one-bar signal lag and enabled ADV limits"
        )
    settings = {
        "gross": min(e["max_position"], e["max_total_exposure"]),
        "cost_rate": e["transaction_cost"] + e["slippage_cost"],
        "funding": e["financing_rate_annual"],
        "borrow": e["borrow_fee_annual"],
        "cash": e["cash_rate_annual"],
        "adv_fraction": e["liquidity"]["max_trade_pct_adv"],
        "lambda_risk": 1.0,
        "lambda_turn": 0.00005,
    }
    if args.strict_execution:
        settings.update(strict_volume=True, unwind_sessions=20)
    if settings["cash"] != 0:
        raise ValueError(
            "This comparison uses zero-interest cash during ineligible folds"
        )
    seed = config["system"]["seed"]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.verify_holdouts:
        manifest = json.loads((root / "manifest.json").read_text())
        if settings != manifest["settings"] or manifest["seed"] != seed:
            raise ValueError("Checkpoint settings and seed must match the original run")
        replicate_synthetic(
            root / "synthetic",
            seed,
            settings,
            e["initial_capital"],
            args.verify_holdouts,
        )
        return
    capitals = dict.fromkeys(POLICIES, e["initial_capital"])
    meta = {
        "capacity_mode": args.capacity_mode,
        "seed": seed,
        "optimizer_seed": args.optimizer_seed
        if args.optimizer_seed is not None
        else seed,
        "steps_per_ppo_per_eligible_fold": args.steps,
        "settings": settings,
        "available_timesteps": [256, 50000, 250000, 1000000],
        "checkpoint": "highest validation net return, earliest tie",
        "config": config,
        "packages": {
            p: version(p)
            for p in (
                "numpy",
                "pandas",
                "torch",
                "stable-baselines3",
                "xgboost",
                "scikit-learn",
                "statsmodels",
            )
        },
    }
    meta["source_sha256"] = {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in (
            "experiments/portfolio_control/residual_control_study.py",
            "env/spread_control.py",
            "env/features/residual.py",
            "backtest/residual_reports.py",
        )
    }
    prices, volume, dollars, truth = synthetic(seed)
    syn = root / "synthetic"
    syn.mkdir(exist_ok=True)
    data = make_data(prices, volume, dollars, 1400, [], truth)
    bars = run_fold(
        data,
        1400,
        1652,
        settings,
        capitals.copy(),
        seed,
        args.steps,
        syn,
        e["initial_capital"],
        args.optimizer_seed,
    )
    summary = save_results([bars], syn)
    passed = all(
        summary.at[p, "net_return"] > max(0, summary.at["random", "net_return"])
        and summary.at[p, "direction_match"] >= 0.60
        for p in ("ppo", "ppo_ridge")
    )
    meta["synthetic_gate_passed"] = bool(passed)
    (root / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(summary.to_string(), flush=True)
    if args.synthetic_only or not passed:
        return
    config["data"]["quality_report_path"] = str(root / "market_data_quality.csv")
    panels, returns, index = prepare_data(config)
    meta["snapshot_sha256"] = DataLoader.last_manifest["cache_sha256"]
    engine = WalkForwardEngine(panels, returns, config, None, index)
    screens, folds, results = [], [], []
    for n, window in enumerate(engine._generate_validation_windows(), 1):
        train, val, test = window[:3]
        px = train[config["data"]["price_type"]]
        eligible = px.gt(0).all()
        for raw in train.values():
            eligible &= raw.notna().all()
        for raw in panels.values():
            eligible &= raw.loc[: px.index[-1]].count() >= config["data"]["min_history"]
        px = px.loc[:, eligible]
        frame, pair = screen(
            px, train["Close"] * train["Volume"], config, n, args.capacity_mode
        )
        screens.append(frame)
        folds.append(
            {
                "window": n,
                "train_start": px.index[0],
                "train_end": px.index[-1],
                "validation_start": val["Close"].index[0],
                "test_start": test["Close"].index[0],
                "test_end": test["Close"].index[-1],
                "assets": px.shape[1],
                "negative_corr": len(frame),
                "stationary": int(frame.stationary.sum()) if len(frame) else 0,
                "selected": None if pair is None else "/".join(pair),
            }
        )
        print("window", n, "selected", pair, flush=True)
        if pair is None:
            rows = []
            for split, slice_ in (("test", test), ("validation", val)):
                for policy in POLICIES:
                    bar = cash_bars(
                        slice_["Close"].index,
                        capitals[policy] if split == "test" else e["initial_capital"],
                        policy,
                    )
                    rows.append(bar.assign(split=split))
            bars = pd.concat(rows, ignore_index=True)
        else:
            cols = list(pair)
            sl = slice(px.index[0], test["Close"].index[-1])
            prices = panels[config["data"]["price_type"]].loc[sl, cols]
            volume = panels["Volume"].loc[sl, cols]
            data = make_data(
                prices,
                volume,
                panels["Close"].loc[sl, cols] * volume,
                len(px),
                e.get("unborrowable_symbols", []),
            )
            bars = run_fold(
                data,
                len(px),
                len(px) + len(val["Close"]),
                settings,
                capitals,
                seed,
                args.steps,
                root / f"window_{n:02d}",
                e["initial_capital"],
                args.optimizer_seed,
            )
        results.append(bars.assign(window=n))
    pd.concat(screens, ignore_index=True).to_csv(root / "screen.csv", index=False)
    pd.DataFrame(folds).to_csv(root / "folds.csv", index=False)
    print(save_results(results, root).to_string(), flush=True)
    (root / "manifest.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("outputs/runs/audit_ppo/input.yaml")
    )
    parser.add_argument(
        "--steps", type=int, choices=[256, 50000, 250000, 1000000], default=50000
    )
    parser.add_argument(
        "--verify-holdouts",
        type=int,
        default=0,
        help="Replay saved synthetic policies on independent future paths",
    )
    parser.add_argument(
        "--capacity-mode",
        choices=["full-target", "partial-fill"],
        default="full-target",
    )
    parser.add_argument(
        "--strict-execution",
        action="store_true",
        help="Cap all fills by realized volume and reserve the final 20 sessions for liquidation",
    )
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument(
        "--optimizer-seed",
        type=int,
        help="Seed PPO and episode sampling independently of the market and baseline seeds",
    )
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
