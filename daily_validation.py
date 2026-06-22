import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from main_train import main, _json_safe


OUTPUT_DIR = Path("outputs/daily_validation")


VARIANTS = [
    ("ppo_base", []),
    ("ppo_no_wavelet", [(("data", "wavelet", "enabled"), False)]),
    ("ppo_no_cointegration_filter", [(("pairs", "use_cointegration"), False)]),
    ("strict_inverse_corr_0_8", [(("pairs", "correlation_threshold"), 0.8)]),
]


BASELINE_NAMES = {
    "cash": "cash_no_trade",
    "random": "random_policy",
    "zscore_mean_reversion": "zscore_mean_reversion",
}


def _set_nested(cfg, path, value):
    cur = cfg
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def _variant_config(base, name, updates):
    cfg = copy.deepcopy(base)
    for path, value in updates:
        _set_nested(cfg, path, value)

    run_dir = OUTPUT_DIR / name
    cfg.setdefault("output", {})["run_dir"] = str(run_dir)
    cfg.setdefault("training", {})["save_dir"] = str(run_dir / "models" / "ppo")
    cfg.setdefault("logging", {})["log_path"] = str(run_dir / "logs")
    return cfg


def _write_config(cfg, name):
    cfg_dir = OUTPUT_DIR / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(path)


def _read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _metric(metrics, key, default=np.nan):
    value = metrics.get(key, default) if metrics else default
    if isinstance(value, (float, int)) and not math.isfinite(float(value)):
        return np.nan
    return value


def _friction_stats(frictions):
    if frictions is None or len(frictions) == 0:
        return {
            "turnover": 0.0,
            "transaction_costs": 0.0,
            "carry_costs": 0.0,
            "total_costs": 0.0,
            "borrow_financing_costs": 0.0,
            "total_trades": 0,
            "average_holding_period": np.nan,
        }

    df = frictions.copy()
    turnover = float(df.get("turnover", pd.Series(dtype=float)).sum())
    transaction_costs = float(df.get("transaction_costs", pd.Series(dtype=float)).sum())
    carry_costs = float(df.get("carry_costs", pd.Series(dtype=float)).sum())
    borrow = float(df.get("borrow_cost", pd.Series(dtype=float)).sum())
    financing = float(df.get("financing_cost", pd.Series(dtype=float)).sum())
    trade_events = int((df.get("turnover", pd.Series(dtype=float)).abs() > 1e-9).sum())
    active_days = int((df.get("active_positions", pd.Series(dtype=float)) > 0).sum())
    avg_holding = active_days / trade_events if trade_events else np.nan

    return {
        "turnover": turnover,
        "transaction_costs": transaction_costs,
        "carry_costs": carry_costs,
        "total_costs": transaction_costs + carry_costs,
        "borrow_financing_costs": borrow + financing,
        "total_trades": trade_events,
        "average_holding_period": avg_holding,
    }


def _summary_row(strategy, status, results=None, frictions=None, error=None):
    metrics = (results or {}).get("metrics", {})
    trade_pnls = (results or {}).get("trade_pnls", [])
    friction = _friction_stats(frictions)
    initial_capital = 100000.0
    avg_trade_pnl = float(np.mean(trade_pnls)) if trade_pnls else np.nan

    return {
        "strategy": strategy,
        "status": status,
        "error": error,
        "window_count": len((results or {}).get("window_metrics", [])),
        "final_value": _metric(metrics, "final_value"),
        "total_return": _metric(metrics, "total_return"),
        "cagr": _metric(metrics, "cagr"),
        "sharpe": _metric(metrics, "sharpe"),
        "sortino": _metric(metrics, "sortino"),
        "max_drawdown": _metric(metrics, "max_drawdown"),
        "calmar": _metric(metrics, "mar"),
        "win_rate": _metric(metrics, "win_rate"),
        "profit_factor": _metric(metrics, "profit_factor"),
        "average_trade_pnl": avg_trade_pnl,
        "average_trade_return": avg_trade_pnl / initial_capital if math.isfinite(avg_trade_pnl) else np.nan,
        **friction,
    }


def _named_result(metrics, equity_curve, trade_pnls, window_metrics=None):
    return {
        "metrics": metrics,
        "equity_curve": equity_curve,
        "trade_pnls": trade_pnls,
        "window_metrics": window_metrics or [],
    }


def _pair_rows(strategy, results, pair_contributions=None):
    rows = []
    pair_contributions = pair_contributions or {}
    for window in results.get("selected_pairs", []):
        for pair_type, key in [("traditional", "traditional_pairs"), ("inverse", "inverse_pairs")]:
            for pair in window.get(key, []):
                pair_name = f"{pair[0]}-{pair[1]}"
                rows.append({
                    "strategy": strategy,
                    "window": window.get("window"),
                    "train_start": window.get("train_start", window.get("start")),
                    "validation_start": window.get("validation_start"),
                    "test_start": window.get("test_start"),
                    "end": window.get("end"),
                    "pair": pair_name,
                    "pair_type": pair_type,
                    "pnl_contribution": pair_contributions.get(pair_name, 0.0),
                })
    return rows


def _save_equity_curves(curves):
    if not curves:
        pd.DataFrame().to_csv(OUTPUT_DIR / "equity_curves.csv", index=False)
        return
    df = pd.concat(curves, axis=1)
    df.index.name = "date"
    df.to_csv(OUTPUT_DIR / "equity_curves.csv")


def _markdown_table(rows):
    if not rows:
        return "No rows."
    df = pd.DataFrame(rows)
    cols = [
        "strategy",
        "status",
        "total_return",
        "cagr",
        "sharpe",
        "max_drawdown",
        "calmar",
        "win_rate",
        "profit_factor",
        "total_trades",
        "turnover",
        "total_costs",
    ]
    df = df[[c for c in cols if c in df.columns]].copy()
    for col in df.columns:
        if col not in {"strategy", "status"}:
            df[col] = df[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    lines = ["| " + " | ".join(df.columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def _write_report(summary_rows, statuses, base_config):
    manifest = _read_json(OUTPUT_DIR / "data_manifest.json")
    z_row = next((r for r in summary_rows if r["strategy"] == "zscore_mean_reversion"), None)
    ppo_row = next((r for r in summary_rows if r["strategy"] == "ppo_base"), None)
    if ppo_row and z_row and ppo_row["status"] == "pass" and z_row["status"] == "pass":
        if ppo_row.get("total_return", np.nan) > z_row.get("total_return", np.nan):
            verdict = "daily benchmark passed mechanically; PPO beat z-score on this cheap run, but this is not final evidence of edge"
        else:
            verdict = "daily benchmark passed mechanically; PPO did not beat z-score after costs on this cheap run"
    else:
        verdict = "daily benchmark inconclusive because one or more required comparison rows failed"

    lines = [
        "# Inverse Pairs Daily Validation Report",
        "",
        "## Protocol",
        "- Data frequency: daily end-of-day bars only.",
        f"- Price type: `{base_config.get('data', {}).get('price_type')}`.",
        "- Signals use one-bar-lagged features in daily mode; trades are modeled on the next available daily close-to-close interval.",
        "- No intraday stops, limit-order fills, minute-level fills, or intraday execution claims are made.",
        "- Pair selection, feature selection/scaling, and PPO training occur inside train windows only.",
        "- Validation windows are available to the locked protocol; test windows are evaluated after training.",
        "- PPO timesteps are intentionally tiny for this cheap daily benchmark and must not be interpreted as tuned performance.",
        "",
        "## Data Snapshot",
        f"- Source: `{manifest.get('source')}`",
        f"- Original source: `{manifest.get('original_source')}`",
        f"- Cache path: `{manifest.get('cache_path')}`",
        f"- Cache SHA-256: `{manifest.get('cache_sha256')}`",
        f"- Download timestamp UTC: `{manifest.get('download_timestamp_utc')}`",
        f"- Symbols requested: {len(manifest.get('symbols_requested', []))}",
        f"- Symbols kept sample: {manifest.get('symbols_kept', [])[:30]}",
        f"- Symbols kept count: {len(manifest.get('symbols_kept', []))}",
        f"- Actual date range: {manifest.get('actual_date_range')}",
        "",
        "## Results",
        _markdown_table(summary_rows),
        "",
        "## Variant Status",
    ]
    for name, status in statuses.items():
        lines.append(f"- `{name}`: {status.get('status')} {status.get('error', '')}")

    lines.extend([
        "",
        "## Artifacts",
        "- `outputs/daily_validation/data_manifest.json`",
        "- `outputs/daily_validation/config_snapshot.json`",
        "- `outputs/daily_validation/data_quality_report.md`",
        "- `outputs/daily_validation/summary.csv`",
        "- `outputs/daily_validation/summary.json`",
        "- `outputs/daily_validation/equity_curves.csv`",
        "- `outputs/daily_validation/trades.csv`",
        "- `outputs/daily_validation/pair_results.csv`",
        "",
        "## Limitations",
        "- This run is a cheap daily-bar validation benchmark, not an expensive final evidence run.",
        "- `trades.csv` is a bar-level turnover/cost event log, not a closed-position trade ledger.",
        "- Average holding period is a turnover-event proxy derived from active-position days divided by turnover events.",
        "- Adjusted close execution is a total-return modeling convention and is not an intraday fill model.",
        "",
        "## Verdict",
        verdict,
    ])

    with open(OUTPUT_DIR / "inverse_pairs_daily_validation_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def run_daily_validation(config_path="config/daily_validation_config.yaml"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r") as f:
        base = yaml.safe_load(f)

    config_snapshot = {"base_config_path": config_path, "variants": {}}
    statuses = {}
    summary_rows = []
    curves = {}
    trade_tables = []
    pair_rows = []
    baseline_harvested = False

    for name, updates in VARIANTS:
        cfg = _variant_config(base, name, updates)
        variant_path = _write_config(cfg, name)
        config_snapshot["variants"][name] = cfg

        try:
            results = main(config_path=variant_path, smoke=False)
            statuses[name] = {"status": "pass", "config_path": variant_path}

            frictions = results.get("frictions", pd.DataFrame())
            summary_rows.append(_summary_row(name, "pass", results, frictions))
            curves[name] = results["equity_curve"].rename(name)
            if not frictions.empty:
                trade_tables.append(frictions.assign(strategy=name))
            pair_rows.extend(_pair_rows(name, results, results.get("pair_contributions")))

            if not baseline_harvested:
                for raw_name, out_name in BASELINE_NAMES.items():
                    metrics = results.get("baseline_global_metrics", {}).get(raw_name)
                    curve = results.get("baseline_equity_curves", {}).get(raw_name)
                    fr = results.get("baseline_frictions", {}).get(raw_name, pd.DataFrame())
                    if metrics is None or curve is None:
                        continue
                    named = _named_result(
                        metrics=metrics,
                        equity_curve=curve,
                        trade_pnls=results.get("baseline_trade_pnls", {}).get(raw_name, []),
                        window_metrics=results.get("window_metrics", []),
                    )
                    summary_rows.append(_summary_row(out_name, "pass", named, fr))
                    curves[out_name] = curve.rename(out_name)
                    if not fr.empty:
                        trade_tables.append(fr.assign(strategy=out_name))
                    pair_rows.extend(_pair_rows(
                        out_name,
                        results,
                        results.get("baseline_pair_contributions", {}).get(raw_name, {}),
                    ))
                baseline_harvested = True
        except Exception as exc:
            statuses[name] = {
                "status": "fail",
                "config_path": variant_path,
                "error": str(exc),
            }
            summary_rows.append(_summary_row(name, "fail", error=str(exc)))

    with open(OUTPUT_DIR / "config_snapshot.json", "w") as f:
        json.dump(_json_safe(config_snapshot), f, indent=2)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(_json_safe({"statuses": statuses, "summary": summary_rows}), f, indent=2)

    _save_equity_curves(curves)

    if trade_tables:
        pd.concat(trade_tables, ignore_index=True).to_csv(OUTPUT_DIR / "trades.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / "trades.csv", index=False)

    pd.DataFrame(pair_rows).to_csv(OUTPUT_DIR / "pair_results.csv", index=False)
    _write_report(summary_rows, statuses, base)

    base_status = statuses.get("ppo_base", {}).get("status")
    if base_status != "pass":
        raise RuntimeError(
            "Daily validation did not complete the required ppo_base run. "
            f"Status: {statuses.get('ppo_base')}"
        )

    return {
        "statuses": statuses,
        "summary": summary_rows,
        "report_path": str(OUTPUT_DIR / "inverse_pairs_daily_validation_report.md"),
    }


if __name__ == "__main__":
    run_daily_validation()
