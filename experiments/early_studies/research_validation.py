import argparse
import copy
import json
import os
from pathlib import Path

import yaml

from main_train import main, _json_safe


REGIMES = {
    "2014_2019": ("2014-01-01", "2019-12-31"),
    "2020": ("2020-01-01", "2020-12-31"),
    "2021_2022": ("2021-01-01", "2022-12-31"),
    "2023_2024": ("2023-01-01", "2024-01-01"),
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

    run_dir = f"outputs/research_validation/{name}"
    cfg["output"]["run_dir"] = run_dir
    cfg["training"]["save_dir"] = f"{run_dir}/models/ppo"
    cfg["data"]["quality_report_path"] = f"{run_dir}/data_quality_report.csv"
    cfg["logging"]["log_path"] = f"{run_dir}/logs"
    return cfg


def _write_config(cfg, name):
    config_dir = Path("outputs/research_validation/configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(path)


def _summarize_results(results):
    window_metrics = results.get("window_metrics", [])
    baseline_metrics = {}
    if window_metrics:
        baseline_metrics = window_metrics[0].get("baseline_metrics", {})

    return {
        "metrics": results["metrics"],
        "selected_pairs": results.get("selected_pairs", []),
        "validation_protocol": results.get("validation_protocol"),
        "window_count": len(window_metrics),
        "first_window_baselines": baseline_metrics,
    }


def _available_regime_notes(cfg):
    source = cfg.get("data", {}).get("source", "yfinance")
    start = cfg.get("data", {}).get("start_date")
    end = cfg.get("data", {}).get("end_date")
    notes = {}
    for regime, (regime_start, regime_end) in REGIMES.items():
        covered = source != "synthetic" and start <= regime_start and end >= regime_end
        notes[regime] = {
            "requested": [regime_start, regime_end],
            "status": "available_for_full_data_run" if covered else "skipped_not_covered_by_local_synthetic_data",
        }
    return notes


def run_suite(config_path, max_variants=None):
    with open(config_path, "r") as f:
        base = yaml.safe_load(f)

    variants = [
        ("base", []),
        ("no_wavelet", [(("data", "wavelet", "enabled"), False)]),
        ("ppo_without_cointegration_filter", [(("pairs", "use_cointegration"), False)]),
        ("strict_inverse_corr_0_8", [(("pairs", "correlation_threshold"), 0.8)]),
    ]

    if max_variants is not None:
        variants = variants[:max_variants]

    summary = {
        "protocol": "synthetic rolling train/validation/test; pair and feature selection fit on train only",
        "variants": {},
        "stress_regimes": _available_regime_notes(base),
    }

    for name, updates in variants:
        cfg = _variant_config(base, name, updates)
        variant_path = _write_config(cfg, name)
        try:
            results = main(config_path=variant_path, smoke=False)
            summary["variants"][name] = {
                "status": "pass",
                **_summarize_results(results),
            }
        except Exception as exc:
            summary["variants"][name] = {
                "status": "fail",
                "error": str(exc),
            }

    os.makedirs("outputs/research_validation", exist_ok=True)
    with open("outputs/research_validation/summary.json", "w") as f:
        json.dump(_json_safe(summary), f, indent=2)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run local research validation suite.")
    parser.add_argument("--config", default="config/validation_smoke_config.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic local validation config.")
    parser.add_argument("--small-wf", action="store_true", help="Run small walk-forward validation suite.")
    parser.add_argument("--max-variants", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_suite(args.config, max_variants=args.max_variants)
