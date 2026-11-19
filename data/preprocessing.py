import pandas as pd
import numpy as np
import json
import os

from data.loaders import DataLoader
from data.loaders import save_data_quality_report
from data.returns import compute_log_returns

from utils.logger import get_logger
logger = get_logger("Preprocessing")


def _save_json(obj, path):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _markdown_table(df):
    if df.empty:
        return ""
    cols = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        values = [str(row[col]) for col in cols]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _save_data_quality_markdown(report, manifest, path):
    if not path:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [
        "# Daily Data Quality Report",
        "",
        f"- Source: `{manifest.get('source')}`",
        f"- Requested symbols: {len(manifest.get('symbols_requested', []))}",
        f"- Kept symbols: {len(manifest.get('symbols_kept', []))}",
        f"- Requested range: {manifest.get('requested_date_range', {})}",
        f"- Actual range: {manifest.get('actual_date_range', {})}",
        f"- Cache path: `{manifest.get('cache_path')}`",
        f"- Cache SHA-256: `{manifest.get('cache_sha256')}`",
        "",
        "## Missingness By Field",
        "",
    ]

    if report is None or report.empty:
        lines.append("No quality report rows were generated.")
    else:
        summary = (
            report.groupby("field", dropna=False)
            .agg(
                tickers=("ticker", "count"),
                avg_missing_pct=("missing_pct", "mean"),
                max_missing_pct=("missing_pct", "max"),
                kept=("kept", "sum"),
            )
            .reset_index()
        )
        lines.append(_markdown_table(summary))
        lines.extend(["", "## Per-Symbol Missingness", ""])
        cols = [
            "field",
            "ticker",
            "total_rows",
            "raw_non_null",
            "missing_count",
            "missing_pct",
            "first_valid",
            "last_valid",
            "kept",
        ]
        lines.append(_markdown_table(report[cols]))

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ======================================================================
#                    STANDARDIZED PREPROCESSING PIPELINE
# ======================================================================
def prepare_data(config):
    """
    FINAL VERSION:
    - Loads OHLCV using DataLoader (NO preprocessing performed there)
    - Computes log returns
    - Aligns OHLCV + cleaned returns
    - Returns:
        ohlcv: dict of DataFrames
        cleaned_returns: aligned log returns
        idx: aligned DatetimeIndex

    No pair selection.
    No feature computation.
    """

    logger.info("\n[DATA] ========= PREPARING FULL DATA PIPELINE =========")

    data_cfg = config["data"]

    if data_cfg.get("require_full_coverage", False):
        raise ValueError("Walk-forward selection requires fold-local coverage. Set require_full_coverage to false.")

    ohlcv = DataLoader.load_prices(
        assets_path=data_cfg["assets_path"],
        start=data_cfg["start_date"],
        end=data_cfg["end_date"],
        min_history=data_cfg["min_history"],
        source=data_cfg.get("source", "yfinance"),
        local_path=data_cfg.get("local_path"),
        cache_path=data_cfg.get("cache_path"),
        max_ffill_gap=data_cfg.get("max_ffill_gap"),
        min_retained_symbols=data_cfg.get("min_retained_symbols"),
        preserve_calendar=True,
        coverage_tolerance_days=data_cfg.get("coverage_tolerance_days", 0),
        download_threads=data_cfg.get("download_threads", True),
        download_chunk_size=data_cfg.get("download_chunk_size"),
        synthetic_periods=data_cfg.get("synthetic_periods", 340),
        seed=config.get("system", {}).get("seed", 42),
    )

    report_path = data_cfg.get("quality_report_path", "outputs/audit/data_quality_report.csv")
    save_data_quality_report(DataLoader.last_quality_report, report_path)
    _save_json(DataLoader.last_manifest, data_cfg.get("manifest_path"))
    _save_data_quality_markdown(
        DataLoader.last_quality_report,
        DataLoader.last_manifest,
        data_cfg.get("quality_report_md_path"),
    )

    price_type = data_cfg.get("price_type", "Adj Close")
    close = ohlcv[price_type]

    logger.info("[DATA] OHLCV loaded. Computing returns...")

    # ----------------------------------------------------------
    # 2) Compute log returns
    # ----------------------------------------------------------
    log_ret = compute_log_returns(close, keep_calendar=True)

    cleaned_returns = log_ret

    # ----------------------------------------------------------
    # 5) Align OHLCV with cleaned returns
    # ----------------------------------------------------------
    idx = close.index.intersection(cleaned_returns.index).sort_values()

    for key in ohlcv:
        ohlcv[key] = ohlcv[key].loc[idx]

    cleaned_returns = cleaned_returns.loc[idx]

    logger.info(
        f"[DATA] Final aligned shapes: OHLCV={ohlcv['Close'].shape}, "
        f"RETURNS={cleaned_returns.shape}"
    )
    logger.info(f"[DATA] Total timesteps: {len(idx)}")
    logger.info("[DATA] ========= PREPROCESSING COMPLETE =========\n")

    return ohlcv, cleaned_returns, idx
