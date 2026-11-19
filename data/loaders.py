import hashlib
import os
from datetime import datetime, timezone
import yaml
import numpy as np
import pandas as pd
import yfinance as yf

from utils.logger import get_logger

logger = get_logger("DataLoader")


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Replace non-finite values without filling missing prices."""
    return df.replace([np.inf, -np.inf], np.nan).copy()


def clean_df(df: pd.DataFrame, max_ffill_gap=None) -> pd.DataFrame:
    """
    Causal cleaner for already-filtered time series.

    It never backfills. Forward fill is allowed only from past observations,
    optionally bounded by max_ffill_gap.
    """
    out = sanitize_df(df)
    out = out.ffill(limit=max_ffill_gap)
    return out.copy()


def build_data_quality_report(panels, kept_tickers=None):
    rows = []
    kept_tickers = set(kept_tickers or [])

    for field, df in panels.items():
        total = len(df)
        for ticker in df.columns:
            series = df[ticker]
            valid = series.dropna()
            rows.append(
                {
                    "field": field,
                    "ticker": ticker,
                    "total_rows": total,
                    "raw_non_null": int(series.notna().sum()),
                    "missing_count": int(series.isna().sum()),
                    "missing_pct": float(series.isna().mean()) if total else 0.0,
                    "first_valid": valid.index[0].isoformat() if len(valid) else "",
                    "last_valid": valid.index[-1].isoformat() if len(valid) else "",
                    "kept": ticker in kept_tickers if kept_tickers else False,
                }
            )

    return pd.DataFrame(rows)


def save_data_quality_report(report: pd.DataFrame, path: str):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    report.to_csv(path, index=False)


class DataLoader:
    last_quality_report = pd.DataFrame()
    last_manifest = {}

    @staticmethod
    def load_universe(path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"[DataLoader] Universe YAML empty: {path}")

        tickers = set()
        base_dir = os.path.dirname(path)
        include_files = data.get("include_files", [])
        if isinstance(include_files, str):
            include_files = [include_files]

        for include_path in include_files:
            resolved = include_path
            if not os.path.isabs(resolved):
                resolved = os.path.normpath(os.path.join(base_dir, include_path))
                if not os.path.exists(resolved):
                    resolved = os.path.normpath(include_path)
            tickers.update(DataLoader.load_universe(resolved))

        for _, groups in data.items():
            if _ == "include_files":
                continue
            for group in groups:
                if isinstance(group, list):
                    tickers.update(x.strip() for x in group)
                elif isinstance(group, str):
                    tickers.update(x.strip() for x in group.split("-"))
                else:
                    raise TypeError(f"[Universe] Invalid type: {type(group)}")

        tickers = sorted(list(tickers))
        logger.info(f"[DataLoader] Universe tickers loaded: {len(tickers)}")
        return tickers

    @staticmethod
    def _synthetic_ohlcv(tickers, start, periods=340, seed=42):
        idx = pd.bdate_range(start=start, periods=periods)
        rng = np.random.default_rng(seed)

        base = 100 + np.cumsum(rng.normal(0.02, 0.6, size=periods))
        close = pd.DataFrame(index=idx)

        for i, ticker in enumerate(tickers):
            if i == 0:
                values = base
            elif i == 1:
                values = 240 - base + rng.normal(0, 0.15, size=periods)
            else:
                values = 50 + np.cumsum(rng.normal(0.01 * (i + 1), 0.45, size=periods))
            close[ticker] = np.maximum(values, 5.0)

        panels = {
            "Close": close,
            "Adj Close": close.copy(),
            "Open": close * (1 + rng.normal(0, 0.001, size=close.shape)),
            "High": close * (1 + np.abs(rng.normal(0.002, 0.001, size=close.shape))),
            "Low": close * (1 - np.abs(rng.normal(0.002, 0.001, size=close.shape))),
            "Volume": pd.DataFrame(
                rng.integers(500_000, 2_000_000, size=close.shape),
                index=idx,
                columns=close.columns,
            ),
        }
        return panels

    @staticmethod
    def _extract_yfinance_panels(raw):
        return {
            "Open": sanitize_df(raw["Open"]),
            "High": sanitize_df(raw["High"]),
            "Low": sanitize_df(raw["Low"]),
            "Close": sanitize_df(raw["Close"]),
            "Adj Close": sanitize_df(raw["Adj Close"]),
            "Volume": sanitize_df(raw["Volume"]),
        }

    @staticmethod
    def _download_yfinance_panels(tickers, start, end, threads=True, chunk_size=None):
        chunks = [tickers]
        if chunk_size:
            chunks = [
                tickers[i:i + int(chunk_size)]
                for i in range(0, len(tickers), int(chunk_size))
            ]

        merged = None
        for i, chunk in enumerate(chunks, start=1):
            logger.info(
                f"[DataLoader] yfinance chunk {i}/{len(chunks)}: "
                f"{len(chunk)} tickers"
            )
            raw = yf.download(
                chunk,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=threads,
            )
            if raw.empty:
                logger.warning(f"[DataLoader] Empty yfinance chunk {i}; continuing")
                continue

            panels = DataLoader._extract_yfinance_panels(raw)
            if merged is None:
                merged = panels
            else:
                for field in merged:
                    merged[field] = pd.concat([merged[field], panels[field]], axis=1)

        if merged is None:
            raise RuntimeError("[DataLoader] No OHLCV data returned by yfinance.")

        return {
            field: panel.loc[:, ~panel.columns.duplicated()].sort_index()
            for field, panel in merged.items()
        }

    @staticmethod
    def _sha256(path):
        if not path or not os.path.exists(path):
            return None

        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _save_csv_panels(panels, path):
        if not path:
            return

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        flat = pd.DataFrame(index=panels["Close"].index)
        for field in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            prefix = field.replace(" ", "_") + "_"
            for ticker in panels[field].columns:
                flat[f"{prefix}{ticker}"] = panels[field][ticker]

        flat.index.name = "Date"
        flat.to_csv(path)

    @staticmethod
    def _build_manifest(
        panels,
        source,
        requested_symbols,
        requested_start,
        requested_end,
        cache_path=None,
        download_timestamp=None,
        original_source=None,
    ):
        quality = DataLoader.last_quality_report
        missingness = {}
        if not quality.empty:
            for row in quality.to_dict(orient="records"):
                missingness.setdefault(row["ticker"], {})[row["field"]] = {
                    "missing_count": int(row["missing_count"]),
                    "missing_pct": float(row["missing_pct"]),
                    "raw_non_null": int(row["raw_non_null"]),
                    "first_valid": row["first_valid"],
                    "last_valid": row["last_valid"],
                    "kept": bool(row["kept"]),
                }

        row_counts = {
            field: {
                ticker: int(df[ticker].notna().sum())
                for ticker in df.columns
            }
            for field, df in panels.items()
        }

        index = panels["Close"].index
        return {
            "source": source,
            "original_source": original_source,
            "symbols_requested": sorted(requested_symbols),
            "symbols_kept": sorted(panels["Close"].columns.tolist()),
            "requested_date_range": {
                "start": requested_start,
                "end": requested_end,
            },
            "actual_date_range": {
                "start": index[0].isoformat() if len(index) else None,
                "end": index[-1].isoformat() if len(index) else None,
            },
            "row_counts": row_counts,
            "missingness": missingness,
            "cache_path": cache_path,
            "cache_sha256": DataLoader._sha256(cache_path),
            "download_timestamp_utc": download_timestamp,
            "manifest_created_utc": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _finalize_panels(
        panels,
        min_history,
        max_ffill_gap=None,
        min_retained_symbols=None,
        require_full_coverage=False,
        coverage_start=None,
        coverage_end=None,
        coverage_tolerance_days=0,
        preserve_calendar=False,
    ):
        if preserve_calendar:
            # Do not let future raw observation counts determine fold eligibility.
            common = sorted(set.intersection(*(set(df.columns) for df in panels.values())))
            if len(common) < (min_retained_symbols or 1):
                raise RuntimeError("Insufficient symbols in the requested snapshot")
            DataLoader.last_quality_report = build_data_quality_report(panels, kept_tickers=common)
            return {key: df[common].copy() for key, df in panels.items()}

        def filter_min(df):
            return df.loc[:, df.count() >= min_history]

        filtered = {key: filter_min(df) for key, df in panels.items()}

        common = set(filtered["Open"].columns)
        for key in filtered:
            common &= set(filtered[key].columns)

        common = sorted(list(common))
        if not common:
            raise RuntimeError("[DataLoader] No commonly-available tickers after filtering.")

        if require_full_coverage:
            start_bound = pd.Timestamp(coverage_start) + pd.Timedelta(days=int(coverage_tolerance_days))
            end_bound = pd.Timestamp(coverage_end) - pd.Timedelta(days=int(coverage_tolerance_days))
            coverage_keep = []
            for ticker in common:
                valid = None
                for key in filtered:
                    mask = filtered[key][ticker].notna()
                    valid = mask if valid is None else (valid & mask)
                if not valid.any():
                    continue
                first_valid = valid[valid].index[0]
                last_valid = valid[valid].index[-1]
                if first_valid <= start_bound and last_valid >= end_bound:
                    coverage_keep.append(ticker)

            common = sorted(coverage_keep)
            if not common:
                raise RuntimeError("[DataLoader] No tickers meet full-period coverage filters.")

        if min_retained_symbols is not None and len(common) < int(min_retained_symbols):
            raise RuntimeError(
                "[DataLoader] Insufficient retained symbols after history/OHLCV filtering: "
                f"{len(common)} retained, required {int(min_retained_symbols)}."
            )

        final = {
            key: clean_df(df[common], max_ffill_gap=max_ffill_gap)
            for key, df in filtered.items()
        }

        # A forward fill may bridge an interior gap, never extend a delisted series.
        for key, frame in final.items():
            final[key] = frame.where(filtered[key][common].notna().iloc[::-1].cummax().iloc[::-1])
        valid_mask = None
        for df in final.values():
            mask = df.notna().all(axis=1)
            valid_mask = mask if valid_mask is None else (valid_mask & mask)

        valid_index = final["Close"].index[valid_mask]
        final = {key: df.loc[valid_index].copy() for key, df in final.items()}

        DataLoader.last_quality_report = build_data_quality_report(panels, kept_tickers=common)
        return final

    @staticmethod
    def load_prices(
        assets_path: str,
        start: str,
        end: str,
        min_history: int = 500,
        source: str = "yfinance",
        local_path: str = None,
        cache_path: str = None,
        max_ffill_gap=None,
        min_retained_symbols=None,
        require_full_coverage=False,
        coverage_tolerance_days=0,
        download_threads=True,
        download_chunk_size=None,
        preserve_calendar=False,
        synthetic_periods: int = 340,
        seed: int = 42,
    ):
        tickers = DataLoader.load_universe(assets_path)

        if source == "synthetic":
            panels = DataLoader._synthetic_ohlcv(
                tickers=tickers,
                start=start,
                periods=synthetic_periods,
                seed=seed,
            )
        elif source == "csv":
            if not local_path:
                raise ValueError("[DataLoader] local_path is required for source='csv'")
            panels = DataLoader._load_csv_panels(local_path)
        elif source == "yfinance" and cache_path and os.path.exists(cache_path):
            logger.info(f"[DataLoader] Loading fixed yfinance snapshot cache: {cache_path}")
            panels = DataLoader._load_csv_panels(cache_path)
            cached = set(panels["Close"].columns)
            missing = sorted(set(tickers) - cached)
            if missing:
                raise RuntimeError(
                    "[DataLoader] Snapshot cache is missing requested symbols "
                    f"{missing}. Use a different cache_path for a new fixed snapshot."
                )
            meta_path = f"{cache_path}.meta.json"
            cache_meta = {}
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    import json
                    cache_meta = json.load(f)
            download_timestamp = cache_meta.get("download_timestamp_utc")
            original_source = cache_meta.get("source", "yfinance")
            source = "yfinance_cache"
        elif source == "yfinance":
            logger.info(
                f"[DataLoader] Pulling OHLCV for {len(tickers)} tickers "
                f"from {start} to {end}..."
            )
            panels = DataLoader._download_yfinance_panels(
                tickers=tickers,
                start=start,
                end=end,
                threads=download_threads,
                chunk_size=download_chunk_size,
            )
            download_timestamp = datetime.now(timezone.utc).isoformat()
            if cache_path:
                DataLoader._save_csv_panels(panels, cache_path)
                with open(f"{cache_path}.meta.json", "w") as f:
                    import json
                    json.dump(
                        {
                            "source": "yfinance",
                            "download_timestamp_utc": download_timestamp,
                            "requested_start": start,
                            "requested_end": end,
                            "symbols_requested": tickers,
                        },
                        f,
                        indent=2,
                    )
        else:
            raise ValueError(f"[DataLoader] Unsupported data source: {source}")

        download_timestamp = locals().get("download_timestamp")
        original_source = locals().get("original_source")
        if source != "synthetic":
            if any(not df.index.is_unique for df in panels.values()):
                raise ValueError("Duplicate dates in price snapshot")
            requested = sorted(set(tickers).intersection(panels["Close"].columns))
            panels = {
                key: df.loc[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end)), requested]
                for key, df in panels.items()
            }
        final = DataLoader._finalize_panels(
            panels=panels,
            min_history=min_history,
            preserve_calendar=preserve_calendar,
            max_ffill_gap=max_ffill_gap,
            min_retained_symbols=min_retained_symbols,
            require_full_coverage=require_full_coverage,
            coverage_start=start,
            coverage_end=end,
            coverage_tolerance_days=coverage_tolerance_days,
        )
        DataLoader.last_manifest = DataLoader._build_manifest(
            panels=final,
            source=source,
            requested_symbols=tickers,
            requested_start=start,
            requested_end=end,
            cache_path=cache_path or local_path,
            download_timestamp=download_timestamp,
            original_source=original_source,
        )

        logger.info(f"[DataLoader] Final OHLCV loaded: {final['Close'].shape}")
        return final

    @staticmethod
    def _load_csv_panels(path):
        df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
        panels = {}
        for field in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            prefix = field.replace(" ", "_") + "_"
            cols = [c for c in df.columns if c.startswith(prefix)]
            if not cols:
                raise ValueError(f"[DataLoader] Missing CSV columns for {field}")
            panel = df[cols].copy()
            panel.columns = [c[len(prefix):] for c in cols]
            panels[field] = sanitize_df(panel)
        return panels

    @staticmethod
    def load_single_price(
        ticker: str,
        start: str,
        end: str,
        price_type="Adj Close",
    ):
        try:
            df = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )
        except Exception as e:
            logger.error(f"[DataLoader] download failed for {ticker}: {e}")
            return pd.DataFrame(index=pd.DatetimeIndex([]), columns=[ticker])

        if df.empty:
            logger.error(f"[DataLoader] no data for {ticker}")
            return pd.DataFrame(index=pd.DatetimeIndex([]), columns=[ticker])

        if price_type not in df.columns:
            logger.error(
                f"[DataLoader] '{price_type}' missing for {ticker}. "
                f"Available: {df.columns}"
            )
            return pd.DataFrame(index=pd.DatetimeIndex([]), columns=[ticker])

        series = df[price_type].dropna().copy()

        if series.empty:
            logger.error(f"[DataLoader] '{price_type}' empty for {ticker}")
            return pd.DataFrame(index=pd.DatetimeIndex([]), columns=[ticker])

        series.index = pd.to_datetime(series.index)

        out = pd.DataFrame(series)
        out.columns = [ticker]

        return clean_df(out)
