import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from uuid import uuid4

from data.loaders import clean_df, DataLoader
from data.wavelet import causal_wavelet_denoise


def _panels_with_late_start():
    dates = pd.bdate_range("2020-01-01", periods=4)
    base = pd.DataFrame(
        {
            "A": [np.nan, 10.0, 11.0, 12.0],
            "B": [20.0, 21.0, 22.0, 23.0],
        },
        index=dates,
    )
    return {
        "Open": base.copy(),
        "High": base.copy(),
        "Low": base.copy(),
        "Close": base.copy(),
        "Adj Close": base.copy(),
        "Volume": base.copy(),
    }


def test_clean_df_never_backfills():
    dates = pd.bdate_range("2020-01-01", periods=3)
    df = pd.DataFrame({"A": [np.nan, 1.0, 2.0]}, index=dates)

    cleaned = clean_df(df)

    assert pd.isna(cleaned.iloc[0, 0])
    assert cleaned.iloc[1, 0] == 1.0


def test_loader_finalize_drops_unfilled_leading_rows_instead_of_bfill():
    final = DataLoader._finalize_panels(_panels_with_late_start(), min_history=3)

    first_idx = final["Close"].index[0]
    assert first_idx == pd.Timestamp("2020-01-02")
    assert final["Close"].iloc[0]["A"] == 10.0
    assert not final["Close"].isna().any().any()


def test_yfinance_snapshot_cache_is_used_without_download(monkeypatch):
    panels = DataLoader._synthetic_ohlcv(["A", "B"], start="2020-01-01", periods=5)
    test_dir = Path("outputs/test_tmp") / f"cache_{uuid4().hex}"
    test_dir.mkdir(parents=True, exist_ok=True)
    cache_path = test_dir / "snapshot.csv"
    DataLoader._save_csv_panels(panels, str(cache_path))

    assets_path = test_dir / "assets.yaml"
    with open(assets_path, "w") as f:
        yaml.safe_dump({"assets": [["A", "B"]]}, f)

    def _fail_download(*args, **kwargs):
        raise AssertionError("download should not be called when cache exists")

    monkeypatch.setattr("data.loaders.yf.download", _fail_download)

    loaded = DataLoader.load_prices(
        assets_path=str(assets_path),
        start="2020-01-01",
        end="2020-01-10",
        min_history=3,
        source="yfinance",
        cache_path=str(cache_path),
    )

    assert loaded["Close"].shape[1] == 2
    assert DataLoader.last_manifest["source"] == "yfinance_cache"
    assert DataLoader.last_manifest["cache_sha256"]


def test_universe_include_files_and_min_retained_guard():
    test_dir = Path("outputs/test_tmp") / f"universe_{uuid4().hex}"
    test_dir.mkdir(parents=True, exist_ok=True)
    base_path = test_dir / "base.yaml"
    include_path = test_dir / "include.yaml"
    with open(include_path, "w") as f:
        yaml.safe_dump({"included": [["A", "B"]]}, f)
    with open(base_path, "w") as f:
        yaml.safe_dump({"include_files": [str(include_path)], "local": [["C"]]}, f)

    assert DataLoader.load_universe(str(base_path)) == ["A", "B", "C"]

    panels = DataLoader._synthetic_ohlcv(["A", "B"], start="2020-01-01", periods=5)
    try:
        DataLoader._finalize_panels(
            panels,
            min_history=3,
            min_retained_symbols=3,
        )
    except RuntimeError as exc:
        assert "Insufficient retained symbols" in str(exc)
    else:
        raise AssertionError("expected min_retained_symbols guard to fail")


def test_causal_wavelet_output_does_not_change_when_future_changes():
    idx = pd.bdate_range("2020-01-01", periods=140)
    series = pd.Series(np.sin(np.arange(140) / 7.0), index=idx)

    baseline = causal_wavelet_denoise(series, window=32, level=2)
    changed = series.copy()
    changed.iloc[100:] += 100.0
    revised = causal_wavelet_denoise(changed, window=32, level=2)

    pd.testing.assert_series_equal(baseline.iloc[:100], revised.iloc[:100])
