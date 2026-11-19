"""Restore and verify the market files used by a saved relative-value study."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path, PureWindowsPath

import requests


def restore(manifest, destination):
    metadata = json.loads(manifest.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    for original, digest in metadata["files"].items():
        name = PureWindowsPath(original).name
        if name in {
            "prices.parquet",
            "membership_intervals.parquet",
            "no_trade_days.csv",
        }:
            url = (
                "https://github.com/Johnbrick123/sp500-data/releases/download/data/"
                + name
            )
        elif name.endswith(".parquet"):
            revision = metadata["minute_revision"]
            url = (
                f"https://huggingface.co/datasets/syncster-data/historical-market-data/resolve/{revision}"
                f"/data/stocks/1m/{name[0]}/{name}"
            )
        else:
            continue
        target = destination / name
        if target.exists():
            with target.open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != digest:
                    raise ValueError(
                        f"Existing file differs from the frozen snapshot: {target}"
                    )
            continue
        checksum = hashlib.sha256()
        with tempfile.NamedTemporaryFile(dir=destination, delete=False) as tmp:
            part = Path(tmp.name)
        try:
            with (
                requests.get(url, stream=True, timeout=(15, 90)) as response,
                part.open("wb") as writer,
            ):
                response.raise_for_status()
                for chunk in response.iter_content(1024 * 1024):
                    checksum.update(chunk)
                    writer.write(chunk)
            if checksum.hexdigest() != digest:
                raise ValueError(
                    f"Upstream {name} has changed. Restore the frozen cache to reproduce this run."
                )
            part.replace(target)
        finally:
            part.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/runs/rv_matched_minute/manifest.json"),
    )
    parser.add_argument(
        "--cache", type=Path, default=Path("outputs/runs/cache/rv_sp500")
    )
    args = parser.parse_args()
    restore(args.manifest, args.cache)
