"""
Reproducible ingestion of historical XAUUSD intraday OHLCV datasets for the
canonical 1m and 5m timeframes from a MIT-licensed public repository source.

Mirrors the strict protocol of the canonical 15m ingestion module
(`src/data/ingest_xauusd_15m.py`):
- strict interval grid verification,
- timestamp monotonicity/uniqueness and UTC-awareness,
- finite OHLC values with explicit candle geometry invariants,
- non-negative finite volume handling,
- gap diagnostics (weekend/session/holiday/unexplained),
- SHA-256 provenance manifest with raw and output hashes,
- atomic crash-safe multi-file commit protocol (CSV, manifest, commit marker).

No proxy prices, resampled substitutes, or synthetic values are used: the
ingested bars are exactly the bars published by the cited source, and every
artifact is anchored to its source via cryptographic provenance.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import urllib.request
import uuid

import pandas as pd

from src.data.ingest_xauusd_15m import (
    _validate_numeric_series,
    verify_dataset_and_manifest_integrity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Same MIT-licensed source organization and dataset family as the canonical 15m
# dataset (getdata-finance/xauusd-15m-ohlcv-metals-historical-data).
SOURCE_LICENSE = "MIT"

INTERVAL_SPECIFICATIONS = {
    "1m": {
        "timeframe": "1m",
        "bucket_minutes": 1,
        "grid_check": lambda dt: (dt.second == 0) & (dt.microsecond == 0),
        "slug": "xauusd-1m-ohlcv-metals-historical-data",
        "raw_csv_name": "XAUUSD_1m.csv",
        "default_dataset_id": "ds_moving_average_1m",
        "dir_name": "xauusd_1m",
        "csv_name": "ds_moving_average_1m.csv",
        "license_url": "https://github.com/getdata-finance/xauusd-1m-ohlcv-metals-historical-data/blob/main/LICENSE",
    },
    "5m": {
        "timeframe": "5m",
        "bucket_minutes": 5,
        "grid_check": lambda dt: (dt.minute % 5 == 0) & (dt.second == 0) & (dt.microsecond == 0),
        "slug": "xauusd-5m-ohlcv-metals-historical-data",
        "raw_csv_name": "XAUUSD_5m.csv",
        "default_dataset_id": "ds_moving_average_5m",
        "dir_name": "xauusd_5m",
        "csv_name": "ds_moving_average_5m.csv",
        "license_url": "https://github.com/getdata-finance/xauusd-5m-ohlcv-metals-historical-data/blob/main/LICENSE",
    },
}

SUPPORTED_INGEST_INTERVALS = tuple(INTERVAL_SPECIFICATIONS.keys())


def interval_specification(interval: str) -> dict:
    """Return the immutable ingestion specification for a supported interval."""
    spec = INTERVAL_SPECIFICATIONS.get(interval)
    if spec is None:
        raise ValueError(
            f"Unsupported ingestion interval: {interval}. "
            f"Supported intervals: {', '.join(SUPPORTED_INGEST_INTERVALS)}"
        )
    return spec


def default_source_url(interval: str) -> str:
    spec = interval_specification(interval)
    return (
        "https://raw.githubusercontent.com/getdata-finance/"
        f"{spec['slug']}/main/{spec['raw_csv_name']}"
    )


def analyze_temporal_gaps_for_interval(timestamps: pd.Series, bucket_minutes: int) -> dict:
    """Analyze gap distribution for intraday bars of the given interval.

    Distinguishes weekend gaps, daily session breaks, and market holidays,
    mirroring the canonical 15m gap analysis but scaled to the bar size.
    """
    diffs = timestamps.diff().dropna()
    gap_mask = diffs > pd.Timedelta(minutes=bucket_minutes)
    gaps = diffs[gap_mask]

    weekend_gaps = []
    session_gaps = []
    holiday_gaps = []
    unexplained_gaps = []

    for idx, delta in gaps.items():
        prev_ts = timestamps.loc[idx - 1]
        curr_ts = timestamps.loc[idx]
        delta_sec = delta.total_seconds()

        gap_info = {
            "start": prev_ts.isoformat(),
            "end": curr_ts.isoformat(),
            "gap_seconds": delta_sec,
        }

        # Weekend gap: Friday close to Sunday open (~48 hours)
        if prev_ts.weekday() == 4 and curr_ts.weekday() == 6:
            weekend_gaps.append(gap_info)
        # Daily session break: typical ~1 hour break around 21:00-22:00 UTC
        elif delta_sec <= 7200 and (prev_ts.hour in (20, 21, 22) or curr_ts.hour in (21, 22, 23)):
            session_gaps.append(gap_info)
        # Market holiday closures: Good Friday (April 2-3), Memorial Day (May 25-26),
        # Juneteenth/Independence-Day-eve (June 19, July 3), Labor Day (September 7-8)
        elif (prev_ts.month == 4 and prev_ts.day in (2, 3)) or (prev_ts.month == 5 and prev_ts.day in (25, 26)):
            holiday_gaps.append(gap_info)
        elif (prev_ts.month in (6, 7) and prev_ts.day in (19, 3)) or (prev_ts.month == 9 and prev_ts.day in (7, 8)):
            holiday_gaps.append(gap_info)
        else:
            unexplained_gaps.append(gap_info)

    return {
        "total_gaps_count": len(gaps),
        "weekend_gaps_count": len(weekend_gaps),
        "session_gaps_count": len(session_gaps),
        "holiday_gaps_count": len(holiday_gaps),
        "unexplained_gaps_count": len(unexplained_gaps),
        "max_gap_seconds": gaps.max().total_seconds() if not gaps.empty else 0.0,
        "unexplained_gaps": unexplained_gaps,
    }


def ingest_xauusd_interval(
    interval: str,
    *,
    source_url: str | None = None,
    output_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    commit_path: str | Path | None = None,
    dataset_id: str | None = None,
    raw_data_bytes: bytes | None = None,
    simulate_failure_step: str | None = None,
) -> tuple[Path, dict]:
    """Download, validate, and atomically persist a canonical XAUUSD intraday dataset.

    Enforces the identical atomic staging and multi-file commit protocol as the
    canonical 15m ingestion, with provenance anchored to the exact published
    source payload.
    """
    spec = interval_specification(interval)
    bucket_minutes = spec["bucket_minutes"]
    timeframe = spec["timeframe"]
    resolved_dataset_id = dataset_id or spec["default_dataset_id"]

    source_dir = REPO_ROOT / "data" / "raw" / spec["dir_name"]
    if output_path is None:
        out_p = source_dir / spec["csv_name"]
    else:
        out_p = Path(output_path)
    manifest_p = Path(manifest_path) if manifest_path is not None else out_p.with_suffix(".manifest.json")
    commit_p = Path(commit_path) if commit_path is not None else out_p.with_suffix(".commit.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    resolved_source_url = source_url or default_source_url(interval)

    # Acquire raw bytes if not provided directly
    if raw_data_bytes is None:
        req = urllib.request.Request(
            resolved_source_url,
            headers={"User-Agent": "AI-Trading-Lab/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw_bytes = resp.read()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch raw dataset from {resolved_source_url}: {exc}"
            ) from exc
    else:
        raw_bytes = raw_data_bytes

    if not raw_bytes:
        raise ValueError("Source dataset payload is empty.")

    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    # Parse CSV DataFrame
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise ValueError(f"Failed to parse source CSV payload: {exc}") from exc

    if df.empty:
        raise ValueError("Source dataset DataFrame is empty.")

    # Validate required columns
    required_raw_cols = {"datetime", "open", "high", "low", "close"}
    missing = required_raw_cols - set(df.columns)
    if missing:
        raise ValueError(f"Source CSV missing required columns: {sorted(missing)}")

    # Timeframe and Timestamp Validation
    try:
        parsed_ts = pd.to_datetime(df["datetime"], utc=True)
    except Exception as exc:
        raise ValueError(f"Source CSV contains invalid or unparseable timestamps: {exc}") from exc

    if parsed_ts.isna().any():
        raise ValueError("Source CSV contains invalid or unparseable timestamps.")

    if not parsed_ts.is_monotonic_increasing:
        raise ValueError("Source CSV timestamps are not monotonic increasing.")

    if not parsed_ts.is_unique:
        raise ValueError("Source CSV contains duplicate timestamps.")

    # Strict interval grid alignment check
    grid_mask = ~spec["grid_check"](parsed_ts.dt)
    if grid_mask.any():
        raise ValueError(
            f"Source CSV contains timestamps off the {timeframe} grid."
        )

    # Validate OHLC values and finite float numbers
    open_s = _validate_numeric_series(df["open"], "open")
    high_s = _validate_numeric_series(df["high"], "high")
    low_s = _validate_numeric_series(df["low"], "low")
    close_s = _validate_numeric_series(df["close"], "close")

    # Validate explicit candle geometry invariants:
    # high >= max(open, close, low) and low <= min(open, close, high)
    invalid_high = (high_s < open_s) | (high_s < close_s) | (high_s < low_s)
    invalid_low = (low_s > open_s) | (low_s > close_s) | (low_s > high_s)

    if invalid_high.any() or invalid_low.any():
        raise ValueError("Source CSV contains invalid OHLC candle geometry.")

    # Volume policy: if present, MUST be finite and non-negative.
    volume_s = None
    if "volume" in df.columns:
        vol_num = pd.to_numeric(df["volume"], errors="coerce")
        if vol_num.isna().any():
            raise ValueError("Column 'volume' contains invalid or NaN numeric values.")
        if vol_num.apply(lambda x: math.isinf(x) if isinstance(x, (int, float)) else True).any():
            raise ValueError("Column 'volume' contains non-finite infinity values.")
        if (vol_num < 0.0).any():
            raise ValueError("Column 'volume' contains negative volume values.")
        volume_s = vol_num

    # Gap diagnostics
    gap_analysis = analyze_temporal_gaps_for_interval(parsed_ts, bucket_minutes)

    # Build canonical output DataFrame
    canonical_df = pd.DataFrame(
        {
            "timestamp": df["datetime"],
            "open": open_s,
            "high": high_s,
            "low": low_s,
            "close": close_s,
        }
    )
    if volume_s is not None:
        canonical_df["volume"] = volume_s

    # Serialize canonical CSV to buffer for output checksum calculation
    csv_buf = io.StringIO()
    canonical_df.to_csv(csv_buf, index=False)
    output_csv_bytes = csv_buf.getvalue().encode("utf-8")
    output_sha256 = hashlib.sha256(output_csv_bytes).hexdigest()

    # Build machine-readable provenance manifest
    manifest_data = {
        "manifest_version": "1.0",
        "dataset_id": resolved_dataset_id,
        "symbol": "XAUUSD",
        "timeframe": timeframe,
        "source_url": resolved_source_url,
        "source_license": SOURCE_LICENSE,
        "source_license_url": spec["license_url"],
        "raw_sha256": raw_sha256,
        "output_sha256": output_sha256,
        "row_count": len(canonical_df),
        "columns": list(canonical_df.columns),
        "start_timestamp_utc": parsed_ts.min().isoformat(),
        "end_timestamp_utc": parsed_ts.max().isoformat(),
        "validation_summary": {
            "columns_valid": True,
            "timestamps_monotonic": True,
            "timestamps_unique": True,
            "timestamps_grid_aligned": True,
            "ohlc_geometry_valid": True,
            "volume_valid": volume_s is not None,
        },
        "gap_diagnostics": gap_analysis,
    }

    manifest_json_str = json.dumps(manifest_data, indent=2)

    commit_data = {
        "commit_version": "1.0",
        "dataset_id": resolved_dataset_id,
        "symbol": "XAUUSD",
        "timeframe": timeframe,
        "raw_sha256": raw_sha256,
        "output_sha256": output_sha256,
        "row_count": len(canonical_df),
        "committed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    commit_json_str = json.dumps(commit_data, indent=2)

    # Unique PID/UUID staging paths to prevent concurrent worker collision
    run_id = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp_csv = out_p.with_suffix(f".tmp.{run_id}.csv")
    tmp_json = manifest_p.with_suffix(f".tmp.{run_id}.manifest.json")
    tmp_commit = commit_p.with_suffix(f".tmp.{run_id}.commit.json")

    try:
        # Write staged temporary files
        tmp_csv.write_bytes(output_csv_bytes)
        tmp_json.write_text(manifest_json_str, encoding="utf-8")
        tmp_commit.write_text(commit_json_str, encoding="utf-8")

        # Read back staged CSV and verify schema & hash
        readback_df = pd.read_csv(tmp_csv)
        if len(readback_df) != len(canonical_df):
            raise RuntimeError("Readback verification failed: row count mismatch.")

        readback_sha = hashlib.sha256(tmp_csv.read_bytes()).hexdigest()
        if readback_sha != output_sha256:
            raise RuntimeError("Readback verification failed: SHA-256 checksum mismatch.")

        if simulate_failure_step == "before_csv_replace":
            raise RuntimeError("Simulated failure before CSV replacement!")

        # Step 1: Replace CSV
        os.replace(tmp_csv, out_p)

        if simulate_failure_step == "before_manifest_replace":
            raise RuntimeError("Simulated failure after CSV replace but before Manifest replace!")

        # Step 2: Replace Manifest
        os.replace(tmp_json, manifest_p)

        if simulate_failure_step == "before_commit_replace":
            raise RuntimeError("Simulated failure before commit marker replace!")

        # Step 3: Replace Commit Marker
        os.replace(tmp_commit, commit_p)

    except Exception as exc:
        # Clean up temporary staged files on error
        for p in (tmp_csv, tmp_json, tmp_commit):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        raise RuntimeError(f"Atomic dataset persistence failed: {exc}") from exc

    # Final integrity readback: published CSV/manifest/commit must verify.
    verify_dataset_and_manifest_integrity(out_p, manifest_p, commit_p)

    return out_p, manifest_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest a canonical XAUUSD intraday dataset (1m or 5m) with provenance."
    )
    parser.add_argument("--interval", choices=SUPPORTED_INGEST_INTERVALS, required=True)
    args = parser.parse_args()

    out_path, meta = ingest_xauusd_interval(args.interval)
    print("Downloaded, validated, and persisted XAUUSD", f"{args.interval} dataset and manifest successfully:")
    print("  Output Path:", out_path)
    print("  Source:", meta["source_url"])
    print("  Source License:", meta["source_license"])
    print("  Raw SHA-256:", meta["raw_sha256"])
    print("  Output SHA-256:", meta["output_sha256"])
    print("  Row Count:", meta["row_count"])
    print("  Start UTC:", meta["start_timestamp_utc"])
    print("  End UTC:", meta["end_timestamp_utc"])
    print("  Unexplained gaps:", meta["gap_diagnostics"]["unexplained_gaps_count"])
