"""
Ingest, validate, and persist historical XAUUSD 5-minute market data with provenance manifests,
atomic crash-safe multi-file generation commit protocol, and strict 5-minute grid verification.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import uuid
import urllib.parse
import urllib.request

import pandas as pd

from src.data.ingest_xauusd_15m import _validate_numeric_series
from src.data.validation import verify_dataset_and_manifest_integrity

DEFAULT_SOURCE_URL = (
    "https://raw.githubusercontent.com/getdata-finance/"
    "xauusd-5m-ohlcv-metals-historical-data/main/XAUUSD_5m.csv"
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "xauusd_5m"
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "ds_moving_average_5m.csv"


def analyze_5m_temporal_gaps(timestamps: pd.Series) -> dict:
    """Analyze gap distribution for 5-minute bars (>300 seconds).

    Distinguishes weekend gaps, daily session breaks, and market holidays.
    """
    diffs = timestamps.diff().dropna()
    gap_mask = diffs > pd.Timedelta(minutes=5)
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
        # Market holiday closures
        elif (prev_ts.month == 4 and prev_ts.day in (2, 3)) or (prev_ts.month == 5 and prev_ts.day in (25, 26)):
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


def download_and_validate_xauusd_5m(
    source_url: str = DEFAULT_SOURCE_URL,
    output_path: str | Path = DEFAULT_OUTPUT_FILE,
    manifest_path: str | Path | None = None,
    commit_path: str | Path | None = None,
    raw_data_bytes: bytes | None = None,
    simulate_failure_step: str | None = None,
) -> tuple[Path, dict]:
    """Download, validate, and atomically persist canonical XAUUSD 5m historical dataset.

    Enforces atomic staging and multi-file commit protocol with full rollback protection.
    """
    out_p = Path(output_path)
    manifest_p = Path(manifest_path) if manifest_path is not None else out_p.with_suffix(".manifest.json")
    commit_p = Path(commit_path) if commit_path is not None else out_p.with_suffix(".commit.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Acquire raw bytes if not provided directly
    if raw_data_bytes is None:
        req = urllib.request.Request(
            source_url,
            headers={"User-Agent": "AI-Trading-Lab/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw_bytes = resp.read()
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch raw dataset from {source_url}: {exc}") from exc
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

    # Strict 5-minute grid alignment check
    off_grid = (parsed_ts.dt.minute % 5 != 0) | (parsed_ts.dt.second != 0) | (parsed_ts.dt.microsecond != 0)
    if off_grid.any():
        raise ValueError("Source CSV contains timestamps off the 5-minute grid.")

    # Validate OHLC values and finite float numbers
    open_s = _validate_numeric_series(df["open"], "open")
    high_s = _validate_numeric_series(df["high"], "high")
    low_s = _validate_numeric_series(df["low"], "low")
    close_s = _validate_numeric_series(df["close"], "close")

    # Validate explicit candle geometry invariants
    invalid_high = (high_s < open_s) | (high_s < close_s) | (high_s < low_s)
    invalid_low = (low_s > open_s) | (low_s > close_s) | (low_s > high_s)

    if invalid_high.any() or invalid_low.any():
        raise ValueError("Source CSV contains invalid OHLC candle geometry.")

    # Validate Volume policy
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
    gap_analysis = analyze_5m_temporal_gaps(parsed_ts)

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

    # Serialize canonical CSV
    csv_buf = io.StringIO()
    canonical_df.to_csv(csv_buf, index=False)
    output_csv_str = csv_buf.getvalue()
    output_csv_bytes = output_csv_str.encode("utf-8")
    output_sha256 = hashlib.sha256(output_csv_bytes).hexdigest()

    # Build machine-readable provenance manifest
    manifest_data = {
        "manifest_version": "1.0",
        "dataset_id": "ds_moving_average_5m",
        "symbol": "XAUUSD",
        "timeframe": "5m",
        "source_url": source_url,
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
        "dataset_id": "ds_moving_average_5m",
        "symbol": "XAUUSD",
        "timeframe": "5m",
        "raw_sha256": raw_sha256,
        "output_sha256": output_sha256,
        "row_count": len(canonical_df),
        "committed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    commit_json_str = json.dumps(commit_data, indent=2)

    run_id = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp_csv = out_p.with_suffix(f".tmp.{run_id}.csv")
    tmp_json = manifest_p.with_suffix(f".tmp.{run_id}.manifest.json")
    tmp_commit = commit_p.with_suffix(f".tmp.{run_id}.commit.json")

    try:
        tmp_csv.write_bytes(output_csv_bytes)
        tmp_json.write_text(manifest_json_str, encoding="utf-8")
        tmp_commit.write_text(commit_json_str, encoding="utf-8")

        readback_df = pd.read_csv(tmp_csv)
        if len(readback_df) != len(canonical_df):
            raise RuntimeError("Readback verification failed: row count mismatch.")

        readback_sha = hashlib.sha256(tmp_csv.read_bytes()).hexdigest()
        if readback_sha != output_sha256:
            raise RuntimeError("Readback verification failed: SHA-256 checksum mismatch.")

        if simulate_failure_step == "before_csv_replace":
            raise RuntimeError("Simulated failure before CSV replacement!")

        os.replace(tmp_csv, out_p)

        if simulate_failure_step == "before_manifest_replace":
            raise RuntimeError("Simulated failure after CSV replace but before Manifest replace!")

        os.replace(tmp_json, manifest_p)

        if simulate_failure_step == "before_commit_replace":
            raise RuntimeError("Simulated failure before commit marker replace!")

        os.replace(tmp_commit, commit_p)

    except Exception as exc:
        for p in (tmp_csv, tmp_json, tmp_commit):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        raise RuntimeError(f"Atomic dataset persistence failed: {exc}") from exc

    return out_p, manifest_data


if __name__ == "__main__":
    out_path, meta = download_and_validate_xauusd_5m()
    print("Downloaded, validated, and persisted XAUUSD 5m dataset and manifest successfully:")
    print("  Output Path:", out_path)
    print("  Raw SHA-256:", meta["raw_sha256"])
    print("  Output SHA-256:", meta["output_sha256"])
    print("  Row Count:", meta["row_count"])
    print("  Start UTC:", meta["start_timestamp_utc"])
    print("  End UTC:", meta["end_timestamp_utc"])
