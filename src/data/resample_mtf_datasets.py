"""
Resample canonical XAUUSD 15-minute dataset into deterministic higher-timeframe
datasets (30m, 1h, 4h) with strict group completeness checks, grid alignment,
provenance manifests, and atomic multi-file commit protocol.
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

import pandas as pd

from src.data.ingest_xauusd_15m import (
    _validate_numeric_series,
    analyze_temporal_gaps,
)
from src.data.validation import verify_dataset_and_manifest_integrity

DEFAULT_15M_CSV = Path(__file__).resolve().parents[2] / "data" / "raw" / "xauusd_15m" / "ds_moving_average_15m.csv"
DEFAULT_15M_MANIFEST = Path(__file__).resolve().parents[2] / "data" / "raw" / "xauusd_15m" / "ds_moving_average_15m.manifest.json"

TIMEFRAME_SPECS = {
    "30m": {
        "freq": "30min",
        "expected_bars": 2,
        "dataset_id": "ds_moving_average_30m",
        "dir_name": "xauusd_30m",
        "csv_name": "ds_moving_average_30m.csv",
        "grid_check": lambda dt: (dt.dt.minute % 30 == 0) & (dt.dt.second == 0) & (dt.dt.microsecond == 0),
    },
    "1h": {
        "freq": "1h",
        "expected_bars": 4,
        "dataset_id": "ds_moving_average_1h",
        "dir_name": "xauusd_1h",
        "csv_name": "ds_moving_average_1h.csv",
        "grid_check": lambda dt: (dt.dt.minute == 0) & (dt.dt.second == 0) & (dt.dt.microsecond == 0),
    },
    "4h": {
        "freq": "4h",
        "expected_bars": 16,
        "dataset_id": "ds_moving_average_4h",
        "dir_name": "xauusd_4h",
        "csv_name": "ds_moving_average_4h.csv",
        "grid_check": lambda dt: (dt.dt.hour % 4 == 0) & (dt.dt.minute == 0) & (dt.dt.second == 0) & (dt.dt.microsecond == 0),
    },
}


def resample_xauusd_timeframe(
    timeframe: str,
    source_15m_csv: str | Path = DEFAULT_15M_CSV,
    source_15m_manifest: str | Path = DEFAULT_15M_MANIFEST,
    output_dir: str | Path | None = None,
    simulate_failure_step: str | None = None,
) -> tuple[Path, dict]:
    """Resample canonical 15m dataset into specified target timeframe (30m, 1h, 4h).

    Requires exact constituent bar count completeness per target interval.
    Enforces grid alignment, OHLC geometry, provenance manifests, and atomic persistence.
    """
    if timeframe not in TIMEFRAME_SPECS:
        raise ValueError(f"Unsupported timeframe for resampling: {timeframe}. Supported: {list(TIMEFRAME_SPECS)}")

    spec = TIMEFRAME_SPECS[timeframe]
    src_csv_p = Path(source_15m_csv)
    src_man_p = Path(source_15m_manifest)

    # 1. Verify 15m source dataset and manifest integrity
    df_15m, src_manifest = verify_dataset_and_manifest_integrity(src_csv_p, src_man_p)
    src_bytes = src_csv_p.read_bytes()
    src_output_sha = hashlib.sha256(src_bytes).hexdigest()

    # 2. Resample requiring complete constituent groups
    parsed_ts = pd.to_datetime(df_15m["timestamp"], utc=True)
    df_work = df_15m.copy()
    df_work["dt"] = parsed_ts

    freq = spec["freq"]
    req_bars = spec["expected_bars"]

    # Filter out incomplete groups
    grouped = df_work.groupby(pd.Grouper(key="dt", freq=freq))
    complete_groups = grouped.filter(lambda g: len(g) == req_bars)

    if complete_groups.empty:
        raise ValueError(f"No complete {timeframe} groups (requiring {req_bars} x 15m bars) found in source dataset.")

    agg_dict = {
        "timestamp": "first",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in complete_groups.columns:
        agg_dict["volume"] = "sum"

    resampled_df = (
        complete_groups.groupby(pd.Grouper(key="dt", freq=freq))
        .agg(agg_dict)
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # 3. Validate resampled DataFrame schema and grid alignment
    res_ts = pd.to_datetime(resampled_df["timestamp"], utc=True)

    if not res_ts.is_monotonic_increasing:
        raise ValueError(f"Resampled {timeframe} timestamps are not monotonic increasing.")

    if not res_ts.is_unique:
        raise ValueError(f"Resampled {timeframe} contains duplicate timestamps.")

    if not spec["grid_check"](res_ts).all():
        raise ValueError(f"Resampled {timeframe} dataset contains timestamps off the {timeframe} grid.")

    # Validate numeric OHLC series and candle geometry
    open_s = _validate_numeric_series(resampled_df["open"], "open")
    high_s = _validate_numeric_series(resampled_df["high"], "high")
    low_s = _validate_numeric_series(resampled_df["low"], "low")
    close_s = _validate_numeric_series(resampled_df["close"], "close")

    invalid_high = (high_s < open_s) | (high_s < close_s) | (high_s < low_s)
    invalid_low = (low_s > open_s) | (low_s > close_s) | (low_s > high_s)

    if invalid_high.any() or invalid_low.any():
        raise ValueError(f"Resampled {timeframe} dataset contains invalid OHLC candle geometry.")

    volume_s = None
    if "volume" in resampled_df.columns:
        vol_num = pd.to_numeric(resampled_df["volume"], errors="coerce")
        if vol_num.isna().any() or vol_num.apply(lambda x: math.isinf(x) if isinstance(x, (int, float)) else True).any():
            raise ValueError(f"Resampled {timeframe} volume contains NaN or infinite values.")
        if (vol_num < 0.0).any():
            raise ValueError(f"Resampled {timeframe} volume contains negative values.")
        volume_s = vol_num

    # 4. Prepare target output paths
    base_data_dir = src_csv_p.parents[1] if output_dir is None else Path(output_dir)
    target_dir = base_data_dir / spec["dir_name"]
    target_dir.mkdir(parents=True, exist_ok=True)

    target_csv = target_dir / spec["csv_name"]
    target_manifest = target_dir / spec["csv_name"].replace(".csv", ".manifest.json")
    target_commit = target_dir / spec["csv_name"].replace(".csv", ".commit.json")

    # Serialize canonical CSV
    canonical_df = pd.DataFrame(
        {
            "timestamp": resampled_df["timestamp"],
            "open": open_s,
            "high": high_s,
            "low": low_s,
            "close": close_s,
        }
    )
    if volume_s is not None:
        canonical_df["volume"] = volume_s

    csv_buf = io.StringIO()
    canonical_df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")
    output_sha256 = hashlib.sha256(csv_bytes).hexdigest()

    # Gap diagnostics
    gap_analysis = analyze_temporal_gaps(res_ts)

    # Build machine-readable manifest
    manifest_data = {
        "manifest_version": "1.0",
        "dataset_id": spec["dataset_id"],
        "symbol": "XAUUSD",
        "timeframe": timeframe,
        "resampled_from": {
            "source_dataset_id": src_manifest.get("dataset_id", "ds_moving_average_15m"),
            "source_timeframe": "15m",
            "source_output_sha256": src_output_sha,
            "required_constituent_bars": req_bars,
        },
        "raw_sha256": src_output_sha,
        "output_sha256": output_sha256,
        "row_count": len(canonical_df),
        "columns": list(canonical_df.columns),
        "start_timestamp_utc": res_ts.min().isoformat(),
        "end_timestamp_utc": res_ts.max().isoformat(),
        "validation_summary": {
            "columns_valid": True,
            "timestamps_monotonic": True,
            "timestamps_unique": True,
            "timestamps_grid_aligned": True,
            "ohlc_geometry_valid": True,
            "volume_valid": volume_s is not None,
            "resampling_group_completeness_valid": True,
        },
        "gap_diagnostics": gap_analysis,
    }

    manifest_json_str = json.dumps(manifest_data, indent=2)

    commit_data = {
        "commit_version": "1.0",
        "dataset_id": spec["dataset_id"],
        "symbol": "XAUUSD",
        "timeframe": timeframe,
        "raw_sha256": src_output_sha,
        "output_sha256": output_sha256,
        "row_count": len(canonical_df),
        "committed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    commit_json_str = json.dumps(commit_data, indent=2)

    # 5. Atomic Staging and Commit Protocol
    run_id = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp_csv = target_csv.with_suffix(f".tmp.{run_id}.csv")
    tmp_json = target_manifest.with_suffix(f".tmp.{run_id}.manifest.json")
    tmp_commit = target_commit.with_suffix(f".tmp.{run_id}.commit.json")

    try:
        tmp_csv.write_bytes(csv_bytes)
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

        os.replace(tmp_csv, target_csv)

        if simulate_failure_step == "before_manifest_replace":
            raise RuntimeError("Simulated failure before manifest replacement!")

        os.replace(tmp_json, target_manifest)

        if simulate_failure_step == "before_commit_replace":
            raise RuntimeError("Simulated failure before commit marker replacement!")

        os.replace(tmp_commit, target_commit)

    except Exception as exc:
        for p in (tmp_csv, tmp_json, tmp_commit):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        raise RuntimeError(f"Atomic dataset persistence failed for {timeframe}: {exc}") from exc

    return target_csv, manifest_data


def generate_all_mtf_datasets(
    source_15m_csv: str | Path = DEFAULT_15M_CSV,
    source_15m_manifest: str | Path = DEFAULT_15M_MANIFEST,
) -> dict[str, tuple[Path, dict]]:
    """Generate all target MTF datasets (30m, 1h, 4h) from canonical 15m source."""
    results = {}
    for tf in ("30m", "1h", "4h"):
        out_p, meta = resample_xauusd_timeframe(
            timeframe=tf,
            source_15m_csv=source_15m_csv,
            source_15m_manifest=source_15m_manifest,
        )
        results[tf] = (out_p, meta)
    return results


if __name__ == "__main__":
    res = generate_all_mtf_datasets()
    print("Resampled all higher-timeframe datasets successfully:")
    for tf, (p, meta) in res.items():
        print(f"  [{tf}] Path: {p}")
        print(f"       SHA256: {meta['output_sha256']}")
        print(f"       Rows: {meta['row_count']}")
