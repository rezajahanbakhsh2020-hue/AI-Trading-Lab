"""Regressions for canonical XAUUSD 5m and 1m historical dataset ingestion.

Mirrors the canonical 15m ingestion contract (validation, provenance manifests,
atomic multi-file commit protocol, fail-closed integrity readback) for the
intrategy timeframes that were previously absent from the repository, and ties
the ingested artifacts into canonical research dataset resolution.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import json

import pandas as pd
import pytest

from src.data.ingest_xauusd_intervals import (
    INTERVAL_SPECIFICATIONS,
    SUPPORTED_INGEST_INTERVALS,
    analyze_temporal_gaps_for_interval,
    ingest_xauusd_interval,
    interval_specification,
)
from src.data.ingest_xauusd_15m import verify_dataset_and_manifest_integrity
from src.evaluation.research_runner import resolve_historical_dataset
from src.evaluation.research_constitution import DatasetScope


@pytest.mark.parametrize("interval", SUPPORTED_INGEST_INTERVALS)
def test_specifications_are_canonical(interval: str) -> None:
    spec = interval_specification(interval)
    assert spec["timeframe"] == interval
    assert spec["default_dataset_id"].endswith(interval)
    assert spec["dir_name"] == f"xauusd_{interval}"
    assert spec["license_url"].startswith("https://github.com/getdata-finance/")


@pytest.mark.parametrize("interval", SUPPORTED_INGEST_INTERVALS)
def test_valid_ingestion_and_manifest_generation(tmp_path: Path, interval: str) -> None:
    """Valid ingestion produces CSV, manifest, commit marker with consistent hashes."""
    spec = interval_specification(interval)
    bucket = spec["bucket_minutes"]
    base = pd.Timestamp("2026-03-26T02:30:00+00:00")
    rows = [
        (base + pd.Timedelta(minutes=bucket * i)).isoformat()
        for i in range(6)
    ]
    payload = (
        "datetime,open,high,low,close,volume\n"
        + "\n".join(f"{ts},{100 + i},{105 + i},{99 + i},{102 + i},{1000 + i}" for i, ts in enumerate(rows))
        + "\n"
    ).encode("utf-8")

    out_file = tmp_path / spec["csv_name"]
    manifest_file = tmp_path / "ds.manifest.json"
    commit_file = tmp_path / "ds.commit.json"

    p, meta = ingest_xauusd_interval(
        interval,
        output_path=out_file,
        manifest_path=manifest_file,
        commit_path=commit_file,
        raw_data_bytes=payload,
    )

    assert p.exists()
    assert manifest_file.exists()
    assert commit_file.exists()
    assert meta["row_count"] == 6
    assert meta["output_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert meta["start_timestamp_utc"] == rows[0]
    assert meta["end_timestamp_utc"] == rows[-1]
    assert meta["source_license"] == "MIT"
    assert meta["gap_diagnostics"]["unexplained_gaps_count"] == 0

    df, man = verify_dataset_and_manifest_integrity(out_file, manifest_file, commit_file)
    assert len(df) == 6
    assert man["dataset_id"] == spec["default_dataset_id"]
    assert man["timeframe"] == interval
    assert man["symbol"] == "XAUUSD"


@pytest.mark.parametrize("interval", SUPPORTED_INGEST_INTERVALS)
def test_off_grid_timestamps_rejected(tmp_path: Path, interval: str) -> None:
    spec = interval_specification(interval)
    bucket = spec["bucket_minutes"]
    # Timestamp not on the interval grid (offset by one second on the sub-minute part).
    base = pd.Timestamp("2026-03-26T02:30:01+00:00")
    payload = (
        "datetime,open,high,low,close,volume\n"
        f"{base.isoformat()},100.0,105.0,99.0,102.0,1000\n"
        f"{(base + pd.Timedelta(minutes=bucket)).isoformat()},102.0,106.0,101.0,104.0,1100\n"
    ).encode("utf-8")
    out_file = tmp_path / spec["csv_name"]
    manifest_file = tmp_path / "ds.manifest.json"

    with pytest.raises(ValueError, match="timestamps off the .* grid"):
        ingest_xauusd_interval(
            interval,
            output_path=out_file,
            manifest_path=manifest_file,
            raw_data_bytes=payload,
        )

    # No artifacts may be published on failure.
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("interval", SUPPORTED_INGEST_INTERVALS)
def test_invalid_ohlc_geometry_rejected(tmp_path: Path, interval: str) -> None:
    spec = interval_specification(interval)
    bucket = spec["bucket_minutes"]
    base = pd.Timestamp("2026-03-26T02:30:00+00:00")
    payload = (
        "datetime,open,high,low,close,volume\n"
        f"{base.isoformat()},100.0,105.0,101.5,102.0,1000\n"
        f"{(base + pd.Timedelta(minutes=bucket)).isoformat()},102.0,106.0,101.0,104.0,1100\n"
    ).encode("utf-8")
    # high=105 >= low=101.5 valid; craft a case where close > high.
    payload = (
        "datetime,open,high,low,close,volume\n"
        f"{base.isoformat()},100.0,105.0,99.0,102.0,1000\n"
        f"{(base + pd.Timedelta(minutes=bucket)).isoformat()},102.0,104.0,101.0,105.0,1100\n"
    ).encode("utf-8")

    with pytest.raises(ValueError, match="invalid OHLC candle geometry"):
        ingest_xauusd_interval(
            interval,
            output_path=tmp_path / spec["csv_name"],
            manifest_path=tmp_path / "ds.manifest.json",
            raw_data_bytes=payload,
        )


@pytest.mark.parametrize("interval", SUPPORTED_INGEST_INTERVALS)
def test_empty_payload_rejected(tmp_path: Path, interval: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        ingest_xauusd_interval(
            interval,
            output_path=tmp_path / "ds.csv",
            manifest_path=tmp_path / "ds.manifest.json",
            raw_data_bytes=b"",
        )


@pytest.mark.parametrize("interval", SUPPORTED_INGEST_INTERVALS)
def test_mismatched_manifest_and_csv_generation_fails_closed(tmp_path: Path, interval: str) -> None:
    """Tampering with CSV after commit must fail the integrity readback."""
    spec = interval_specification(interval)
    bucket = spec["bucket_minutes"]
    base = pd.Timestamp("2026-03-26T02:30:00+00:00")
    payload = (
        "datetime,open,high,low,close,volume\n"
        f"{base.isoformat()},100.0,105.0,99.0,102.0,1000\n"
        f"{(base + pd.Timedelta(minutes=bucket)).isoformat()},102.0,106.0,101.0,104.0,1100\n"
    ).encode("utf-8")
    out_file = tmp_path / spec["csv_name"]
    manifest_file = tmp_path / "ds.manifest.json"

    ingest_xauusd_interval(
        interval,
        output_path=out_file,
        manifest_path=manifest_file,
        raw_data_bytes=payload,
    )

    out_file.write_text(
        "timestamp,open,high,low,close\n2026-03-26T02:30:00+00:00,100,105,99,102\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Mismatched dataset/manifest generation detected"):
        verify_dataset_and_manifest_integrity(out_file, manifest_file)


@pytest.mark.parametrize("interval", SUPPORTED_INGEST_INTERVALS)
def test_fault_injection_rolls_back_staged_files(tmp_path: Path, interval: str) -> None:
    """Failure before CSV replacement must leave no data, manifest, or commit marker."""
    spec = interval_specification(interval)
    bucket = spec["bucket_minutes"]
    base = pd.Timestamp("2026-03-26T02:30:00+00:00")
    payload = (
        "datetime,open,high,low,close,volume\n"
        f"{base.isoformat()},100.0,105.0,99.0,102.0,1000\n"
        f"{(base + pd.Timedelta(minutes=bucket)).isoformat()},102.0,106.0,101.0,104.0,1100\n"
    ).encode("utf-8")
    out_file = tmp_path / spec["csv_name"]
    manifest_file = tmp_path / "ds.manifest.json"
    commit_file = tmp_path / "ds.commit.json"

    with pytest.raises(RuntimeError, match="Atomic dataset persistence failed"):
        ingest_xauusd_interval(
            interval,
            output_path=out_file,
            manifest_path=manifest_file,
            commit_path=commit_file,
            raw_data_bytes=payload,
            simulate_failure_step="before_csv_replace",
        )

    assert not out_file.exists()
    assert not manifest_file.exists()
    assert not commit_file.exists()


@pytest.mark.parametrize("interval", SUPPORTED_INGEST_INTERVALS)
def test_duplicate_and_dst_ambiguity_failures(tmp_path: Path, interval: str) -> None:
    """Duplicate timestamps in the source payload fail closed."""
    spec = interval_specification(interval)
    bucket = spec["bucket_minutes"]
    base = pd.Timestamp("2026-03-26T02:30:00+00:00")
    ts = base.isoformat()
    payload = (
        "datetime,open,high,low,close,volume\n"
        f"{ts},100.0,105.0,99.0,102.0,1000\n"
        f"{ts},100.0,105.0,99.0,102.0,1000\n"
    ).encode("utf-8")

    with pytest.raises(ValueError, match="duplicate timestamps"):
        ingest_xauusd_interval(
            interval,
            output_path=tmp_path / spec["csv_name"],
            manifest_path=tmp_path / "ds.manifest.json",
            raw_data_bytes=payload,
        )


def test_unsupported_interval_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported ingestion interval"):
        ingest_xauusd_interval("2m", output_path=tmp_path / "ds.csv")
    with pytest.raises(ValueError, match="Unsupported ingestion interval"):
        interval_specification("7m")


def test_gap_classification_parameterized_by_bucket() -> None:
    """Gaps are measured strictly above the interval bucket size."""
    bucket = 5
    base = pd.Timestamp("2026-03-26T10:00:00+00:00")
    timestamps = pd.Series(
        [
            base,
            base + pd.Timedelta(minutes=bucket),
            base + pd.Timedelta(minutes=2 * bucket + 10),  # 10-minute gap over 5m bucket
        ]
    )
    analysis = analyze_temporal_gaps_for_interval(timestamps, bucket)
    assert analysis["total_gaps_count"] == 1
    assert analysis["max_gap_seconds"] == 900.0
    assert analysis["unexplained_gaps_count"] == 1


@pytest.mark.parametrize(
    ("dir_name", "dataset_id", "timeframe"),
    [
        ("xauusd_1m", "ds_moving_average_1m", "1m"),
        ("xauusd_5m", "ds_moving_average_5m", "5m"),
    ],
)
@pytest.mark.skipif(
    not all(
        (Path("data/raw") / spec["dir_name"] / spec["csv_name"]).exists()
        for spec in INTERVAL_SPECIFICATIONS.values()
    ),
    reason="Persisted 1m/5m datasets are ingested on the PR branch only",
)
def test_ingested_real_datasets_resolve_through_canonical_scope(
    dir_name: str, dataset_id: str, timeframe: str
) -> None:
    """The committed 1m/5m datasets resolve via strict DatasetScope identity checks."""
    data_dir = Path("data/raw") / dir_name
    df, manifest = verify_dataset_and_manifest_integrity(
        data_dir / f"{dataset_id}.csv",
        data_dir / f"{dataset_id}.manifest.json",
        data_dir / f"{dataset_id}.commit.json",
    )
    assert not df.empty
    assert manifest["timeframe"] == timeframe
    assert manifest["source_license"] == "MIT"

    scope = DatasetScope(
        dataset_id=manifest["dataset_id"],
        symbol=manifest["symbol"],
        timeframe=manifest["timeframe"],
        start_date=manifest["start_timestamp_utc"][:10],
        end_date=manifest["end_timestamp_utc"][:10],
    )
    resolved = resolve_historical_dataset(scope)
    assert not resolved.empty
