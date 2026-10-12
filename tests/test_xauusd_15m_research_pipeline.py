"""
Adversarial regression and forensic tests for XAUUSD 15m dataset ingestion,
schema validation, provenance manifests, atomic multi-file commit rollback,
off-grid timestamp detection, research execution, and fail-closed readiness gating.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.ingest_xauusd_15m import (
    analyze_temporal_gaps,
    download_and_validate_xauusd_15m,
    verify_dataset_and_manifest_integrity,
)
from src.evaluation.hypothesis_generator import accept_hypothesis_for_research
from src.evaluation.live_execution_runtime import verify_timeframe_production_readiness
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
    HypothesisStatus,
    ResearchHypothesis,
    WalkForwardProtocol,
)
from src.evaluation.research_runner import resolve_historical_dataset, run_research_experiment
from src.evaluation.research_store import resolve_promoted_candidate


@pytest.fixture
def mock_15m_csv_bytes() -> bytes:
    """Fixture providing a minimal, valid 15m CSV payload."""
    data = (
        "datetime,open,high,low,close,volume\n"
        "2026-03-26T02:30:00+00:00,100.0,105.0,99.0,102.0,1000\n"
        "2026-03-26T02:45:00+00:00,102.0,106.0,101.0,104.0,1100\n"
        "2026-03-26T03:00:00+00:00,104.0,108.0,103.0,107.0,1200\n"
        "2026-03-26T03:15:00+00:00,107.0,109.0,105.0,106.0,1300\n"
        "2026-03-26T03:30:00+00:00,106.0,110.0,104.0,108.0,1400\n"
    )
    return data.encode("utf-8")


def test_valid_ingestion_and_manifest_generation(tmp_path: Path, mock_15m_csv_bytes: bytes) -> None:
    """Test valid ingestion, raw/output SHA256 hashes, and manifest schema consistency."""
    out_file = tmp_path / "ds_moving_average_15m.csv"
    manifest_file = tmp_path / "ds_moving_average_15m.manifest.json"
    commit_file = tmp_path / "ds_moving_average_15m.commit.json"

    p, meta = download_and_validate_xauusd_15m(
        output_path=out_file,
        manifest_path=manifest_file,
        commit_path=commit_file,
        raw_data_bytes=mock_15m_csv_bytes,
    )

    assert p.exists()
    assert manifest_file.exists()
    assert commit_file.exists()
    assert meta["row_count"] == 5
    assert meta["raw_sha256"] != ""
    assert meta["output_sha256"] != ""
    assert meta["start_timestamp_utc"] == "2026-03-26T02:30:00+00:00"
    assert meta["end_timestamp_utc"] == "2026-03-26T03:30:00+00:00"

    # Verify readback integrity
    df, man = verify_dataset_and_manifest_integrity(out_file, manifest_file, commit_file)
    assert len(df) == 5
    assert man["output_sha256"] == meta["output_sha256"]


def test_off_grid_timestamps_rejected(tmp_path: Path) -> None:
    """Test that timestamps off the 15-minute grid (e.g. 12:07:33) raise ValueError."""
    off_grid_data = (
        "datetime,open,high,low,close,volume\n"
        "2026-03-26T02:37:15+00:00,100.0,105.0,99.0,102.0,1000\n"
    ).encode("utf-8")
    out_file = tmp_path / "ds_moving_average_15m.csv"
    manifest_file = tmp_path / "ds_moving_average_15m.manifest.json"

    with pytest.raises(ValueError, match="timestamps off the 15-minute grid"):
        download_and_validate_xauusd_15m(
            output_path=out_file,
            manifest_path=manifest_file,
            raw_data_bytes=off_grid_data,
        )


def test_mismatched_manifest_and_csv_generation_fails_closed(tmp_path: Path, mock_15m_csv_bytes: bytes) -> None:
    """Test that when CSV is updated but manifest is stale/mismatched, reader fails closed."""
    out_file = tmp_path / "ds_moving_average_15m.csv"
    manifest_file = tmp_path / "ds_moving_average_15m.manifest.json"

    download_and_validate_xauusd_15m(
        output_path=out_file,
        manifest_path=manifest_file,
        raw_data_bytes=mock_15m_csv_bytes,
    )

    # Tamper with CSV content to simulate interrupted or mismatched publication
    out_file.write_text("timestamp,open,high,low,close\n2026-03-26T02:30:00+00:00,100,105,99,102\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Mismatched dataset/manifest generation detected"):
        verify_dataset_and_manifest_integrity(out_file, manifest_file)


def test_fault_injection_interrupted_publication(tmp_path: Path, mock_15m_csv_bytes: bytes) -> None:
    """Phase 1: Test fault injection simulating failure during multi-file replacement."""
    out_file = tmp_path / "ds_moving_average_15m.csv"
    manifest_file = tmp_path / "ds_moving_average_15m.manifest.json"
    commit_file = tmp_path / "ds_moving_average_15m.commit.json"

    # Pre-populate valid initial dataset
    download_and_validate_xauusd_15m(
        output_path=out_file,
        manifest_path=manifest_file,
        commit_path=commit_file,
        raw_data_bytes=mock_15m_csv_bytes,
    )

    initial_csv = out_file.read_bytes()
    initial_manifest = manifest_file.read_bytes()

    # Valid replacement payload
    valid_replacement_payload = (
        "datetime,open,high,low,close,volume\n"
        "2026-03-26T02:30:00+00:00,110.0,115.0,109.0,112.0,2000\n"
        "2026-03-26T02:45:00+00:00,112.0,116.0,111.0,114.0,2100\n"
    ).encode("utf-8")

    # Case A: Staging failure before CSV replacement preserves initial files
    with pytest.raises(RuntimeError, match="Atomic dataset persistence failed"):
        download_and_validate_xauusd_15m(
            output_path=out_file,
            manifest_path=manifest_file,
            commit_path=commit_file,
            raw_data_bytes=valid_replacement_payload,
            simulate_failure_step="before_csv_replace",
        )
    assert out_file.read_bytes() == initial_csv
    assert manifest_file.read_bytes() == initial_manifest

    # Case B: Interrupted publication step (after CSV replace but before manifest replace)
    with pytest.raises(RuntimeError, match="Atomic dataset persistence failed"):
        download_and_validate_xauusd_15m(
            output_path=out_file,
            manifest_path=manifest_file,
            commit_path=commit_file,
            raw_data_bytes=valid_replacement_payload,
            simulate_failure_step="before_manifest_replace",
        )

    # Verify reader detects generation mismatch between CSV and manifest, and fails closed
    with pytest.raises(RuntimeError, match="Mismatched dataset/manifest generation detected"):
        verify_dataset_and_manifest_integrity(out_file, manifest_file, commit_file)


def test_non_finite_numeric_and_volume_rejected(tmp_path: Path) -> None:
    """Test non-finite infinity values and negative volume rejection."""
    out_file = tmp_path / "ds_moving_average_15m.csv"
    manifest_file = tmp_path / "ds_moving_average_15m.manifest.json"

    # Inf in high
    inf_data = "datetime,open,high,low,close\n2026-03-26T02:30:00+00:00,100.0,inf,99.0,102.0\n".encode("utf-8")
    with pytest.raises(ValueError, match="contains non-finite infinity values"):
        download_and_validate_xauusd_15m(
            output_path=out_file, manifest_path=manifest_file, raw_data_bytes=inf_data
        )

    # Negative volume
    neg_vol = "datetime,open,high,low,close,volume\n2026-03-26T02:30:00+00:00,100.0,105.0,99.0,102.0,-10.0\n".encode("utf-8")
    with pytest.raises(ValueError, match="contains negative volume values"):
        download_and_validate_xauusd_15m(
            output_path=out_file, manifest_path=manifest_file, raw_data_bytes=neg_vol
        )


def test_gap_diagnostics_and_holiday_classification() -> None:
    """Phase 2: Test temporal gap analysis and holiday gap classification."""
    ts = pd.Series(
        [
            pd.Timestamp("2026-03-27 21:00:00+00:00"),  # Friday
            pd.Timestamp("2026-03-29 22:00:00+00:00"),  # Sunday open (weekend gap)
            pd.Timestamp("2026-04-02 20:45:00+00:00"),  # Good Friday holiday
            pd.Timestamp("2026-04-03 14:30:00+00:00"),
        ]
    )
    diag = analyze_temporal_gaps(ts)
    assert diag["total_gaps_count"] == 3
    assert diag["weekend_gaps_count"] == 1
    assert diag["holiday_gaps_count"] == 1


def test_15m_dataset_resolution_and_research_execution() -> None:
    """Test dataset resolution via resolve_historical_dataset and research experiment execution."""
    ds = DatasetScope(
        dataset_id="ds_moving_average_15m",
        symbol="XAUUSD",
        timeframe="15m",
        start_date="2026-03-26",
        end_date="2026-09-25",
    )

    resolved = resolve_historical_dataset(ds)
    assert not resolved.empty
    assert len(resolved) == 11989
    assert list(resolved.columns[:5]) == ["timestamp", "open", "high", "low", "close"]

    hyp = ResearchHypothesis(
        statement="XAUUSD 15m Baseline Moving Average Strategy",
        methodology_version="1.0",
        strategy_name="baseline",
        strategy_version="1.0",
        dataset_scope=ds,
        execution_assumptions=ExecutionAssumptions(transaction_cost=0.0001, slippage=0.0001, latency_ms=0.0),
        code_provenance=CodeProvenance(commit_sha="dd9988845b7c310db2c4b1387d78524f985381e1"),
        benchmark_reference="buy_and_hold",
        walk_forward_protocol=WalkForwardProtocol(train_size=1000, test_size=200),
        parameters={"fast_window": 20, "slow_window": 50},
        status=HypothesisStatus.GENERATED,
    )

    accepted = accept_hypothesis_for_research(hyp)
    evidence = run_research_experiment(spec=accepted)

    assert evidence.experiment_fingerprint == accepted.fingerprint
    assert len(evidence.partitions) == 4
    # Strategy fails walk-forward qualification cleanly with explicit reasons
    assert evidence.promotion_status.value == "REJECTED"
    assert len(evidence.rejection_reasons) > 0


def test_15m_unpromoted_candidate_fails_closed() -> None:
    """Test that unpromoted 15m candidate fails closed in candidate resolution and readiness."""
    resolved_candidate = resolve_promoted_candidate(symbol="XAUUSD", timeframe="15m")
    assert resolved_candidate is None

    readiness = verify_timeframe_production_readiness(symbol="XAUUSD", timeframe="15m")
    assert readiness["status"] == "BLOCKED"
    assert readiness["reason_code"] == "PromotionUnavailable"
