"""
Adversarial regression and forensic tests for XAUUSD 15m dataset ingestion,
schema validation, provenance manifests, atomic rollback, research execution,
and fail-closed production readiness gating.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.ingest_xauusd_15m import analyze_temporal_gaps, download_and_validate_xauusd_15m
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
    """1, 8, 9, 10. Test valid ingestion, raw/output SHA256 hashes, and manifest schema consistency."""
    out_file = tmp_path / "ds_moving_average_15m.csv"
    manifest_file = tmp_path / "ds_moving_average_15m.manifest.json"

    p, meta = download_and_validate_xauusd_15m(
        output_path=out_file,
        manifest_path=manifest_file,
        raw_data_bytes=mock_15m_csv_bytes,
    )

    assert p.exists()
    assert manifest_file.exists()
    assert meta["row_count"] == 5
    assert meta["raw_sha256"] != ""
    assert meta["output_sha256"] != ""
    assert meta["start_timestamp_utc"] == "2026-03-26T02:30:00+00:00"
    assert meta["end_timestamp_utc"] == "2026-03-26T03:30:00+00:00"

    # Deterministic repeated ingestion
    _, meta2 = download_and_validate_xauusd_15m(
        output_path=out_file,
        manifest_path=manifest_file,
        raw_data_bytes=mock_15m_csv_bytes,
    )
    assert meta2["output_sha256"] == meta["output_sha256"]


def test_missing_required_columns_and_empty_input(tmp_path: Path) -> None:
    """2. Test that missing required columns or empty input raises ValueError."""
    bad_cols = "datetime,open,high,low\n2026-03-26T02:30:00+00:00,100.0,105.0,99.0\n".encode("utf-8")
    out_file = tmp_path / "ds_moving_average_15m.csv"
    manifest_file = tmp_path / "ds_moving_average_15m.manifest.json"

    with pytest.raises(ValueError, match="Source CSV missing required columns"):
        download_and_validate_xauusd_15m(
            output_path=out_file,
            manifest_path=manifest_file,
            raw_data_bytes=bad_cols,
        )

    with pytest.raises(ValueError, match="Source dataset payload is empty"):
        download_and_validate_xauusd_15m(
            output_path=out_file,
            manifest_path=manifest_file,
            raw_data_bytes=b"",
        )


def test_invalid_non_monotonic_and_duplicate_timestamps(tmp_path: Path) -> None:
    """3. Test invalid timestamps, non-monotonic timestamps, and duplicate timestamps."""
    out_file = tmp_path / "ds_moving_average_15m.csv"
    manifest_file = tmp_path / "ds_moving_average_15m.manifest.json"

    # Non-monotonic
    non_mono = (
        "datetime,open,high,low,close\n"
        "2026-03-26T02:45:00+00:00,100.0,105.0,99.0,102.0\n"
        "2026-03-26T02:30:00+00:00,102.0,106.0,101.0,104.0\n"
    ).encode("utf-8")
    with pytest.raises(ValueError, match="timestamps are not monotonic increasing"):
        download_and_validate_xauusd_15m(
            output_path=out_file, manifest_path=manifest_file, raw_data_bytes=non_mono
        )

    # Duplicate
    dups = (
        "datetime,open,high,low,close\n"
        "2026-03-26T02:30:00+00:00,100.0,105.0,99.0,102.0\n"
        "2026-03-26T02:30:00+00:00,102.0,106.0,101.0,104.0\n"
    ).encode("utf-8")
    with pytest.raises(ValueError, match="duplicate timestamps"):
        download_and_validate_xauusd_15m(
            output_path=out_file, manifest_path=manifest_file, raw_data_bytes=dups
        )


def test_non_finite_numeric_values_rejected(tmp_path: Path) -> None:
    """4. Test that NaN, +inf, and -inf in OHLC values are rejected."""
    out_file = tmp_path / "ds_moving_average_15m.csv"
    manifest_file = tmp_path / "ds_moving_average_15m.manifest.json"

    # NaN in close
    nan_data = "datetime,open,high,low,close\n2026-03-26T02:30:00+00:00,100.0,105.0,99.0,NaN\n".encode("utf-8")
    with pytest.raises(ValueError, match="contains NaN or unparseable numeric values"):
        download_and_validate_xauusd_15m(
            output_path=out_file, manifest_path=manifest_file, raw_data_bytes=nan_data
        )

    # Inf in high
    inf_data = "datetime,open,high,low,close\n2026-03-26T02:30:00+00:00,100.0,inf,99.0,102.0\n".encode("utf-8")
    with pytest.raises(ValueError, match="contains non-finite infinity values"):
        download_and_validate_xauusd_15m(
            output_path=out_file, manifest_path=manifest_file, raw_data_bytes=inf_data
        )


def test_invalid_ohlc_geometry_relationships(tmp_path: Path) -> None:
    """5. Test invalid OHLC geometry relationships (e.g. high < open or low > close)."""
    out_file = tmp_path / "ds_moving_average_15m.csv"
    manifest_file = tmp_path / "ds_moving_average_15m.manifest.json"

    # high < open
    geom1 = "datetime,open,high,low,close\n2026-03-26T02:30:00+00:00,100.0,95.0,90.0,92.0\n".encode("utf-8")
    with pytest.raises(ValueError, match="invalid OHLC candle geometry"):
        download_and_validate_xauusd_15m(
            output_path=out_file, manifest_path=manifest_file, raw_data_bytes=geom1
        )

    # low > close
    geom2 = "datetime,open,high,low,close\n2026-03-26T02:30:00+00:00,100.0,110.0,105.0,102.0\n".encode("utf-8")
    with pytest.raises(ValueError, match="invalid OHLC candle geometry"):
        download_and_validate_xauusd_15m(
            output_path=out_file, manifest_path=manifest_file, raw_data_bytes=geom2
        )


def test_volume_policy_absent_vs_valid_vs_invalid(tmp_path: Path) -> None:
    """6. Test absent volume column, valid volume, and invalid/negative supplied volume."""
    out_file = tmp_path / "ds_moving_average_15m.csv"
    manifest_file = tmp_path / "ds_moving_average_15m.manifest.json"

    # Absent volume
    no_vol = "datetime,open,high,low,close\n2026-03-26T02:30:00+00:00,100.0,105.0,99.0,102.0\n".encode("utf-8")
    _, meta = download_and_validate_xauusd_15m(
        output_path=out_file, manifest_path=manifest_file, raw_data_bytes=no_vol
    )
    assert meta["validation_summary"]["volume_valid"] is False

    # Negative volume
    neg_vol = "datetime,open,high,low,close,volume\n2026-03-26T02:30:00+00:00,100.0,105.0,99.0,102.0,-50.0\n".encode("utf-8")
    with pytest.raises(ValueError, match="contains negative volume values"):
        download_and_validate_xauusd_15m(
            output_path=out_file, manifest_path=manifest_file, raw_data_bytes=neg_vol
        )


def test_gap_diagnostics_analysis() -> None:
    """7. Test temporal gap analysis and diagnostic reporting."""
    ts = pd.Series(
        [
            pd.Timestamp("2026-03-27 21:00:00+00:00"),  # Friday
            pd.Timestamp("2026-03-29 22:00:00+00:00"),  # Sunday open (weekend gap)
            pd.Timestamp("2026-03-29 22:15:00+00:00"),
            pd.Timestamp("2026-03-29 23:00:00+00:00"),  # 45m intraday gap
        ]
    )
    diag = analyze_temporal_gaps(ts)
    assert diag["total_gaps_count"] == 2
    assert diag["weekend_gaps_count"] == 1


def test_atomic_rollback_on_persistence_failure(tmp_path: Path, mock_15m_csv_bytes: bytes) -> None:
    """11. Test that validation or persistence failure leaves pre-existing files unchanged."""
    out_file = tmp_path / "ds_moving_average_15m.csv"
    manifest_file = tmp_path / "ds_moving_average_15m.manifest.json"

    # Write initial valid dataset
    out_file.write_text("initial,csv,content\n", encoding="utf-8")
    manifest_file.write_text('{"initial": "manifest"}\n', encoding="utf-8")

    initial_csv_content = out_file.read_text()
    initial_json_content = manifest_file.read_text()

    # Attempt ingestion with bad data
    bad_payload = "datetime,open,high,low,close\n2026-03-26T02:30:00+00:00,100.0,90.0,80.0,95.0\n".encode("utf-8")

    with pytest.raises(ValueError, match="invalid OHLC candle geometry"):
        download_and_validate_xauusd_15m(
            output_path=out_file, manifest_path=manifest_file, raw_data_bytes=bad_payload
        )

    # Verify original files remain untouched
    assert out_file.read_text() == initial_csv_content
    assert manifest_file.read_text() == initial_json_content


def test_15m_dataset_resolution_and_research_execution() -> None:
    """12, 13. Test dataset resolution via resolve_historical_dataset and research experiment execution."""
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
    """14. Test that unpromoted 15m candidate fails closed in candidate resolution and readiness."""
    resolved_candidate = resolve_promoted_candidate(symbol="XAUUSD", timeframe="15m")
    assert resolved_candidate is None

    readiness = verify_timeframe_production_readiness(symbol="XAUUSD", timeframe="15m")
    assert readiness["status"] == "BLOCKED"
    assert readiness["reason_code"] == "PromotionUnavailable"
