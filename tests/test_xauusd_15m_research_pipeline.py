"""
Regression and forensic tests for XAUUSD 15m dataset ingestion, research execution,
and fail-closed production readiness gating.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.ingest_xauusd_15m import download_and_validate_xauusd_15m
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


def test_ingest_xauusd_15m_valid_download(tmp_path: Path, mock_15m_csv_bytes: bytes) -> None:
    """Test downloading and validating a valid 15m CSV payload."""
    out_file = tmp_path / "ds_moving_average_15m.csv"

    class MockResponse:
        def read(self) -> bytes:
            return mock_15m_csv_bytes

        def __enter__(self) -> MockResponse:
            return self

        def __exit__(self, *args) -> None:
            pass

    with patch("urllib.request.urlopen", return_value=MockResponse()):
        p, meta = download_and_validate_xauusd_15m(output_path=out_file)

    assert p.exists()
    assert meta["row_count"] == 5
    assert meta["start_timestamp_utc"] == "2026-03-26T02:30:00+00:00"
    assert meta["end_timestamp_utc"] == "2026-03-26T03:30:00+00:00"


def test_ingest_xauusd_15m_non_monotonic_fails(tmp_path: Path) -> None:
    """Test that non-monotonic timestamps raise ValueError."""
    bad_data = (
        "datetime,open,high,low,close,volume\n"
        "2026-03-26T02:45:00+00:00,100.0,105.0,99.0,102.0,1000\n"
        "2026-03-26T02:30:00+00:00,102.0,106.0,101.0,104.0,1100\n"
    ).encode("utf-8")

    class MockResponse:
        def read(self) -> bytes:
            return bad_data

        def __enter__(self) -> MockResponse:
            return self

        def __exit__(self, *args) -> None:
            pass

    out_file = tmp_path / "ds_moving_average_15m.csv"
    with patch("urllib.request.urlopen", return_value=MockResponse()):
        with pytest.raises(ValueError, match="timestamps are not monotonic increasing"):
            download_and_validate_xauusd_15m(output_path=out_file)


def test_ingest_xauusd_15m_duplicate_timestamps_fail(tmp_path: Path) -> None:
    """Test that duplicate timestamps raise ValueError."""
    bad_data = (
        "datetime,open,high,low,close,volume\n"
        "2026-03-26T02:30:00+00:00,100.0,105.0,99.0,102.0,1000\n"
        "2026-03-26T02:30:00+00:00,102.0,106.0,101.0,104.0,1100\n"
    ).encode("utf-8")

    class MockResponse:
        def read(self) -> bytes:
            return bad_data

        def __enter__(self) -> MockResponse:
            return self

        def __exit__(self, *args) -> None:
            pass

    out_file = tmp_path / "ds_moving_average_15m.csv"
    with patch("urllib.request.urlopen", return_value=MockResponse()):
        with pytest.raises(ValueError, match="duplicate timestamps"):
            download_and_validate_xauusd_15m(output_path=out_file)


def test_ingest_xauusd_15m_invalid_ohlc_geometry_fails(tmp_path: Path) -> None:
    """Test that invalid OHLC geometry (e.g. high < open) raises ValueError."""
    bad_data = (
        "datetime,open,high,low,close,volume\n"
        "2026-03-26T02:30:00+00:00,100.0,90.0,80.0,95.0,1000\n"  # high (90) < open (100)
    ).encode("utf-8")

    class MockResponse:
        def read(self) -> bytes:
            return bad_data

        def __enter__(self) -> MockResponse:
            return self

        def __exit__(self, *args) -> None:
            pass

    out_file = tmp_path / "ds_moving_average_15m.csv"
    with patch("urllib.request.urlopen", return_value=MockResponse()):
        with pytest.raises(ValueError, match="invalid OHLC candle geometry"):
            download_and_validate_xauusd_15m(output_path=out_file)


def test_15m_dataset_resolution_and_research_execution() -> None:
    """Test dataset resolution and research execution using ingested 15m dataset."""
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
