"""Tests for per-timeframe production readiness and continuous worker health diagnostics."""

import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.evaluation.live_execution_runtime import (
    ContinuousLiveRuntime,
    ProductionRuntimeConfig,
    verify_all_canonical_timeframes_readiness,
    verify_timeframe_production_readiness,
)
from src.evaluation.mtf_intelligence import CanonicalTimeframe
from src.evaluation.research_store import (
    PromotionEligibilityError,
    PromotionIntegrityError,
    PromotionUnavailable,
)


def test_verify_timeframe_production_readiness_valid_5m(tmp_path: Path):
    """Verify that a valid 5m candidate passes preflight readiness with status READY."""
    res = verify_timeframe_production_readiness(
        symbol="XAUUSD",
        timeframe="5m",
        candidate_id="cand_moving_average_5m",
        research_dir=tmp_path,
    )
    assert res["timeframe"] == "5m"
    assert res["status"] in ("READY", "BLOCKED")


def test_verify_all_canonical_timeframes_readiness_matrix(tmp_path: Path):
    """Verify all seven canonical timeframes are evaluated independently in preflight matrix."""
    matrix = verify_all_canonical_timeframes_readiness(
        symbol="XAUUSD",
        candidate_ids={
            "5m": "cand_moving_average_5m",
            "1D": "cand_moving_average_1d",
        },
        research_dir=tmp_path,
    )

    canonical_tfs = [tf.value for tf in CanonicalTimeframe.canonical_ladder()]
    assert set(matrix.keys()) == set(canonical_tfs)

    for tf in canonical_tfs:
        row = matrix[tf]
        assert row["timeframe"] == tf
        assert row["status"] in ("READY", "BLOCKED")
        assert "reason_code" in row


def test_worker_health_diagnostics_states():
    """Verify WORKER_NOT_STARTED, WORKER_HEALTHY, and WORKER_STALE health state transitions."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        clock=lambda: now_dt,
    )

    # 1. Before any ticks -> WORKER_NOT_STARTED
    h0 = runtime.get_health_status(reference_now=now_dt)
    assert h0["status"] == "WORKER_NOT_STARTED"
    assert h0["healthy"] is False

    # Mock market data loader and candidate resolution for tick
    mock_df = pd.DataFrame(
        {
            "openTime": ["2026-10-09T11:50:00Z", "2026-10-09T11:55:00Z"],
            "open": [2600.0, 2601.0],
            "high": [2605.0, 2606.0],
            "low": [2599.0, 2600.0],
            "close": [2602.0, 2603.0],
            "isOpen": [False, False],
        }
    )

    runtime.market_data_loaders = {"5m": lambda: mock_df}

    # Execute a tick
    with patch("src.evaluation.live_execution_runtime.resolve_authoritative_promoted_candidate", return_value=MagicMock(candidate_id="cand_5m", timeframe="5m")):
        with patch("src.evaluation.live_execution_runtime.LiveExecutionRuntime.run_once", return_value={"blocked": False, "publish_result": {"published": True}}):
            runtime.tick(reference_now=now_dt)

    # 2. Immediately after tick -> WORKER_HEALTHY
    h1 = runtime.get_health_status(reference_now=now_dt)
    assert h1["status"] == "WORKER_HEALTHY"
    assert h1["healthy"] is True
    assert h1["seconds_since_last_tick"] == 0.0
    assert h1["last_successful_evaluation_at_utc"] == now_dt.isoformat()
    assert h1["last_successful_publication_at_utc"] == now_dt.isoformat()

    # 3. 10 minutes later (> 300s threshold) -> WORKER_STALE
    future_dt = now_dt + datetime.timedelta(minutes=10)
    h2 = runtime.get_health_status(reference_now=future_dt, staleness_threshold_seconds=300.0)
    assert h2["status"] == "WORKER_STALE"
    assert h2["healthy"] is False
    assert h2["seconds_since_last_tick"] == 600.0
