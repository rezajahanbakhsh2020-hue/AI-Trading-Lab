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


def test_seven_timeframe_canonical_ladder_and_1m_support():
    """Verify exact seven-timeframe ladder including 1m as supported."""
    ladder = CanonicalTimeframe.canonical_ladder()
    ladder_values = [tf.value for tf in ladder]
    assert len(ladder) == 7
    assert ladder_values == ["1m", "5m", "15m", "30m", "1H", "4H", "1D"]
    assert "1m" in ladder_values

    # Test 1m readiness check fails closed to BLOCKED when candidate artifact is missing
    res_1m = verify_timeframe_production_readiness(symbol="XAUUSD", timeframe="1m")
    assert res_1m["timeframe"] == "1m"
    assert res_1m["status"] == "BLOCKED"
    assert res_1m["reason_code"] == "PromotionUnavailable"


def test_1m_readiness_and_non_borrowing_isolation():
    """Verify 1m fails closed when its candidate is missing and does not borrow 5m candidates."""
    matrix = verify_all_canonical_timeframes_readiness(
        symbol="XAUUSD",
        candidate_ids={
            "5m": "cand_moving_average_5m",
            # 1m candidate intentionally missing
        },
    )

    assert matrix["5m"]["status"] == "READY"
    assert matrix["5m"]["candidate_id"] == "cand_moving_average_5m"

    assert matrix["1m"]["status"] == "BLOCKED"
    assert matrix["1m"]["reason_code"] == "PromotionUnavailable"
    assert matrix["1m"]["candidate_id"] is None


def test_canonical_timeframe_casing_normalization():
    """Verify casing/normalization rules (e.g. 1d vs 1D, 4h vs 4H)."""
    # 1D uppercase request with 1d lowercase candidate artifact
    res_1D = verify_timeframe_production_readiness(
        symbol="XAUUSD",
        timeframe="1D",
        candidate_id="cand_moving_average_1d",
    )
    assert res_1D["status"] == "READY"
    assert res_1D["reason_code"] == "READY"
    assert res_1D["candidate_id"] == "cand_moving_average_1d"

    # 1d lowercase request with 1d lowercase candidate artifact
    res_1d = verify_timeframe_production_readiness(
        symbol="XAUUSD",
        timeframe="1d",
        candidate_id="cand_moving_average_1d",
    )
    assert res_1d["status"] == "READY"
    assert res_1d["reason_code"] == "READY"

    # Check CanonicalTimeframe.from_str normalization
    assert CanonicalTimeframe.from_str("1d") == CanonicalTimeframe.ONE_DAY
    assert CanonicalTimeframe.from_str("1D") == CanonicalTimeframe.ONE_DAY
    assert CanonicalTimeframe.from_str("4h") == CanonicalTimeframe.FOUR_HOURS
    assert CanonicalTimeframe.from_str("4H") == CanonicalTimeframe.FOUR_HOURS


def test_independent_per_timeframe_readiness_matrix():
    """Verify independent per-timeframe evaluation across the seven canonical timeframes."""
    matrix = verify_all_canonical_timeframes_readiness(
        symbol="XAUUSD",
        candidate_ids={
            "5m": "cand_moving_average_5m",
            "1D": "cand_moving_average_1d",
        },
    )

    canonical_tfs = [tf.value for tf in CanonicalTimeframe.canonical_ladder()]
    assert len(canonical_tfs) == 7
    assert set(matrix.keys()) == set(canonical_tfs)

    # 5m and 1D are READY
    assert matrix["5m"]["status"] == "READY"
    assert matrix["5m"]["candidate_id"] == "cand_moving_average_5m"

    assert matrix["1D"]["status"] == "READY"
    assert matrix["1D"]["candidate_id"] == "cand_moving_average_1d"

    # Unpromoted timeframes (15m, 30m, 1H, 4H) must be BLOCKED with PromotionUnavailable
    for unpromoted_tf in ("15m", "30m", "1H", "4H"):
        row = matrix[unpromoted_tf]
        assert row["status"] == "BLOCKED"
        assert row["reason_code"] == "PromotionUnavailable"
        assert row["candidate_id"] is None


def test_mismatched_candidate_scope_rejection():
    """Verify candidate from another timeframe is rejected without borrowing or automatic substitution."""
    # Attempting to use a 5m candidate for 15m timeframe
    res = verify_timeframe_production_readiness(
        symbol="XAUUSD",
        timeframe="15m",
        candidate_id="cand_moving_average_5m",
    )
    assert res["status"] == "BLOCKED"
    assert res["reason_code"] in ("TIMEFRAME_IDENTITY_MISMATCH", "PromotionEligibilityError")


def test_render_worker_configuration_truthfulness():
    """Verify render.yaml worker configuration matches verified READY timeframes without unverified mappings."""
    render_yaml_path = Path("render.yaml")
    assert render_yaml_path.exists()
    content = render_yaml_path.read_text(encoding="utf-8")

    # Worker configured for all seven canonical timeframes
    assert "--timeframes 1m,5m,15m,30m,1H,4H,1D" in content
    assert "--candidate-id 5m=cand_moving_average_5m,1D=cand_moving_average_1d" in content


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


def test_worker_health_diagnoses_degraded_execution():
    """Verify worker health reports WORKER_DEGRADED when tick heartbeat is fresh but evaluations fail or are all blocked."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        clock=lambda: now_dt,
    )

    # 1. Fresh heartbeat but all-blocked evaluation
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

    with patch("src.evaluation.live_execution_runtime.resolve_authoritative_promoted_candidate", return_value=None):
        tick_summary = runtime.tick(reference_now=now_dt)

    # Immediate health check on the tick summary reflects current tick without 1-tick delay
    assert tick_summary["health"]["status"] == "WORKER_DEGRADED"
    assert tick_summary["health"]["heartbeat_healthy"] is True
    assert tick_summary["health"]["execution_healthy"] is False
    assert tick_summary["health"]["healthy"] is False

    h_blocked = runtime.get_health_status(reference_now=now_dt)
    assert h_blocked["status"] == "WORKER_DEGRADED"
    assert h_blocked["heartbeat_healthy"] is True
    assert h_blocked["execution_healthy"] is False
    assert h_blocked["healthy"] is False

    # 2. Normal deduplication is NOT treated as hard failure
    with patch("src.evaluation.live_execution_runtime.resolve_authoritative_promoted_candidate", return_value=MagicMock(candidate_id="cand_5m", timeframe="5m")):
        with patch("src.evaluation.live_execution_runtime.LiveExecutionRuntime.run_once", return_value={"blocked": False, "publish_result": {"published": True}}):
            runtime.tick(reference_now=now_dt)

    # Subsequent tick with same candle -> SKIPPED_DEDUPLICATED
    with patch("src.evaluation.live_execution_runtime.resolve_authoritative_promoted_candidate", return_value=MagicMock(candidate_id="cand_5m", timeframe="5m")):
        runtime.tick(reference_now=now_dt)

    h_dedup = runtime.get_health_status(reference_now=now_dt)
    assert h_dedup["status"] == "WORKER_HEALTHY"
    assert h_dedup["heartbeat_healthy"] is True
    assert h_dedup["execution_healthy"] is True
    assert h_dedup["healthy"] is True


def test_worker_health_stale_heartbeat_overrides_execution_health():
    """Verify WORKER_STALE takes precedence over execution health when heartbeat exceeds threshold."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        clock=lambda: now_dt,
    )

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

    with patch("src.evaluation.live_execution_runtime.resolve_authoritative_promoted_candidate", return_value=MagicMock(candidate_id="cand_5m", timeframe="5m")):
        with patch("src.evaluation.live_execution_runtime.LiveExecutionRuntime.run_once", return_value={"blocked": False, "publish_result": {"published": True}}):
            runtime.tick(reference_now=now_dt)

    future_dt = now_dt + datetime.timedelta(minutes=10)
    h_stale = runtime.get_health_status(reference_now=future_dt, staleness_threshold_seconds=300.0)
    assert h_stale["status"] == "WORKER_STALE"
    assert h_stale["heartbeat_healthy"] is False
    assert h_stale["execution_healthy"] is True
    assert h_stale["healthy"] is False


def test_empty_or_unknown_or_malformed_evaluations_are_degraded():
    """Verify empty ({}), unknown status, or malformed evaluation evidence is classified as degraded."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        clock=lambda: now_dt,
    )
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    # 1. Explicit empty current_evaluations mapping
    h_empty = runtime.get_health_status(reference_now=now_dt, current_evaluations={})
    assert h_empty["status"] == "WORKER_DEGRADED"
    assert h_empty["heartbeat_healthy"] is True
    assert h_empty["execution_healthy"] is False

    # 2. Unknown / unrecognized status without positive evidence
    h_unknown = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": {"status": "UNKNOWN_CUSTOM_STATUS"}})
    assert h_unknown["status"] == "WORKER_DEGRADED"
    assert h_unknown["execution_healthy"] is False

    # 3. Malformed / non-dict evaluation object
    h_malformed = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": "string_not_dict"})
    assert h_malformed["status"] == "WORKER_DEGRADED"
    assert h_malformed["execution_healthy"] is False

    # 4. blocked=False alone without positive evidence shape or status
    h_blocked_false_no_evidence = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": {"blocked": False}})
    assert h_blocked_false_no_evidence["status"] == "WORKER_DEGRADED"
    assert h_blocked_false_no_evidence["execution_healthy"] is False


def test_no_closed_candle_is_a_recognized_benign_skip():
    """Verify SKIPPED_NO_CLOSED_CANDLE is a recognized benign skip and classified as execution healthy."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        clock=lambda: now_dt,
    )
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    h_nocandle = runtime.get_health_status(
        reference_now=now_dt,
        current_evaluations={"5m": {"status": "SKIPPED_NO_CLOSED_CANDLE", "reason": "open_candle"}},
    )
    assert h_nocandle["status"] == "WORKER_HEALTHY"
    assert h_nocandle["heartbeat_healthy"] is True
    assert h_nocandle["execution_healthy"] is True


def test_mixed_valid_and_failed_timeframes_policy():
    """Verify mixed timeframe outcomes with at least 1 valid operational outcome are healthy."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m", "15m"],
        clock=lambda: now_dt,
    )
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    mixed_evals = {
        "5m": {"blocked": False, "status": "EVALUATED"},
        "15m": {"blocked": True, "status": "BLOCKED"},
    }
    h_mixed = runtime.get_health_status(reference_now=now_dt, current_evaluations=mixed_evals)
    assert h_mixed["status"] == "WORKER_HEALTHY"
    assert h_mixed["execution_healthy"] is True
