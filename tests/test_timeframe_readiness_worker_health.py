"""Tests for per-timeframe production readiness and continuous worker health diagnostics."""

import datetime
from pathlib import Path
from typing import Any
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


# Producer-grounded fixtures
def make_real_run_once_success(*, decision: str = "BUY", publish_result: dict[str, Any] | None = None, candidate_id: str = "cand_moving_average_5m", timeframe: str = "5m") -> dict[str, Any]:
    """Construct a complete, real run_once success dictionary matching producer guarantees."""
    if publish_result is None:
        publish_result = {
            "status": "PUBLISHED",
            "published": True,
            "http_code": 200,
            "event_id": "evt_123",
            "publication_id": "evt_123",
            "remote_event_id": "evt_123",
            "attempts": 1,
        }
    return {
        "blocked": False,
        "symbol": "XAUUSD",
        "interval": timeframe,
        "timeframe": timeframe,
        "decision": decision,
        "strategy": "cand_moving_average_5m",
        "stability_score": 0.85,
        "candidate_id": candidate_id,
        "evidence_id": "ev_123",
        "research_fingerprint": "fp_123",
        "strategy_version": "1.0.0",
        "current_lifecycle_state": "PERSISTED",
        "record": {
            "decision_id": "dec_123",
            "signal_id": "sig_123",
            "decision": decision,
            "symbol": "XAUUSD",
            "interval": "5m",
        },
        "publication": {
            "contract_version": "1.0",
            "event_id": "evt_123",
            "timestamp": "2026-10-09T12:00:00Z",
        },
        "contract_payload": {
            "contract_version": "1.0",
            "event_id": "evt_123",
        },
        "publish_result": publish_result,
    }


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
    res = verify_timeframe_production_readiness(
        symbol="XAUUSD",
        timeframe="15m",
        candidate_id="cand_moving_average_5m",
    )
    assert res["status"] == "BLOCKED"
    assert res["reason_code"] in ("TIMEFRAME_IDENTITY_MISMATCH", "PromotionEligibilityError")


def test_render_worker_configuration_truthfulness():
    """Verify render.yaml worker configuration matches verified canonical seven-timeframe ladder with 1m."""
    render_yaml_path = Path("render.yaml")
    assert render_yaml_path.exists()
    content = render_yaml_path.read_text(encoding="utf-8")

    assert "--timeframes 1m,5m,15m,30m,1H,4H,1D" in content
    assert "--candidate-id 5m=cand_moving_average_5m,1D=cand_moving_average_1d" in content


# ==============================================================================
# WORKER HEALTH DIAGNOSTICS REGRESSION TEST MATRIX
# ==============================================================================

# --- 1. Worker Skip Validation ---

def test_exact_real_skipped_deduplicated_shape_is_accepted():
    """1. Exact real SKIPPED_DEDUPLICATED shape is accepted."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    valid_dedup = {
        "status": "SKIPPED_DEDUPLICATED",
        "reason": "candle_already_evaluated",
        "candle_timestamp": "2026-10-09T11:55:00Z",
        "candidate_id": "cand_5m",
        "timeframe": "5m",
    }
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": valid_dedup})
    assert h["status"] == "WORKER_HEALTHY"
    assert h["execution_healthy"] is True


def test_exact_real_skipped_no_closed_candle_shape_is_accepted():
    """2. Exact real SKIPPED_NO_CLOSED_CANDLE shape is accepted."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        candidate_ids={"5m": "cand_moving_average_5m"},
        clock=lambda: now_dt,
    )
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    valid_nocandle = {
        "status": "SKIPPED_NO_CLOSED_CANDLE",
        "reason": "latest_candle_is_open_or_unavailable",
        "symbol": "XAUUSD",
        "timeframe": "5m",
    }
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": valid_nocandle})
    assert h["status"] == "WORKER_HEALTHY"
    assert h["execution_healthy"] is True


def test_missing_skip_reason_rejected():
    """3. Missing skip reason is rejected."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    missing_reason = {"status": "SKIPPED_DEDUPLICATED"}
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": missing_reason})
    assert h["status"] == "WORKER_DEGRADED"
    assert h["execution_healthy"] is False


def test_valid_skip_status_with_contradictory_reason_rejected():
    """4. A valid skip status with a contradictory reason is rejected."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    contradictory = {
        "status": "SKIPPED_NO_CLOSED_CANDLE",
        "reason": "candle_already_evaluated",
    }
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": contradictory})
    assert h["status"] == "WORKER_DEGRADED"
    assert h["execution_healthy"] is False


def test_reason_only_result_without_status_rejected():
    """5. A reason-only result without a recognized status is rejected."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    reason_only = {"reason": "latest_candle_is_open_or_unavailable"}
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": reason_only})
    assert h["status"] == "WORKER_DEGRADED"
    assert h["execution_healthy"] is False


def test_malformed_required_identity_in_skip_rejected():
    """6. Malformed required identity field in skip is rejected."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    malformed_id = {
        "status": "SKIPPED_DEDUPLICATED",
        "reason": "candle_already_evaluated",
        "candle_timestamp": 123456789,  # Should be ISO string, not integer
    }
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": malformed_id})
    assert h["status"] == "WORKER_DEGRADED"
    assert h["execution_healthy"] is False


def test_failure_or_error_field_cannot_be_overridden_by_skip_label():
    """7. A failure/error field cannot be overridden by a benign skip label."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    error_in_skip = {
        "status": "SKIPPED_DEDUPLICATED",
        "reason": "candle_already_evaluated",
        "error": "Unexpected exception during tick",
    }
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": error_in_skip})
    assert h["status"] == "WORKER_DEGRADED"
    assert h["execution_healthy"] is False


# --- 2. Successful Evaluation Validation ---

def test_real_successful_run_once_when_publication_disabled():
    """8. Real successful run_once result when publication is disabled."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    pub_disabled = {
        "status": "SKIPPED_DISABLED",
        "published": False,
        "reason": "Publisher disabled in configuration",
    }
    res = make_real_run_once_success(publish_result=pub_disabled)
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": res})
    assert h["status"] == "WORKER_HEALTHY"
    assert h["execution_healthy"] is True


def test_real_successful_run_once_when_publication_not_requested():
    """9. Real successful run_once result when publication is not requested (publish_result is None)."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    res = make_real_run_once_success(publish_result=None)
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": res})
    assert h["status"] == "WORKER_HEALTHY"
    assert h["execution_healthy"] is True


def test_real_successful_result_with_accepted_publisher_ack():
    """10. Real successful result with an accepted publisher acknowledgement."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    res = make_real_run_once_success()
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": res})
    assert h["status"] == "WORKER_HEALTHY"
    assert h["execution_healthy"] is True


def test_blocked_false_alone_is_unhealthy():
    """11. {'blocked': False} alone is unhealthy."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": {"blocked": False}})
    assert h["status"] == "WORKER_DEGRADED"
    assert h["execution_healthy"] is False


def test_blocked_false_with_none_record_rejected():
    """12. {'blocked': False, 'record': None} is not accepted solely because the key exists."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    res = {"blocked": False, "record": None, "publication": None, "contract_payload": None}
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": res})
    assert h["status"] == "WORKER_DEGRADED"
    assert h["execution_healthy"] is False


def test_blocked_false_with_none_contract_payload_rejected():
    """13. {'blocked': False, 'contract_payload': None} is not accepted solely because key exists."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    res = {"blocked": False, "contract_payload": None}
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": res})
    assert h["status"] == "WORKER_DEGRADED"
    assert h["execution_healthy"] is False


def test_synthetic_published_true_without_evidence_rejected():
    """14. {'blocked': False, 'publish_result': {'published': True}} without evidence is rejected."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    res = {"blocked": False, "publish_result": {"published": True}}
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": res})
    assert h["status"] == "WORKER_DEGRADED"
    assert h["execution_healthy"] is False


def test_unknown_empty_non_dict_contradictory_shapes_fail_closed():
    """15. Unknown, empty, non-dictionary, contradictory, and malformed result shapes fail closed."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    for bad_shape in ({}, [], "string", None, 123, {"status": "UNKNOWN_ABC"}, {"blocked": True, "record": {"decision": "BUY"}}):
        h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": bad_shape})
        assert h["status"] == "WORKER_DEGRADED", f"Failed for shape: {bad_shape}"
        assert h["execution_healthy"] is False


def test_real_no_trade_skipped_no_trade_is_healthy():
    """16. A real no-trade outcome with intentional SKIPPED_NO_TRADE behavior is healthy."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    pub_no_trade = {
        "status": "SKIPPED_NO_TRADE",
        "published": False,
        "reason": "Decision is NO TRADE and skip_if_no_trade=True",
    }
    res = make_real_run_once_success(decision="NO TRADE", publish_result=pub_no_trade)
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": res})
    assert h["status"] == "WORKER_HEALTHY"
    assert h["execution_healthy"] is True


# --- 3. Publication Outcomes ---

def test_valid_published_result_accepted():
    """17. Valid PUBLISHED result with published=True is accepted."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    res = make_real_run_once_success()
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": res})
    assert h["status"] == "WORKER_HEALTHY"
    assert h["execution_healthy"] is True


@pytest.mark.parametrize("failure_status", [
    "FAILED",
    "REJECTED",
    "AUTH_FAILED",
    "FORBIDDEN",
    "UNAVAILABLE",
    "TIMED_OUT",
    "INVALID_RESPONSE",
    "SKIPPED_BLOCKED",
    "MISCONFIGURED",
])
def test_publisher_failure_statuses_are_unhealthy_when_publication_required(failure_status: str):
    """18. Each actual publication failure status is unhealthy when publication is required."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    pub_failed = {
        "status": failure_status,
        "published": False,
        "error": f"Simulated {failure_status}",
    }
    res = make_real_run_once_success(publish_result=pub_failed)
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": res})
    assert h["status"] == "WORKER_DEGRADED"
    assert h["execution_healthy"] is False


def test_success_looking_status_with_published_false_rejected():
    """19. A success-looking status with published=False is rejected."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    contradictory_pub = {
        "status": "PUBLISHED",
        "published": False,
    }
    res = make_real_run_once_success(publish_result=contradictory_pub)
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": res})
    assert h["status"] == "WORKER_DEGRADED"
    assert h["execution_healthy"] is False


def test_published_true_paired_with_failure_status_rejected():
    """20. published=True paired with a failure status is rejected as contradictory."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    contradictory_pub = {
        "status": "FAILED",
        "published": True,
    }
    res = make_real_run_once_success(publish_result=contradictory_pub)
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": res})
    assert h["status"] == "WORKER_DEGRADED"
    assert h["execution_healthy"] is False


def test_missing_or_malformed_publication_status_rejected():
    """21. Missing/malformed publication status is not accepted as successful delivery."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    bad_pub = {"published": True}  # Missing status field
    res = make_real_run_once_success(publish_result=bad_pub)
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": res})
    assert h["status"] == "WORKER_DEGRADED"
    assert h["execution_healthy"] is False


def test_skipped_disabled_handled_correctly():
    """22. SKIPPED_DISABLED is handled correctly when publisher is disabled."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    pub_disabled = {
        "status": "SKIPPED_DISABLED",
        "published": False,
        "reason": "Publisher disabled in configuration",
    }
    res = make_real_run_once_success(publish_result=pub_disabled)
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": res})
    assert h["status"] == "WORKER_HEALTHY"
    assert h["execution_healthy"] is True


def test_publication_failure_degrades_end_to_end_health():
    """23. Publication failure degrades end-to-end worker health despite valid local decision."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    pub_failed = {
        "status": "FAILED",
        "published": False,
        "error": "Connection refused by Project 2 gateway",
    }
    res = make_real_run_once_success(publish_result=pub_failed)
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={"5m": res})
    assert h["status"] == "WORKER_DEGRADED"
    assert h["execution_healthy"] is False


# --- 4. Aggregate Health and Heartbeat ---

def test_all_timeframes_blocked_or_failed_yields_worker_degraded():
    """24. All timeframe outcomes blocked/failed => WORKER_DEGRADED with fresh heartbeat."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m", "15m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    all_blocked = {
        "5m": {"blocked": True, "status": "BLOCKED", "reason": "PromotionUnavailable"},
        "15m": {"status": "FAILED_RECOVERABLE", "error": "Provider offline"},
    }
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations=all_blocked)
    assert h["status"] == "WORKER_DEGRADED"
    assert h["heartbeat_healthy"] is True
    assert h["execution_healthy"] is False


def test_one_healthy_timeframe_plus_one_blocked_timeframe_is_healthy():
    """25. One genuinely healthy timeframe plus another blocked timeframe preserves healthy-at-least-one policy."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m", "15m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    mixed = {
        "5m": make_real_run_once_success(),
        "15m": {"blocked": True, "status": "BLOCKED", "reason": "PromotionUnavailable"},
    }
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations=mixed)
    assert h["status"] == "WORKER_HEALTHY"
    assert h["execution_healthy"] is True


def test_empty_current_tick_evaluations_yields_worker_degraded():
    """26. Empty current-tick evaluations => WORKER_DEGRADED, with no history fallback."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()
    runtime._execution_history = [{"evaluations": {"5m": make_real_run_once_success()}}]

    # Explicit current_evaluations={} must NOT fall back to history
    h = runtime.get_health_status(reference_now=now_dt, current_evaluations={})
    assert h["status"] == "WORKER_DEGRADED"
    assert h["execution_healthy"] is False


def test_omitted_current_evaluations_uses_latest_history():
    """27. Omitted current evaluations may use the latest history according to established contract."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()
    runtime._execution_history = [{"evaluations": {"5m": make_real_run_once_success()}}]

    h = runtime.get_health_status(reference_now=now_dt, current_evaluations=None)
    assert h["status"] == "WORKER_HEALTHY"
    assert h["execution_healthy"] is True


def test_stale_heartbeat_overrides_healthy_execution():
    """28. Stale heartbeat overrides healthy execution => WORKER_STALE."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    future_dt = now_dt + datetime.timedelta(minutes=10)
    h = runtime.get_health_status(
        reference_now=future_dt,
        staleness_threshold_seconds=300.0,
        current_evaluations={"5m": make_real_run_once_success()},
    )
    assert h["status"] == "WORKER_STALE"
    assert h["heartbeat_healthy"] is False
    assert h["execution_healthy"] is True


def test_malformed_and_future_heartbeat_timestamps_are_unhealthy():
    """29. Malformed and future heartbeat timestamps are not healthy."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()

    # Malformed timestamp
    runtime._last_tick_at_utc = "invalid-iso-string"
    h_malformed = runtime.get_health_status(reference_now=now_dt)
    assert h_malformed["status"] == "WORKER_STALE"
    assert h_malformed["heartbeat_healthy"] is False

    # Future timestamp relative to reference_now
    runtime._last_tick_at_utc = (now_dt + datetime.timedelta(hours=1)).isoformat()
    h_future = runtime.get_health_status(reference_now=now_dt)
    assert h_future["status"] == "WORKER_STALE"
    assert h_future["heartbeat_healthy"] is False


def test_worker_not_started_status_is_deterministic():
    """30. WORKER_NOT_STARTED remains deterministic."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)

    h0 = runtime.get_health_status(reference_now=now_dt)
    assert h0["status"] == "WORKER_NOT_STARTED"
    assert h0["healthy"] is False


def test_same_tick_health_reflects_current_tick_without_delay():
    """31. Same-tick health reflects the current tick without a one-tick delay."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)

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

    real_res = make_real_run_once_success()
    with patch("src.evaluation.live_execution_runtime.resolve_authoritative_promoted_candidate", return_value=MagicMock(candidate_id="cand_moving_average_5m", timeframe="5m")):
        with patch("src.evaluation.live_execution_runtime.LiveExecutionRuntime.run_once", return_value=real_res):
            tick_summary = runtime.tick(reference_now=now_dt)

    assert tick_summary["health"]["status"] == "WORKER_HEALTHY"
    assert tick_summary["health"]["execution_healthy"] is True


def test_repeated_deduplicated_candles_remain_operationally_healthy():
    """32. Repeated deduplicated candles remain operationally healthy when exact deduplication contract is satisfied."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)

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

    real_res = make_real_run_once_success()
    with patch("src.evaluation.live_execution_runtime.resolve_authoritative_promoted_candidate", return_value=MagicMock(candidate_id="cand_moving_average_5m", timeframe="5m")):
        with patch("src.evaluation.live_execution_runtime.LiveExecutionRuntime.run_once", return_value=real_res):
            runtime.tick(reference_now=now_dt)

    # Second tick with same candle -> SKIPPED_DEDUPLICATED
    with patch("src.evaluation.live_execution_runtime.resolve_authoritative_promoted_candidate", return_value=MagicMock(candidate_id="cand_moving_average_5m", timeframe="5m")):
        tick_summary_2 = runtime.tick(reference_now=now_dt)

    assert tick_summary_2["health"]["status"] == "WORKER_HEALTHY"
    assert tick_summary_2["health"]["execution_healthy"] is True


def test_multi_tick_failed_publication_then_recovery_sequence():
    """Verify that a tick with failed publication is not deduplicated and retries until healthy."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)

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

    # Tick 1: Publication fails
    failed_pub_res = {
        "status": "FAILED",
        "published": False,
        "error": "HTTP 503 Service Unavailable",
    }
    tick1_res = make_real_run_once_success(publish_result=failed_pub_res)

    with patch("src.evaluation.live_execution_runtime.resolve_authoritative_promoted_candidate", return_value=MagicMock(candidate_id="cand_moving_average_5m", timeframe="5m")):
        with patch("src.evaluation.live_execution_runtime.LiveExecutionRuntime.run_once", return_value=tick1_res) as mock_run_once:
            tick_summary_1 = runtime.tick(reference_now=now_dt)
            assert mock_run_once.call_count == 1

    # Assert Tick 1 is WORKER_DEGRADED due to publication failure
    assert tick_summary_1["health"]["status"] == "WORKER_DEGRADED"
    assert tick_summary_1["health"]["execution_healthy"] is False

    # Assert key was NOT added to deduplication set
    dedup_key = ("XAUUSD", "5m", "2026-10-09T11:55:00+00:00", "cand_moving_average_5m")
    assert dedup_key not in runtime._evaluated_candles

    # Tick 2: Same candle, publication now succeeds via recovery
    successful_pub_res = {
        "status": "PUBLISHED",
        "published": True,
        "http_code": 200,
        "event_id": "evt_123",
    }
    tick2_res = make_real_run_once_success(publish_result=successful_pub_res)

    with patch("src.evaluation.live_execution_runtime.resolve_authoritative_promoted_candidate", return_value=MagicMock(candidate_id="cand_moving_average_5m", timeframe="5m")):
        with patch("src.evaluation.live_execution_runtime.LiveExecutionRuntime.run_once", return_value=tick2_res) as mock_run_once_2:
            tick_summary_2 = runtime.tick(reference_now=now_dt)
            # Must run run_once again (not skipped as deduplicated)
            assert mock_run_once_2.call_count == 1

    # Assert Tick 2 is WORKER_HEALTHY
    assert tick_summary_2["health"]["status"] == "WORKER_HEALTHY"
    assert tick_summary_2["health"]["execution_healthy"] is True

    # Assert key IS now added to deduplication set
    assert dedup_key in runtime._evaluated_candles

    # Tick 3: Same candle -> Now cleanly deduplicated
    with patch("src.evaluation.live_execution_runtime.resolve_authoritative_promoted_candidate", return_value=MagicMock(candidate_id="cand_moving_average_5m", timeframe="5m")):
        with patch("src.evaluation.live_execution_runtime.LiveExecutionRuntime.run_once") as mock_run_once_3:
            tick_summary_3 = runtime.tick(reference_now=now_dt)
            # run_once should NOT be called on tick 3
            assert mock_run_once_3.call_count == 0

    assert tick_summary_3["evaluations"]["5m"]["status"] == "SKIPPED_DEDUPLICATED"
    assert tick_summary_3["health"]["status"] == "WORKER_HEALTHY"


def test_multi_tick_persistent_publication_failure_remains_degraded():
    """Verify that persistent publication failure prevents false-positive deduplicated health."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)

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

    failed_pub_res = {
        "status": "FAILED",
        "published": False,
        "error": "HTTP 500 Internal Server Error",
    }
    tick_res = make_real_run_once_success(publish_result=failed_pub_res)

    # Tick 1
    with patch("src.evaluation.live_execution_runtime.resolve_authoritative_promoted_candidate", return_value=MagicMock(candidate_id="cand_moving_average_5m", timeframe="5m")):
        with patch("src.evaluation.live_execution_runtime.LiveExecutionRuntime.run_once", return_value=tick_res):
            tick_1 = runtime.tick(reference_now=now_dt)

    assert tick_1["health"]["status"] == "WORKER_DEGRADED"

    # Tick 2: Same candle, publication fails again
    with patch("src.evaluation.live_execution_runtime.resolve_authoritative_promoted_candidate", return_value=MagicMock(candidate_id="cand_moving_average_5m", timeframe="5m")):
        with patch("src.evaluation.live_execution_runtime.LiveExecutionRuntime.run_once", return_value=tick_res) as mock_run_once_2:
            tick_2 = runtime.tick(reference_now=now_dt)
            assert mock_run_once_2.call_count == 1

    assert tick_2["health"]["status"] == "WORKER_DEGRADED"
    assert tick_2["evaluations"]["5m"].get("status") != "SKIPPED_DEDUPLICATED"
    assert tick_2["evaluations"]["5m"]["blocked"] is False


def test_structured_per_timeframe_health_diagnostics():
    """Verify get_health_status includes structured per-timeframe health diagnostics."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m", "1m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()
    runtime._timeframe_readiness = {
        "5m": {"status": "READY", "candidate_id": "cand_5m", "reason_code": "READY"},
        "1m": {"status": "BLOCKED", "candidate_id": None, "reason_code": "PromotionUnavailable"},
    }

    evals = {
        "5m": make_real_run_once_success(candidate_id="cand_5m", timeframe="5m"),
        "1m": {"blocked": True, "status": "BLOCKED", "reason": "PromotionUnavailable"},
    }

    h = runtime.get_health_status(reference_now=now_dt, current_evaluations=evals)
    assert h["status"] == "WORKER_HEALTHY"
    assert "timeframes" in h
    tf_diag = h["timeframes"]

    assert "5m" in tf_diag
    assert tf_diag["5m"]["readiness"] == "READY"
    assert tf_diag["5m"]["candidate_id"] == "cand_5m"
    assert tf_diag["5m"]["execution"] == "HEALTHY"
    assert tf_diag["5m"]["publication_status"] == "PUBLISHED"

    assert "1m" in tf_diag
    assert tf_diag["1m"]["readiness"] == "BLOCKED"
    assert tf_diag["1m"]["reason_code"] == "PromotionUnavailable"
    assert tf_diag["1m"]["execution"] == "BLOCKED"


def test_diagnostic_precedence_runtime_failure_overrides_ready_preflight():
    """Verify runtime evaluation error overrides READY preflight status in per-timeframe health diagnostics."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()
    runtime._timeframe_readiness = {
        "5m": {"status": "READY", "candidate_id": "cand_5m", "reason_code": "READY"},
    }

    # Current evaluation failed with provider error
    evals = {
        "5m": {"status": "FAILED_RECOVERABLE", "error": "Provider connection refused"},
    }

    h = runtime.get_health_status(reference_now=now_dt, current_evaluations=evals)
    assert h["status"] == "WORKER_DEGRADED"
    assert h["timeframes"]["5m"]["reason_code"] == "Provider connection refused"
    assert h["timeframes"]["5m"]["execution"] == "UNHEALTHY"


@pytest.mark.parametrize("pub_failure_status", [
    "FAILED",
    "REJECTED",
    "AUTH_FAILED",
    "FORBIDDEN",
    "UNAVAILABLE",
    "TIMED_OUT",
    "INVALID_RESPONSE",
    "SKIPPED_BLOCKED",
    "MISCONFIGURED",
])
def test_diagnostic_precedence_all_publication_failures_override_ready_preflight(pub_failure_status: str):
    """Verify ALL publication failure statuses override READY preflight status in per-timeframe health diagnostics."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()
    runtime._timeframe_readiness = {
        "5m": {"status": "READY", "candidate_id": "cand_5m", "reason_code": "READY"},
    }

    failed_pub = {
        "status": pub_failure_status,
        "published": False,
        "reason": f"Simulated publication failure: {pub_failure_status}",
        "error": f"Error detail for {pub_failure_status}",
    }
    evals = {
        "5m": make_real_run_once_success(publish_result=failed_pub),
    }

    h = runtime.get_health_status(reference_now=now_dt, current_evaluations=evals)
    assert h["status"] == "WORKER_DEGRADED"
    assert h["timeframes"]["5m"]["publication_status"] == pub_failure_status
    assert h["timeframes"]["5m"]["reason_code"] != "READY"
    assert pub_failure_status in h["timeframes"]["5m"]["reason_code"] or "Simulated" in h["timeframes"]["5m"]["reason_code"] or "Error detail" in h["timeframes"]["5m"]["reason_code"]
    assert h["timeframes"]["5m"]["execution"] == "UNHEALTHY"


@pytest.mark.parametrize("malformed_blocked_state", [
    None,
    "string_blocked_state",
    [1, 2, 3],
    12345,
    {"reason": "valid_blocked_dict_reason"},
])
def test_malformed_blocked_state_handled_safely_without_exception(malformed_blocked_state):
    """Verify malformed blocked_state values (None, string, list, int) do not raise AttributeError/TypeError in get_health_status."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()
    runtime._timeframe_readiness = {
        "5m": {"status": "READY", "candidate_id": "cand_5m", "reason_code": "READY"},
    }

    evals = {
        "5m": {
            "blocked": True,
            "status": "BLOCKED",
            "blocked_state": malformed_blocked_state,
            "reason": "fallback_blocked_reason",
        },
    }

    h = runtime.get_health_status(reference_now=now_dt, current_evaluations=evals)
    assert h["status"] == "WORKER_DEGRADED"
    assert h["timeframes"]["5m"]["execution"] == "BLOCKED"
    assert h["timeframes"]["5m"]["reason_code"] != "READY"
    if isinstance(malformed_blocked_state, dict) and "reason" in malformed_blocked_state:
        assert h["timeframes"]["5m"]["reason_code"] == "valid_blocked_dict_reason"
    else:
        assert h["timeframes"]["5m"]["reason_code"] in ("fallback_blocked_reason", "BLOCKED")


@pytest.mark.parametrize("tf", ["1m", "5m", "15m", "30m", "1H", "4H", "1D"])
def test_seven_timeframes_unhealthy_evaluations_never_report_ready(tf: str):
    """Verify for every canonical timeframe that an unhealthy evaluation never reports reason_code='READY'."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    all_tfs = ["1m", "5m", "15m", "30m", "1H", "4H", "1D"]
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=all_tfs, clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()
    runtime._timeframe_readiness = {
        key: {"status": "READY", "candidate_id": f"cand_{key}", "reason_code": "READY"}
        for key in all_tfs
    }

    # Simulate target timeframe failing with runtime error while other timeframes succeed
    evals = {
        key: make_real_run_once_success() if key != tf else {"status": "FAILED", "error": f"Simulated failure for {tf}"}
        for key in all_tfs
    }

    h = runtime.get_health_status(reference_now=now_dt, current_evaluations=evals)
    tf_diag = h["timeframes"][tf]

    assert tf_diag["reason_code"] != "READY"
    assert "Simulated failure" in tf_diag["reason_code"] or tf_diag["reason_code"] == "FAILED"
    assert tf_diag["execution"] == "UNHEALTHY"


def test_promotable_evidence_alone_cannot_bypass_production_checks(tmp_path: Path):
    """Verify that a PROMOTABLE label on evidence alone cannot bypass binding and production checks."""
    from src.evaluation.research_constitution import (
        CodeProvenance,
        DatasetScope,
        EvidencePartition,
        EvidencePartitionRole,
        ExecutionAssumptions,
        PromotionStatus,
        ResearchEvidence,
        ResearchExperimentSpec,
    )
    from src.evaluation.research_store import (
        PromotionEligibilityError,
        persist_promoted_candidate_binding,
        resolve_promoted_candidate,
        save_research_experiment,
    )

    # 1. Evidence is PROMOTABLE, but no candidate binding file exists on disk
    spec = ResearchExperimentSpec(
        hypothesis="Test hypothesis",
        methodology_version="1.0",
        strategy_name="moving_average",
        strategy_version="1.0",
        dataset_scope=DatasetScope(
            dataset_id="ds_test_15m",
            symbol="XAUUSD",
            timeframe="15m",
            start_date="2025-01-01",
            end_date="2025-12-31",
        ),
        execution_assumptions=ExecutionAssumptions(
            transaction_cost=0.001,
            slippage=0.001,
            latency_ms=10.0,
        ),
        code_provenance=CodeProvenance(commit_sha="abc1234"),
        benchmark_reference="buy_and_hold",
        parameters={"fast_window": 20, "slow_window": 50},
    )
    evidence = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(
            EvidencePartition(
                role=EvidencePartitionRole.OUT_OF_SAMPLE,
                start_date="2025-01-01",
                end_date="2025-12-31",
                total_return=0.15,
                max_drawdown=0.05,
                sharpe_ratio=1.8,
            ),
        ),
        robustness_verdict={"passed": True},
        promotion_status=PromotionStatus.PROMOTABLE,
    )

    save_research_experiment(evidence, base_dir=tmp_path)

    # Candidate resolution fails closed (returns None) without candidate.json
    resolved = resolve_promoted_candidate(
        candidate_id="cand_unbound_15m",
        symbol="XAUUSD",
        timeframe="15m",
        base_dir=tmp_path,
    )
    assert resolved is None

    readiness = verify_timeframe_production_readiness(
        symbol="XAUUSD",
        timeframe="15m",
        candidate_id="cand_unbound_15m",
        research_dir=tmp_path,
    )
    assert readiness["status"] == "BLOCKED"
    assert readiness["reason_code"] == "PromotionUnavailable"

    # 2. Evidence is PROMOTABLE and candidate binding exists, BUT evidence fails production promotion policy criteria (negative return)
    failing_spec = ResearchExperimentSpec(
        hypothesis="Failing hypothesis",
        methodology_version="1.0",
        strategy_name="failing_strategy",
        strategy_version="1.0",
        dataset_scope=DatasetScope(
            dataset_id="ds_failing_15m",
            symbol="XAUUSD",
            timeframe="15m",
            start_date="2025-01-01",
            end_date="2025-12-31",
        ),
        execution_assumptions=ExecutionAssumptions(
            transaction_cost=0.001,
            slippage=0.001,
            latency_ms=10.0,
        ),
        code_provenance=CodeProvenance(commit_sha="abc1234"),
        benchmark_reference="buy_and_hold",
        parameters={"fast_window": 20, "slow_window": 50},
    )
    from src.evaluation.research_constitution import RejectionReason

    # Scenario B: Evidence claims PROMOTABLE label but contains rejection reasons (e.g. POOR_PERFORMANCE)
    failing_evidence = ResearchEvidence(
        experiment_fingerprint=failing_spec.fingerprint,
        spec=failing_spec,
        partitions=(
            EvidencePartition(
                role=EvidencePartitionRole.OUT_OF_SAMPLE,
                start_date="2025-01-01",
                end_date="2025-12-31",
                total_return=-0.50,
                max_drawdown=0.50,
                sharpe_ratio=-1.0,
            ),
        ),
        robustness_verdict={"passed": False},
        promotion_status=PromotionStatus.PROMOTABLE,  # Label claims PROMOTABLE
        rejection_reasons=(RejectionReason.FAILED_OOS,),
    )
    save_research_experiment(failing_evidence, base_dir=tmp_path)
    persist_promoted_candidate_binding(
        candidate_id="cand_failing_15m",
        evidence=failing_evidence,
        operational_stability_score=0.80,
        base_dir=tmp_path,
        governance_decision=MagicMock(qualified=True, decision_fingerprint="gov_fp_failing"),
    )

    # Resolution must fail closed with PromotionEligibilityError due to rejection reasons in evidence
    with pytest.raises(PromotionEligibilityError, match="contains rejection reasons"):
        resolve_promoted_candidate(
            candidate_id="cand_failing_15m",
            symbol="XAUUSD",
            timeframe="15m",
            base_dir=tmp_path,
        )

    readiness_failing = verify_timeframe_production_readiness(
        symbol="XAUUSD",
        timeframe="15m",
        candidate_id="cand_failing_15m",
        research_dir=tmp_path,
    )
    assert readiness_failing["status"] == "BLOCKED"
    assert readiness_failing["reason_code"] == "PromotionEligibilityError"


def test_malformed_candidate_json_fails_closed_without_crashing_diagnostics(tmp_path: Path):
    """Verify unparseable or malformed candidate JSON fails closed to BLOCKED without crashing readiness report."""
    by_cand_15m = tmp_path / "by_candidate" / "corrupt_cand_15m"
    by_cand_15m.mkdir(parents=True)
    (by_cand_15m / "candidate.json").write_text("{ unparseable syntax }", encoding="utf-8")

    matrix = verify_all_canonical_timeframes_readiness(
        symbol="XAUUSD",
        candidate_ids={"15m": "corrupt_cand_15m"},
        research_dir=tmp_path,
    )

    canonical_tfs = [tf.value for tf in CanonicalTimeframe.canonical_ladder()]
    assert len(matrix) == 7
    assert set(matrix.keys()) == set(canonical_tfs)

    assert matrix["15m"]["status"] == "BLOCKED"
    assert matrix["15m"]["reason_code"] == "PromotionIntegrityError"

    for tf in canonical_tfs:
        assert tf in matrix
        assert matrix[tf]["status"] in ("READY", "BLOCKED")


def test_identity_mismatched_candidate_binding_fails_closed(tmp_path: Path):
    """Verify identity or scope mismatches in candidate binding fail closed to BLOCKED."""
    from tests.test_continuous_live_runtime import persist_momentum_candidate

    cand_id = persist_momentum_candidate(tmp_path, candidate_id="cand_5m_mismatch", timeframe="5m")

    # 1. Candidate timeframe requested (15m) does not match binding timeframe (5m)
    res_tf = verify_timeframe_production_readiness(
        symbol="XAUUSD",
        timeframe="15m",
        candidate_id=cand_id,
        research_dir=tmp_path,
    )
    assert res_tf["status"] == "BLOCKED"
    assert res_tf["reason_code"] in ("TIMEFRAME_IDENTITY_MISMATCH", "PromotionEligibilityError")

    # 2. Symbol requested (EURUSD) does not match candidate symbol (XAUUSD)
    res_sym = verify_timeframe_production_readiness(
        symbol="EURUSD",
        timeframe="5m",
        candidate_id=cand_id,
        research_dir=tmp_path,
    )
    assert res_sym["status"] == "BLOCKED"
    assert res_sym["reason_code"] == "PromotionEligibilityError"


def test_binding_with_missing_evidence_file_fails_closed(tmp_path: Path):
    """Verify a candidate binding pointing to missing evidence file fails closed deterministically."""
    import json

    by_cand_30m = tmp_path / "by_candidate" / "cand_missing_evidence"
    by_cand_30m.mkdir(parents=True)
    binding = {
        "candidate_id": "cand_missing_evidence",
        "strategy_name": "momentum",
        "strategy_version": "1.0",
        "experiment_fingerprint": "missing_experiment_fp_9999",
        "evidence_id": "ev_missing_9999",
        "symbol": "XAUUSD",
        "timeframe": "30m",
        "parameters": {"window": 10},
        "governance_decision_fingerprint": "gov_fp_missing",
        "operational_stability_score": 0.75,
    }
    (by_cand_30m / "candidate.json").write_text(json.dumps(binding), encoding="utf-8")

    res = verify_timeframe_production_readiness(
        symbol="XAUUSD",
        timeframe="30m",
        candidate_id="cand_missing_evidence",
        research_dir=tmp_path,
    )
    assert res["status"] == "BLOCKED"
    assert res["reason_code"] == "PromotionIntegrityError"
    assert "missing_experiment_fp_9999" in res["detail"]

    matrix = verify_all_canonical_timeframes_readiness(
        symbol="XAUUSD",
        candidate_ids={"30m": "cand_missing_evidence"},
        research_dir=tmp_path,
    )
    assert matrix["30m"]["status"] == "BLOCKED"
    assert matrix["30m"]["reason_code"] == "PromotionIntegrityError"


def test_readiness_distinguished_from_live_evaluation_and_publication(tmp_path: Path):
    """Verify readiness preflight checks candidate authorization without triggering market evaluation or publication delivery."""
    from tests.test_continuous_live_runtime import make_tf_market_data, persist_momentum_candidate

    cand_id = persist_momentum_candidate(tmp_path, candidate_id="cand_preflight_test", timeframe="5m")

    pub_mock = MagicMock(enabled=True, publish_url="https://example.com/ingest", api_key="secret")

    with patch("src.evaluation.live_execution_runtime.create_live_market_evaluation") as mock_create_eval, \
         patch("src.evaluation.live_execution_runtime.evaluate_authorized_live_runtime") as mock_eval_runtime:

        # 1. Timeframe readiness preflight
        res = verify_timeframe_production_readiness(
            symbol="XAUUSD",
            timeframe="5m",
            candidate_id=cand_id,
            research_dir=tmp_path,
            publisher=pub_mock,
        )

        assert res["status"] == "READY"
        assert res["reason_code"] == "READY"
        assert "authorization_fingerprint" in res

        # 2. Startup readiness preflight
        cont = ContinuousLiveRuntime(
            symbol="XAUUSD",
            timeframes=["5m"],
            poll_interval=0.0,
            publish=True,
            persist=False,
            research_dir=tmp_path,
            publisher=pub_mock,
            candidate_ids={"5m": cand_id},
            market_data_loaders=lambda tf: make_tf_market_data("2025-01-01 10:00", "5m"),
        )
        startup_readiness = cont.verify_startup_readiness()
        assert startup_readiness["5m"]["status"] == "READY"

        # Neither preflight check MUST invoke live market evaluation or outbound publishing
        mock_create_eval.assert_not_called()
        mock_eval_runtime.assert_not_called()
        pub_mock.publish.assert_not_called()


def test_seven_frame_parity_distinguished_from_single_frame_liveness():
    """Verify seven_frame_parity is False when only 1 of 7 timeframes is evaluated healthy."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    all_tfs = ["1m", "5m", "15m", "30m", "1H", "4H", "1D"]
    runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=all_tfs,
        candidate_ids={"5m": "cand_moving_average_5m"},
        clock=lambda: now_dt,
    )
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()

    evals = {
        "5m": make_real_run_once_success(candidate_id="cand_moving_average_5m", timeframe="5m"),
        "1m": {"blocked": True, "status": "BLOCKED", "reason": "PromotionUnavailable"},
    }

    h = runtime.get_health_status(reference_now=now_dt, current_evaluations=evals)
    assert h["status"] == "WORKER_HEALTHY"
    assert h["execution_healthy"] is True
    assert h["seven_frame_parity"] is False


def test_conflicting_configured_and_readiness_candidates_fail_closed():
    """Verify that a conflict between configured candidate_id and readiness candidate_id fails closed."""
    now_dt = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)
    runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        candidate_ids={"5m": "cand_configured_5m"},
        clock=lambda: now_dt,
    )
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()
    runtime._timeframe_readiness = {
        "5m": {"status": "READY", "candidate_id": "cand_readiness_5m", "reason_code": "READY"},
    }

    evals = {
        "5m": make_real_run_once_success(candidate_id="cand_configured_5m", timeframe="5m"),
    }

    h = runtime.get_health_status(reference_now=now_dt, current_evaluations=evals)
    assert h["status"] == "WORKER_DEGRADED"
    assert h["execution_healthy"] is False
