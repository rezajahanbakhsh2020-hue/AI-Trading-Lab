from src.evaluation.live_runtime_history import create_live_runtime_history
from src.evaluation.live_runtime_snapshot import LiveRuntimeSnapshot
from src.evaluation.live_runtime_readiness_gate import (
    build_live_runtime_readiness_gate,
    is_live_runtime_ready,
    live_runtime_readiness_gate_message,
    live_runtime_readiness_gate_dict,
)


def _snapshot(
    timestamp: str,
    *,
    strategy: str = "momentum",
    decision: str = "BUY",
    trend: str = "UP",
    controller_ready: bool = True,
    session_ready: bool = True,
) -> LiveRuntimeSnapshot:
    """Helper to create test snapshots."""
    return LiveRuntimeSnapshot(
        timestamp=timestamp,
        symbol="XAUUSD",
        interval="5m",
        strategy=strategy,
        stability_score=0.8,
        controller_ready=controller_ready,
        session_ready=session_ready,
        decision=decision,
        trend=trend,
        entry=3000.0,
        stop_loss=2990.0,
        take_profit_1=3010.0,
        take_profit_2=3020.0,
        take_profit_3=3030.0,
        failed_gates=(),
        failed_checks=(),
    )


# === EMPTY state tests ===

def test_empty_history_gate_is_not_ready() -> None:
    """Test that empty history produces NOT READY gate."""
    history = create_live_runtime_history()

    gate = build_live_runtime_readiness_gate(history)

    assert gate.is_ready is False
    assert gate.status == "EMPTY"
    assert "No runtime snapshots available" in gate.reason
    assert "staging mode" in gate.recommendation


def test_is_live_runtime_ready_empty() -> None:
    """Test quick check for empty history."""
    history = create_live_runtime_history()

    assert is_live_runtime_ready(history) is False


def test_readiness_gate_message_empty() -> None:
    """Test readiness gate message for EMPTY state."""
    history = create_live_runtime_history()

    message = live_runtime_readiness_gate_message(history)

    assert "LIVE RUNTIME READINESS GATE" in message
    assert "BLOCKED" in message
    assert "No runtime snapshots" in message


def test_readiness_gate_dict_empty() -> None:
    """Test readiness gate dict for EMPTY state."""
    history = create_live_runtime_history()

    gate_dict = live_runtime_readiness_gate_dict(history)

    assert gate_dict["is_ready"] is False
    assert gate_dict["status"] == "EMPTY"
    assert "No runtime snapshots" in gate_dict["reason"]
    assert "staging mode" in gate_dict["recommendation"]


# === DEGRADED state tests ===

def test_degraded_history_gate_is_not_ready() -> None:
    """Test that degraded history produces NOT READY gate."""
    history = create_live_runtime_history(
        [
            _snapshot(
                "",
                controller_ready=False,
                session_ready=False,
            ),
            _snapshot("invalid"),
        ]
    )

    gate = build_live_runtime_readiness_gate(history)

    assert gate.is_ready is False
    assert gate.status == "DEGRADED"
    assert "degraded" in gate.reason.lower()
    assert "health score" in gate.reason.lower()


def test_is_live_runtime_ready_degraded() -> None:
    """Test quick check for degraded history."""
    history = create_live_runtime_history(
        [
            _snapshot(
                "",
                controller_ready=False,
                session_ready=False,
            ),
        ]
    )

    assert is_live_runtime_ready(history) is False


# === WARNING state tests ===

def test_warning_state_gate_is_not_ready() -> None:
    """Test that WARNING state produces NOT READY gate.
    
    WARNING is triggered by READINESS_GAP alert when a snapshot
    with controller_ready=False is followed by controller_ready=True.
    """
    history = create_live_runtime_history(
        [
            _snapshot(
                "2026-01-01T00:00:00+00:00",
                controller_ready=False,
            ),
            _snapshot(
                "2026-01-01T00:05:00+00:00",
                controller_ready=True,
            ),
        ]
    )

    gate = build_live_runtime_readiness_gate(history)

    assert gate.is_ready is False
    assert gate.status == "WARNING"


def test_readiness_gate_message_warning() -> None:
    """Test readiness gate message for WARNING state."""
    history = create_live_runtime_history(
        [
            _snapshot(
                "2026-01-01T00:00:00+00:00",
                controller_ready=False,
            ),
            _snapshot(
                "2026-01-01T00:05:00+00:00",
                controller_ready=True,
            ),
        ]
    )

    gate = build_live_runtime_readiness_gate(history)

    assert gate.status == "WARNING"
    assert gate.is_ready is False
    message = live_runtime_readiness_gate_message(history)
    assert "LIVE RUNTIME READINESS GATE" in message
    assert "BLOCKED" in message or "✗" in message


# === READY state tests ===

def test_ready_history_gate_is_ready() -> None:
    """Test that healthy history produces READY gate."""
    history = create_live_runtime_history(
        [_snapshot("2026-01-01T00:00:00+00:00")]
    )

    gate = build_live_runtime_readiness_gate(history)

    assert gate.is_ready is True
    assert gate.status == "READY"
    assert "ready" in gate.reason.lower()
    assert "deployment" in gate.recommendation.lower()


def test_is_live_runtime_ready_ready() -> None:
    """Test quick check for ready history."""
    history = create_live_runtime_history(
        [_snapshot("2026-01-01T00:00:00+00:00")]
    )

    assert is_live_runtime_ready(history) is True


def test_readiness_gate_message_ready() -> None:
    """Test readiness gate message for READY state."""
    history = create_live_runtime_history(
        [_snapshot("2026-01-01T00:00:00+00:00")]
    )

    message = live_runtime_readiness_gate_message(history)

    assert "LIVE RUNTIME READINESS GATE" in message
    assert "READY" in message or "✓" in message
    assert "ready" in message.lower()


def test_readiness_gate_dict_ready() -> None:
    """Test readiness gate dict for READY state."""
    history = create_live_runtime_history(
        [_snapshot("2026-01-01T00:00:00+00:00")]
    )

    gate_dict = live_runtime_readiness_gate_dict(history)

    assert gate_dict["is_ready"] is True
    assert gate_dict["status"] == "READY"
    assert "all readiness criteria" in gate_dict["reason"].lower()
    assert "deployment" in gate_dict["recommendation"].lower()


# === Multiple snapshots test ===

def test_ready_with_multiple_snapshots() -> None:
    """Test readiness gate with multiple healthy snapshots."""
    history = create_live_runtime_history(
        [
            _snapshot("2026-01-01T00:00:00+00:00", strategy="momentum"),
            _snapshot("2026-01-01T00:05:00+00:00", strategy="breakout"),
            _snapshot("2026-01-01T00:10:00+00:00", strategy="momentum"),
        ]
    )

    gate = build_live_runtime_readiness_gate(history)

    assert gate.is_ready is True
    assert gate.status == "READY"


# === Invalid input test ===

def test_build_gate_raises_on_non_history_input() -> None:
    """Test that build_live_runtime_readiness_gate validates input type."""
    try:
        build_live_runtime_readiness_gate(None)  # type: ignore
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "LiveRuntimeHistory" in str(e)


# === Dict structure test ===

def test_readiness_gate_dict_contains_all_keys() -> None:
    """Test that dict output contains all required keys."""
    history = create_live_runtime_history(
        [_snapshot("2026-01-01T00:00:00+00:00")]
    )

    gate_dict = live_runtime_readiness_gate_dict(history)

    assert "is_ready" in gate_dict
    assert "status" in gate_dict
    assert "reason" in gate_dict
    assert "recommendation" in gate_dict
    assert len(gate_dict) == 4
