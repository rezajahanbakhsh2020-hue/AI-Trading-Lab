from datetime import datetime

from src.evaluation.live_runtime_preflight import (
    build_live_runtime_preflight,
    is_live_runtime_preflight_ready,
    live_runtime_preflight_message,
)


def _ready_decision() -> dict:
    return {
        "runtime_allowed": True,
        "status": "READY",
        "strategy": "momentum",
        "readiness": {
            "status": "READY",
            "readiness_score": 0.90,
        },
        "runtime_gate": {
            "runtime_ready": True,
            "status": "READY",
        },
    }


def test_preflight_is_ready():
    result = build_live_runtime_preflight(
        _ready_decision()
    )

    assert result["preflight_ready"] is True
    assert result["status"] == "READY"
    assert result["strategy"] == "momentum"
    assert result["failed_checks"] == []

    datetime.fromisoformat(
        result["checked_at"]
    )


def test_preflight_blocks_runtime_permission():
    decision = _ready_decision()
    decision["runtime_allowed"] = False

    result = build_live_runtime_preflight(
        decision
    )

    assert result["preflight_ready"] is False
    assert "runtime_permission" in result[
        "failed_checks"
    ]


def test_preflight_blocks_market_data():
    result = build_live_runtime_preflight(
        _ready_decision(),
        market_data_ready=False,
    )

    assert result["preflight_ready"] is False
    assert "market_data" in result[
        "failed_checks"
    ]


def test_preflight_blocks_quote():
    result = build_live_runtime_preflight(
        _ready_decision(),
        quote_ready=False,
    )

    assert result["preflight_ready"] is False
    assert "quote" in result[
        "failed_checks"
    ]


def test_preflight_blocks_timestamp():
    result = build_live_runtime_preflight(
        _ready_decision(),
        timestamp_ready=False,
    )

    assert result["preflight_ready"] is False
    assert "timestamp" in result[
        "failed_checks"
    ]


def test_preflight_blocks_missing_strategy():
    decision = _ready_decision()
    decision["strategy"] = None

    result = build_live_runtime_preflight(
        decision
    )

    assert result["preflight_ready"] is False
    assert "strategy" in result[
        "failed_checks"
    ]


def test_preflight_blocks_missing_readiness():
    decision = _ready_decision()
    del decision["readiness"]

    result = build_live_runtime_preflight(
        decision
    )

    assert result["preflight_ready"] is False
    assert "readiness" in result[
        "failed_checks"
    ]


def test_preflight_blocks_missing_runtime_gate():
    decision = _ready_decision()
    del decision["runtime_gate"]

    result = build_live_runtime_preflight(
        decision
    )

    assert result["preflight_ready"] is False
    assert "runtime_gate" in result[
        "failed_checks"
    ]


def test_preflight_permission_returns_boolean():
    decision = _ready_decision()

    assert (
        is_live_runtime_preflight_ready(
            decision
        )
        is True
    )

    assert (
        is_live_runtime_preflight_ready(
            decision,
            market_data_ready=False,
        )
        is False
    )


def test_preflight_message_ready():
    message = live_runtime_preflight_message(
        _ready_decision()
    )

    assert message == (
        "LIVE PREFLIGHT READY: momentum"
    )


def test_preflight_message_blocked():
    message = live_runtime_preflight_message(
        _ready_decision(),
        quote_ready=False,
    )

    assert message.startswith(
        "LIVE PREFLIGHT BLOCKED:"
    )
    assert "quote" in message


def test_preflight_rejects_non_mapping():
    try:
        build_live_runtime_preflight([])
    except TypeError:
        return

    raise AssertionError(
        "Expected TypeError for non-mapping input."
    )
