from __future__ import annotations

import pytest

from src.evaluation.live_decision_audit_report import (
    build_live_decision_audit_report,
    validate_live_decision_audit_report,
)


def _record(
    timestamp: str,
    signal_label: str = "BUY",
    trend: str = "UP",
    strategy: str = "momentum",
) -> dict:
    return {
        "timestamp": timestamp,
        "symbol": "XAUUSD",
        "interval": "5m",
        "signal": 1 if signal_label == "BUY" else 0,
        "signal_label": signal_label,
        "trend": trend,
        "strategy": strategy,
        "entry_price": 4429.802,
        "stop_loss": 4385.50398,
        "take_profit": 4518.39804,
        "risk_reward_ratio": 2.0,
        "stability_score": 0.65,
        "market_state": "open",
        "quote_age_seconds": 0,
        "quote_stale": False,
        "candle_count": 201,
    }


def test_audit_report_passes_for_valid_history():
    history = [
        _record("2026-09-10T05:45:00+00:00"),
        _record("2026-09-10T05:50:00+00:00"),
        _record("2026-09-10T05:55:00+00:00"),
    ]

    report = build_live_decision_audit_report(history)

    assert report["record_count"] == 3
    assert report["valid_structure"] is True
    assert report["consistent_timeline"] is True
    assert report["audit_passed"] is True


def test_audit_report_counts_transitions():
    history = [
        _record("2026-09-10T05:45:00+00:00"),
        _record("2026-09-10T05:50:00+00:00"),
        _record("2026-09-10T05:55:00+00:00"),
    ]

    report = build_live_decision_audit_report(history)

    assert report["transition_count"] == 2
    assert len(report["changes"]) == 2


def test_audit_report_detects_changed_transition():
    history = [
        _record(
            "2026-09-10T05:45:00+00:00",
            "BUY",
            "UP",
            "momentum",
        ),
        _record(
            "2026-09-10T05:50:00+00:00",
            "NO TRADE",
            "FLAT",
            "mean_reversion",
        ),
    ]

    report = build_live_decision_audit_report(history)

    assert report["changed_transition_count"] == 1
    assert report["changes"][0]["changed"] is True
    assert report["changes"][0]["signal_label" if False else "changed"] is True


def test_audit_report_detects_timeline_problem():
    history = [
        _record("2026-09-10T05:50:00+00:00"),
        _record("2026-09-10T05:45:00+00:00"),
    ]

    report = build_live_decision_audit_report(history)

    assert report["consistent_timeline"] is False
    assert report["audit_passed"] is False


def test_audit_report_detects_structural_problem():
    record = _record("2026-09-10T05:45:00+00:00")
    del record["strategy"]

    report = build_live_decision_audit_report([record])

    assert report["valid_structure"] is False
    assert report["audit_passed"] is False


def test_empty_history_produces_passing_report():
    report = build_live_decision_audit_report([])

    assert report["record_count"] == 0
    assert report["transition_count"] == 0
    assert report["changed_transition_count"] == 0
    assert report["changes"] == []
    assert report["audit_passed"] is True


def test_audit_report_does_not_recalculate_values():
    record = _record("2026-09-10T05:45:00+00:00")
    record["entry_price"] = 9999.0
    record["stop_loss"] = 9000.0
    record["take_profit"] = 12000.0

    report = build_live_decision_audit_report([record])

    assert report["audit_passed"] is True
    assert record["entry_price"] == 9999.0
    assert record["stop_loss"] == 9000.0
    assert record["take_profit"] == 12000.0


def test_invalid_history_input_is_rejected():
    with pytest.raises(TypeError):
        build_live_decision_audit_report("invalid")


def test_invalid_history_record_is_rejected():
    with pytest.raises(TypeError):
        build_live_decision_audit_report([[]])


def test_validation_accepts_valid_report():
    report = build_live_decision_audit_report(
        [
            _record("2026-09-10T05:45:00+00:00"),
            _record("2026-09-10T05:50:00+00:00"),
        ]
    )

    assert validate_live_decision_audit_report(report) is True


def test_validation_rejects_missing_fields():
    with pytest.raises(ValueError):
        validate_live_decision_audit_report(
            {
                "record_count": 0,
            }
        )


def test_validation_rejects_wrong_boolean_type():
    with pytest.raises(ValueError):
        validate_live_decision_audit_report(
            {
                "record_count": 0,
                "valid_structure": "yes",
                "structural_audit": {},
                "consistent_timeline": True,
                "consistency_audit": {},
                "transition_count": 0,
                "changed_transition_count": 0,
                "changes": [],
                "audit_passed": True,
            }
        )


def test_validation_rejects_wrong_transition_count():
    with pytest.raises(ValueError):
        validate_live_decision_audit_report(
            {
                "record_count": 2,
                "valid_structure": True,
                "structural_audit": {},
                "consistent_timeline": True,
                "consistency_audit": {},
                "transition_count": 2,
                "changed_transition_count": 0,
                "changes": [],
                "audit_passed": True,
            }
        )
