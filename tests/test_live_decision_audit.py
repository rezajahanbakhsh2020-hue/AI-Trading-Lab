from __future__ import annotations

import pytest

from src.evaluation.live_decision_audit import (
    audit_live_decision,
    audit_live_decision_history,
    validate_live_decision_audit,
)


def _record() -> dict:
    return {
        "timestamp": "2026-09-10T05:45:00+00:00",
        "symbol": "XAUUSD",
        "interval": "5m",
        "signal": 1,
        "signal_label": "BUY",
        "trend": "UP",
        "strategy": "momentum",
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


def test_audit_live_decision_accepts_valid_record():
    audit = audit_live_decision(_record())

    assert audit["valid"] is True
    assert audit["missing_fields"] == []
    assert audit["empty_fields"] == []
    assert audit["unexpected_fields"] == []
    assert audit["field_count"] == 16


def test_audit_live_decision_detects_missing_fields():
    record = _record()
    record.pop("trend")

    audit = audit_live_decision(record)

    assert audit["valid"] is False
    assert "trend" in audit["missing_fields"]


def test_audit_live_decision_detects_empty_fields():
    record = _record()
    record["strategy"] = None

    audit = audit_live_decision(record)

    assert audit["valid"] is False
    assert "strategy" in audit["empty_fields"]


def test_audit_live_decision_detects_unexpected_fields():
    record = _record()
    record["unexpected"] = "value"

    audit = audit_live_decision(record)

    assert audit["valid"] is True
    assert audit["unexpected_fields"] == ["unexpected"]


def test_audit_live_decision_does_not_recalculate_trading_values():
    record = _record()
    record["entry_price"] = 9999.0
    record["stop_loss"] = 8888.0
    record["take_profit"] = 7777.0

    audit = audit_live_decision(record)

    assert audit["valid"] is True


def test_audit_live_decision_rejects_invalid_record():
    with pytest.raises(TypeError):
        audit_live_decision([])


def test_audit_live_decision_history_accepts_valid_history():
    history = [_record(), _record()]

    result = audit_live_decision_history(history)

    assert result["record_count"] == 2
    assert result["valid_count"] == 2
    assert result["invalid_count"] == 0
    assert result["valid"] is True
    assert len(result["audits"]) == 2


def test_audit_live_decision_history_detects_invalid_history():
    valid = _record()
    invalid = _record()
    invalid.pop("signal_label")

    result = audit_live_decision_history(
        [valid, invalid]
    )

    assert result["record_count"] == 2
    assert result["valid_count"] == 1
    assert result["invalid_count"] == 1
    assert result["valid"] is False
    assert "signal_label" in result["audits"][1]["missing_fields"]


def test_audit_live_decision_history_accepts_empty_history():
    result = audit_live_decision_history([])

    assert result["record_count"] == 0
    assert result["valid_count"] == 0
    assert result["invalid_count"] == 0
    assert result["valid"] is True
    assert result["audits"] == []


def test_audit_live_decision_history_rejects_invalid_input():
    with pytest.raises(TypeError):
        audit_live_decision_history("invalid")


def test_audit_live_decision_history_rejects_invalid_item():
    with pytest.raises(TypeError):
        audit_live_decision_history([[]])


def test_validate_live_decision_audit_accepts_valid_audit():
    audit = audit_live_decision(_record())

    assert validate_live_decision_audit(audit) is True


def test_validate_live_decision_audit_rejects_missing_field():
    audit = audit_live_decision(_record())
    audit.pop("field_count")

    with pytest.raises(
        ValueError,
        match="missing required fields",
    ):
        validate_live_decision_audit(audit)


def test_validate_live_decision_audit_rejects_invalid_valid_flag():
    audit = audit_live_decision(_record())
    audit["valid"] = "yes"

    with pytest.raises(
        ValueError,
        match="'valid' must be boolean",
    ):
        validate_live_decision_audit(audit)
