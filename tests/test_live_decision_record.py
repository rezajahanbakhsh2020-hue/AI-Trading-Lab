from __future__ import annotations

import pytest

from src.evaluation.live_decision_record import (
    DECISION_RECORD_FIELDS,
    build_live_decision_record,
    record_live_decision,
    validate_live_decision_record,
)


def _snapshot() -> dict:
    return {
        "timestamp": "2026-09-10T05:45:00+00:00",
        "symbol": "XAUUSD",
        "interval": "5m",
        "signal": 1,
        "signal_label": "BUY",
        "trend": "UP",
        "strategy": "momentum",
        "momentum": 0.004802854765700415,
        "entry_price": 4429.802,
        "stop_loss": 4385.5039799999995,
        "take_profit": 4518.39804,
        "risk_reward_ratio": 2.0,
        "stability_score": 0.65,
        "market_state": "open",
        "quote_age_seconds": 0,
        "quote_stale": False,
        "candle_count": 201,
    }


def test_build_live_decision_record_preserves_live_values():
    record = build_live_decision_record(_snapshot())

    assert record["timestamp"] == "2026-09-10T05:45:00+00:00"
    assert record["symbol"] == "XAUUSD"
    assert record["interval"] == "5m"
    assert record["signal"] == 1
    assert record["signal_label"] == "BUY"
    assert record["trend"] == "UP"
    assert record["strategy"] == "momentum"
    assert record["entry_price"] == 4429.802
    assert record["stop_loss"] == 4385.5039799999995
    assert record["take_profit"] == 4518.39804
    assert record["risk_reward_ratio"] == 2.0
    assert record["stability_score"] == 0.65
    assert record["market_state"] == "open"
    assert record["quote_age_seconds"] == 0
    assert record["quote_stale"] is False
    assert record["candle_count"] == 201


def test_build_live_decision_record_contains_all_record_fields():
    record = build_live_decision_record(_snapshot())

    assert set(record) == set(DECISION_RECORD_FIELDS)


def test_build_live_decision_record_does_not_recalculate_values():
    snapshot = _snapshot()
    snapshot["entry_price"] = 9999.0
    snapshot["stop_loss"] = 8888.0
    snapshot["take_profit"] = 7777.0

    record = build_live_decision_record(snapshot)

    assert record["entry_price"] == 9999.0
    assert record["stop_loss"] == 8888.0
    assert record["take_profit"] == 7777.0


def test_build_live_decision_record_preserves_missing_optional_values():
    snapshot = _snapshot()
    snapshot.pop("stability_score")
    snapshot.pop("risk_reward_ratio")

    record = build_live_decision_record(snapshot)

    assert record["stability_score"] is None
    assert record["risk_reward_ratio"] is None


def test_build_live_decision_record_accepts_market_state_alias():
    snapshot = _snapshot()
    snapshot.pop("market_state")
    snapshot["marketState"] = "open"

    record = build_live_decision_record(snapshot)

    assert record["market_state"] == "open"


def test_build_live_decision_record_uses_signal_as_label_when_missing():
    snapshot = _snapshot()
    snapshot.pop("signal_label")
    snapshot["signal"] = "BUY"

    record = build_live_decision_record(snapshot)

    assert record["signal"] == "BUY"
    assert record["signal_label"] == "BUY"


def test_build_live_decision_record_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        build_live_decision_record([])


def test_validate_live_decision_record_accepts_valid_record():
    record = build_live_decision_record(_snapshot())

    assert validate_live_decision_record(record) is True


def test_validate_live_decision_record_rejects_missing_field():
    record = build_live_decision_record(_snapshot())
    record.pop("symbol")

    with pytest.raises(ValueError, match="missing required fields"):
        validate_live_decision_record(record)


def test_validate_live_decision_record_rejects_empty_identity():
    record = build_live_decision_record(_snapshot())
    record["symbol"] = ""

    with pytest.raises(ValueError, match="must not be empty"):
        validate_live_decision_record(record)


def test_validate_live_decision_record_rejects_invalid_record():
    with pytest.raises(TypeError):
        validate_live_decision_record([])


def test_record_live_decision_builds_and_validates():
    record = record_live_decision(_snapshot())

    assert record["symbol"] == "XAUUSD"
    assert record["signal_label"] == "BUY"
    assert record["trend"] == "UP"
    assert record["entry_price"] == 4429.802
    assert record["take_profit"] == 4518.39804


def test_record_live_decision_does_not_invent_tp2_or_tp3():
    record = record_live_decision(_snapshot())

    assert "tp2" not in record
    assert "tp3" not in record
