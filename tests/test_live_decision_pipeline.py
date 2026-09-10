from __future__ import annotations

import pytest

from src.evaluation.live_decision_pipeline import (
    load_and_evaluate_live_decisions,
    process_live_decision,
    validate_live_decision_pipeline_result,
)


def _snapshot(
    timestamp: str,
    signal_label: str = "BUY",
) -> dict:
    return {
        "timestamp": timestamp,
        "symbol": "XAUUSD",
        "interval": "5m",
        "signal": 1 if signal_label == "BUY" else 0,
        "signal_label": signal_label,
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


def test_process_live_decision_builds_record_and_evaluation():
    result = process_live_decision(
        _snapshot("2026-09-10T05:45:00+00:00")
    )

    assert result["record"]["symbol"] == "XAUUSD"
    assert result["record"]["signal_label"] == "BUY"
    assert result["persisted"] is False
    assert result["store_path"] is None

    assert result["evaluation"]["record_count"] == 1
    assert result["evaluation"]["health_status"] == "INITIALIZING"


def test_process_live_decision_appends_to_supplied_history():
    existing = _snapshot("2026-09-10T05:45:00+00:00")
    current = _snapshot("2026-09-10T05:50:00+00:00")

    result = process_live_decision(
        current,
        history=[existing],
    )

    assert len(result["history"]) == 2
    assert result["history"][0]["timestamp"] == (
        "2026-09-10T05:45:00+00:00"
    )
    assert result["history"][1]["timestamp"] == (
        "2026-09-10T05:50:00+00:00"
    )

    assert result["evaluation"]["record_count"] == 2


def test_process_live_decision_persists_history(tmp_path):
    path = tmp_path / "live_decisions.json"

    result = process_live_decision(
        _snapshot("2026-09-10T05:45:00+00:00"),
        store_path=str(path),
    )

    assert result["persisted"] is True
    assert result["store_path"] == str(path)
    assert path.exists()

    loaded = load_and_evaluate_live_decisions(
        str(path)
    )

    assert len(loaded["history"]) == 1
    assert loaded["history"][0]["symbol"] == "XAUUSD"
    assert loaded["evaluation"]["record_count"] == 1


def test_process_live_decision_persistent_calls_accumulate(
    tmp_path,
):
    path = tmp_path / "live_decisions.json"

    first = process_live_decision(
        _snapshot("2026-09-10T05:45:00+00:00"),
        store_path=str(path),
    )

    second = process_live_decision(
        _snapshot(
            "2026-09-10T05:50:00+00:00",
            "NO TRADE",
        ),
        store_path=str(path),
    )

    assert first["evaluation"]["record_count"] == 1
    assert second["evaluation"]["record_count"] == 2
    assert second["evaluation"]["changed_transition_count"] == 1


def test_process_live_decision_preserves_recorded_values():
    snapshot = _snapshot("2026-09-10T05:45:00+00:00")

    snapshot["entry_price"] = 9999.0
    snapshot["stop_loss"] = 9000.0
    snapshot["take_profit"] = 12000.0
    snapshot["stability_score"] = 0.11

    result = process_live_decision(snapshot)

    record = result["record"]

    assert record["entry_price"] == 9999.0
    assert record["stop_loss"] == 9000.0
    assert record["take_profit"] == 12000.0
    assert record["stability_score"] == 0.11


def test_load_and_evaluate_empty_store(tmp_path):
    path = tmp_path / "live_decisions.json"

    process_live_decision(
        _snapshot("2026-09-10T05:45:00+00:00"),
        store_path=str(path),
    )

    result = load_and_evaluate_live_decisions(
        str(path)
    )

    assert result["persisted"] is True
    assert result["evaluation"]["record_count"] == 1
    assert result["evaluation"]["healthy"] is True


def test_process_live_decision_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        process_live_decision([])


def test_process_live_decision_rejects_invalid_history():
    with pytest.raises(TypeError):
        process_live_decision(
            _snapshot("2026-09-10T05:45:00+00:00"),
            history="invalid",
        )


def test_process_live_decision_rejects_invalid_history_record():
    with pytest.raises(TypeError):
        process_live_decision(
            _snapshot("2026-09-10T05:45:00+00:00"),
            history=[[]],
        )


def test_load_and_evaluate_missing_store_is_rejected(
    tmp_path,
):
    path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        load_and_evaluate_live_decisions(str(path))


def test_pipeline_result_validation_accepts_valid_result():
    result = process_live_decision(
        _snapshot("2026-09-10T05:45:00+00:00")
    )

    assert (
        validate_live_decision_pipeline_result(result)
        is True
    )


def test_pipeline_result_validation_rejects_missing_fields():
    with pytest.raises(ValueError):
        validate_live_decision_pipeline_result(
            {
                "record": {},
            }
        )


def test_pipeline_result_validation_rejects_wrong_history_type():
    with pytest.raises(ValueError):
        validate_live_decision_pipeline_result(
            {
                "record": {},
                "history": {},
                "evaluation": {},
                "persisted": False,
                "store_path": None,
            }
        )


def test_pipeline_result_validation_rejects_wrong_persisted_type():
    with pytest.raises(ValueError):
        validate_live_decision_pipeline_result(
            {
                "record": {},
                "history": [],
                "evaluation": {},
                "persisted": "yes",
                "store_path": None,
            }
        )
