from __future__ import annotations

import json

import pytest

from src.evaluation.live_decision_store import (
    append_live_decision_to_store,
    load_live_decision_history,
    save_live_decision_history,
)


def _record(
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


def test_save_live_decision_history_writes_json(tmp_path):
    path = tmp_path / "decisions.json"
    history = [
        _record("2026-09-10T05:45:00+00:00"),
    ]

    result = save_live_decision_history(history, path)

    assert result == path
    assert path.exists()

    stored = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert len(stored) == 1
    assert stored[0]["symbol"] == "XAUUSD"
    assert stored[0]["signal_label"] == "BUY"


def test_load_live_decision_history_restores_records(tmp_path):
    path = tmp_path / "decisions.json"
    history = [
        _record("2026-09-10T05:45:00+00:00"),
        _record(
            "2026-09-10T05:50:00+00:00",
            "NO TRADE",
        ),
    ]

    save_live_decision_history(history, path)

    loaded = load_live_decision_history(path)

    assert loaded == history


def test_save_live_decision_history_creates_parent_directory(
    tmp_path,
):
    path = tmp_path / "nested" / "decisions.json"

    save_live_decision_history(
        [_record("2026-09-10T05:45:00+00:00")],
        path,
    )

    assert path.exists()


def test_append_live_decision_to_store_creates_new_store(
    tmp_path,
):
    path = tmp_path / "decisions.json"
    record = _record("2026-09-10T05:45:00+00:00")

    history = append_live_decision_to_store(
        record,
        path,
    )

    assert history == [record]

    loaded = load_live_decision_history(path)

    assert loaded == [record]


def test_append_live_decision_to_store_preserves_existing_history(
    tmp_path,
):
    path = tmp_path / "decisions.json"

    first = _record("2026-09-10T05:45:00+00:00")
    second = _record(
        "2026-09-10T05:50:00+00:00",
        "NO TRADE",
    )

    save_live_decision_history([first], path)

    history = append_live_decision_to_store(
        second,
        path,
    )

    assert history == [first, second]
    assert load_live_decision_history(path) == [
        first,
        second,
    ]


def test_store_does_not_recalculate_trading_values(
    tmp_path,
):
    path = tmp_path / "decisions.json"

    record = _record("2026-09-10T05:45:00+00:00")
    record["entry_price"] = 9999.0
    record["stop_loss"] = 9000.0
    record["take_profit"] = 12000.0

    save_live_decision_history([record], path)

    loaded = load_live_decision_history(path)

    assert loaded[0]["entry_price"] == 9999.0
    assert loaded[0]["stop_loss"] == 9000.0
    assert loaded[0]["take_profit"] == 12000.0


def test_load_live_decision_history_rejects_invalid_json_shape(
    tmp_path,
):
    path = tmp_path / "decisions.json"

    path.write_text(
        json.dumps({"record": "invalid"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_live_decision_history(path)


def test_load_live_decision_history_rejects_missing_file(
    tmp_path,
):
    path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        load_live_decision_history(path)


def test_save_live_decision_history_rejects_invalid_history(
    tmp_path,
):
    path = tmp_path / "decisions.json"

    with pytest.raises(TypeError):
        save_live_decision_history(
            "invalid",
            path,
        )


def test_save_live_decision_history_rejects_invalid_record(
    tmp_path,
):
    path = tmp_path / "decisions.json"

    with pytest.raises(TypeError):
        save_live_decision_history(
            [[]],
            path,
        )


def test_append_live_decision_to_store_rejects_invalid_record(
    tmp_path,
):
    path = tmp_path / "decisions.json"

    with pytest.raises(TypeError):
        append_live_decision_to_store(
            [],
            path,
        )
