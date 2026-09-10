from __future__ import annotations

import pytest

from src.evaluation.live_decision_history import (
    append_live_decision,
    build_live_decision_history,
    get_latest_live_decision,
)


def _snapshot(
    timestamp: str = "2026-09-10T05:45:00+00:00",
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


def test_build_live_decision_history_normalizes_snapshots():
    history = build_live_decision_history(
        [
            _snapshot(),
            _snapshot(
                timestamp="2026-09-10T05:50:00+00:00",
                signal_label="NO TRADE",
            ),
        ]
    )

    assert len(history) == 2
    assert history[0]["signal_label"] == "BUY"
    assert history[1]["signal_label"] == "NO TRADE"


def test_build_live_decision_history_preserves_order():
    history = build_live_decision_history(
        [
            _snapshot("2026-09-10T05:45:00+00:00"),
            _snapshot("2026-09-10T05:50:00+00:00"),
            _snapshot("2026-09-10T05:55:00+00:00"),
        ]
    )

    assert [
        item["timestamp"]
        for item in history
    ] == [
        "2026-09-10T05:45:00+00:00",
        "2026-09-10T05:50:00+00:00",
        "2026-09-10T05:55:00+00:00",
    ]


def test_build_live_decision_history_empty_input():
    assert build_live_decision_history([]) == []


def test_build_live_decision_history_rejects_invalid_input():
    with pytest.raises(TypeError):
        build_live_decision_history("invalid")


def test_build_live_decision_history_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        build_live_decision_history([[]])


def test_append_live_decision_to_empty_history():
    history = append_live_decision(
        None,
        _snapshot(),
    )

    assert len(history) == 1
    assert history[0]["symbol"] == "XAUUSD"
    assert history[0]["signal_label"] == "BUY"


def test_append_live_decision_preserves_existing_history():
    first = build_live_decision_history(
        [_snapshot()]
    )

    history = append_live_decision(
        first,
        _snapshot(
            timestamp="2026-09-10T05:50:00+00:00",
            signal_label="NO TRADE",
        ),
    )

    assert len(history) == 2
    assert history[0]["signal_label"] == "BUY"
    assert history[1]["signal_label"] == "NO TRADE"


def test_append_live_decision_does_not_mutate_original_history():
    original = build_live_decision_history(
        [_snapshot()]
    )

    append_live_decision(
        original,
        _snapshot(
            timestamp="2026-09-10T05:50:00+00:00",
        ),
    )

    assert len(original) == 1


def test_append_live_decision_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        append_live_decision([], [])


def test_append_live_decision_rejects_invalid_history_item():
    with pytest.raises(TypeError):
        append_live_decision(
            [[]],
            _snapshot(),
        )


def test_get_latest_live_decision_returns_last_record():
    history = build_live_decision_history(
        [
            _snapshot("2026-09-10T05:45:00+00:00"),
            _snapshot("2026-09-10T05:50:00+00:00"),
        ]
    )

    latest = get_latest_live_decision(history)

    assert latest is not None
    assert latest["timestamp"] == "2026-09-10T05:50:00+00:00"


def test_get_latest_live_decision_returns_copy():
    history = build_live_decision_history(
        [_snapshot()]
    )

    latest = get_latest_live_decision(history)

    assert latest is not None
    latest["signal_label"] = "CHANGED"

    assert history[0]["signal_label"] == "BUY"


def test_get_latest_live_decision_empty_history():
    assert get_latest_live_decision([]) is None


def test_get_latest_live_decision_rejects_invalid_history():
    with pytest.raises(TypeError):
        get_latest_live_decision("invalid")
