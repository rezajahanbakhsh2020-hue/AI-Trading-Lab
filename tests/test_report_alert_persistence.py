from __future__ import annotations

import pytest

from src.evaluation.report_alert_persistence import (
    build_alert_persistence_summary,
    calculate_alert_persistence,
    count_persistent_alert_transitions,
    has_persistent_alerts,
)


def test_calculate_alert_persistence():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    assert calculate_alert_persistence(history) == 1 / 3


def test_calculate_alert_persistence_all_persistent():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 1},
    ]

    assert calculate_alert_persistence(history) == 1.0


def test_calculate_alert_persistence_no_persistence():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
    ]

    assert calculate_alert_persistence(history) == 0.0


def test_calculate_alert_persistence_single_snapshot():
    assert calculate_alert_persistence(
        [{"alert_count": 2}]
    ) == 0.0


def test_calculate_alert_persistence_empty():
    with pytest.raises(ValueError):
        calculate_alert_persistence([])


def test_calculate_alert_persistence_rejects_invalid_history():
    with pytest.raises(TypeError):
        calculate_alert_persistence({})


def test_calculate_alert_persistence_rejects_invalid_item():
    with pytest.raises(TypeError):
        calculate_alert_persistence(
            [{"alert_count": 1}, "invalid"]
        )


def test_calculate_alert_persistence_rejects_missing_count():
    with pytest.raises(ValueError):
        calculate_alert_persistence(
            [{"value": 1}]
        )


def test_calculate_alert_persistence_rejects_non_integer_count():
    with pytest.raises(ValueError):
        calculate_alert_persistence(
            [{"alert_count": 1.5}]
        )


def test_calculate_alert_persistence_rejects_boolean_count():
    with pytest.raises(ValueError):
        calculate_alert_persistence(
            [{"alert_count": True}]
        )


def test_calculate_alert_persistence_rejects_negative_count():
    with pytest.raises(ValueError):
        calculate_alert_persistence(
            [{"alert_count": -1}]
        )


def test_count_persistent_alert_transitions():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
    ]

    assert count_persistent_alert_transitions(history) == 2


def test_count_persistent_alert_transitions_empty():
    assert count_persistent_alert_transitions([]) == 0


def test_count_persistent_alert_transitions_single():
    assert count_persistent_alert_transitions(
        [{"alert_count": 1}]
    ) == 0


def test_has_persistent_alerts_true():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
    ]

    assert has_persistent_alerts(history) is True


def test_has_persistent_alerts_false():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
    ]

    assert has_persistent_alerts(history) is False


def test_build_alert_persistence_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
    ]

    result = build_alert_persistence_summary(history)

    assert result == {
        "snapshot_count": 5,
        "transition_count": 4,
        "persistent_transitions": 2,
        "persistence_ratio": 0.5,
        "has_persistent_alerts": True,
    }


def test_build_alert_persistence_summary_single():
    result = build_alert_persistence_summary(
        [{"alert_count": 1}]
    )

    assert result == {
        "snapshot_count": 1,
        "transition_count": 0,
        "persistent_transitions": 0,
        "persistence_ratio": 0.0,
        "has_persistent_alerts": False,
    }


def test_build_alert_persistence_summary_empty():
    result = build_alert_persistence_summary([])

    assert result == {
        "snapshot_count": 0,
        "transition_count": 0,
        "persistent_transitions": 0,
        "persistence_ratio": 0.0,
        "has_persistent_alerts": False,
    }
