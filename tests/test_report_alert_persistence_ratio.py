from __future__ import annotations

import pytest

from src.evaluation.report_alert_persistence_ratio import (
    build_alert_persistence_ratio_summary,
    calculate_alert_change_ratio,
    calculate_alert_persistence_ratio,
)


def test_calculate_alert_persistence_ratio():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 0},
    ]

    # States: active, active, inactive, active, active, inactive
    # Persistent transitions: 2 out of 5.
    assert calculate_alert_persistence_ratio(history) == pytest.approx(
        2 / 5
    )


def test_calculate_alert_persistence_ratio_all_persistent():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert calculate_alert_persistence_ratio(history) == 1.0


def test_calculate_alert_persistence_ratio_no_persistence():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
    ]

    assert calculate_alert_persistence_ratio(history) == 0.0


def test_calculate_alert_persistence_ratio_all_inactive():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_persistence_ratio(history) == 1.0


def test_calculate_alert_persistence_ratio_single_snapshot():
    history = [
        {"alert_count": 5},
    ]

    assert calculate_alert_persistence_ratio(history) == 1.0


def test_calculate_alert_persistence_ratio_empty():
    with pytest.raises(ValueError):
        calculate_alert_persistence_ratio([])


def test_calculate_alert_change_ratio():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 0},
    ]

    assert calculate_alert_change_ratio(history) == pytest.approx(
        3 / 5
    )


def test_calculate_alert_change_ratio_all_persistent():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
    ]

    assert calculate_alert_change_ratio(history) == 0.0


def test_calculate_alert_change_ratio_no_persistence():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
    ]

    assert calculate_alert_change_ratio(history) == 1.0


def test_build_alert_persistence_ratio_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 0},
    ]

    result = build_alert_persistence_ratio_summary(history)

    assert result == {
        "snapshot_count": 6,
        "alert_persistence_ratio": pytest.approx(2 / 5),
        "alert_change_ratio": pytest.approx(3 / 5),
    }


def test_build_alert_persistence_ratio_summary_perfect():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
    ]

    result = build_alert_persistence_ratio_summary(history)

    assert result == {
        "snapshot_count": 3,
        "alert_persistence_ratio": 1.0,
        "alert_change_ratio": 0.0,
    }


def test_build_alert_persistence_ratio_summary_single_snapshot():
    history = [
        {"alert_count": 3},
    ]

    result = build_alert_persistence_ratio_summary(history)

    assert result == {
        "snapshot_count": 1,
        "alert_persistence_ratio": 1.0,
        "alert_change_ratio": 0.0,
    }


def test_build_alert_persistence_ratio_summary_empty():
    with pytest.raises(ValueError):
        build_alert_persistence_ratio_summary([])


def test_rejects_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_alert_persistence_ratio({})


def test_rejects_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_alert_persistence_ratio(
            [{"alert_count": 1}, "invalid"]
        )


def test_rejects_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_persistence_ratio(
            [{"value": 1}]
        )


def test_rejects_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_persistence_ratio(
            [{"alert_count": 1.5}]
        )


def test_rejects_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_persistence_ratio(
            [{"alert_count": True}]
        )


def test_rejects_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_persistence_ratio(
            [{"alert_count": -1}]
        )
