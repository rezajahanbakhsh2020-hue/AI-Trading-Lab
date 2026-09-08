from __future__ import annotations

import pytest

from src.evaluation.report_alert_state_ratio import (
    build_alert_state_ratio_summary,
    calculate_active_state_ratio,
    calculate_inactive_state_ratio,
)


def test_calculate_active_state_ratio():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    assert calculate_active_state_ratio(history) == pytest.approx(
        3 / 5
    )


def test_calculate_inactive_state_ratio():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    assert calculate_inactive_state_ratio(history) == pytest.approx(
        2 / 5
    )


def test_all_active():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert calculate_active_state_ratio(history) == 1.0
    assert calculate_inactive_state_ratio(history) == 0.0


def test_all_inactive():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_active_state_ratio(history) == 0.0
    assert calculate_inactive_state_ratio(history) == 1.0


def test_single_active_snapshot():
    history = [
        {"alert_count":5},
    ]

    assert calculate_active_state_ratio(history) == 1.0
    assert calculate_inactive_state_ratio(history) == 0.0


def test_single_inactive_snapshot():
    history = [
        {"alert_count": 0},
    ]

    assert calculate_active_state_ratio(history) == 0.0
    assert calculate_inactive_state_ratio(history) == 1.0


def test_build_alert_state_ratio_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    result = build_alert_state_ratio_summary(history)

    assert result == {
        "snapshot_count": 5,
        "active_state_ratio": pytest.approx(3 / 5),
        "inactive_state_ratio": pytest.approx(2 / 5),
    }


def test_build_alert_state_ratio_summary_all_active():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
    ]

    result = build_alert_state_ratio_summary(history)

    assert result == {
        "snapshot_count": 2,
        "active_state_ratio": 1.0,
        "inactive_state_ratio": 0.0,
    }


def test_build_alert_state_ratio_summary_all_inactive():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    result = build_alert_state_ratio_summary(history)

    assert result == {
        "snapshot_count": 2,
        "active_state_ratio": 0.0,
        "inactive_state_ratio": 1.0,
    }


def test_empty_history():
    with pytest.raises(ValueError):
        calculate_active_state_ratio([])


def test_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_active_state_ratio({})


def test_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_active_state_ratio(
            [{"alert_count": 1}, "invalid"]
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_active_state_ratio(
            [{"value": 1}]
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_active_state_ratio(
            [{"alert_count": 1.5}]
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_active_state_ratio(
            [{"alert_count": True}]
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_active_state_ratio(
            [{"alert_count": -1}]
        )
