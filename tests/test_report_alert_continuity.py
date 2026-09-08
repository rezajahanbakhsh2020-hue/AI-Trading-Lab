from __future__ import annotations

import pytest

from src.evaluation.report_alert_continuity import (
    build_alert_continuity_summary,
    calculate_alert_continuity,
    calculate_alert_discontinuity,
)


def test_calculate_alert_continuity():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 0},
    ]

    # States: active, active, inactive, active, active, inactive
    # Continuous transitions: 2 out of 5.
    assert calculate_alert_continuity(history) == pytest.approx(
        2 / 5
    )


def test_calculate_alert_discontinuity():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 0},
    ]

    assert calculate_alert_discontinuity(history) == pytest.approx(
        3 / 5
    )


def test_all_active_is_fully_continuous():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert calculate_alert_continuity(history) == 1.0
    assert calculate_alert_discontinuity(history) == 0.0


def test_all_inactive_is_fully_continuous():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_continuity(history) == 1.0
    assert calculate_alert_discontinuity(history) == 0.0


def test_every_transition_is_discontinuous():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
    ]

    assert calculate_alert_continuity(history) == 0.0
    assert calculate_alert_discontinuity(history) == 1.0


def test_single_snapshot():
    history = [
        {"alert_count": 5},
    ]

    assert calculate_alert_continuity(history) == 1.0
    assert calculate_alert_discontinuity(history) == 0.0


def test_empty_history():
    with pytest.raises(ValueError):
        calculate_alert_continuity([])


def test_build_alert_continuity_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 0},
    ]

    result = build_alert_continuity_summary(history)

    assert result == {
        "snapshot_count": 6,
        "transition_count": 5,
        "alert_continuity": pytest.approx(2 / 5),
        "alert_discontinuity": pytest.approx(3 / 5),
    }


def test_build_summary_single_snapshot():
    history = [
        {"alert_count": 2},
    ]

    result = build_alert_continuity_summary(history)

    assert result == {
        "snapshot_count": 1,
        "transition_count": 0,
        "alert_continuity": 1.0,
        "alert_discontinuity": 0.0,
    }


def test_build_summary_empty_history():
    with pytest.raises(ValueError):
        build_alert_continuity_summary([])


def test_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_alert_continuity({})


def test_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_alert_continuity(
            [{"alert_count": 1}, "invalid"]
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_continuity(
            [{"value": 1}]
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_continuity(
            [{"alert_count": 1.5}]
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_continuity(
            [{"alert_count": True}]
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_continuity(
            [{"alert_count": -1}]
        )
