from __future__ import annotations

import pytest

from src.evaluation.report_alert_switch_rate import (
    build_alert_switch_rate_summary,
    calculate_alert_switch_rate,
    count_alert_switches,
)


def test_calculate_alert_switch_rate():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 0},
    ]

    # States: active, inactive, active, active, inactive
    # Switches: 3 out of 4 transitions.
    assert calculate_alert_switch_rate(history) == pytest.approx(
        3 / 4
    )


def test_count_alert_switches():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 0},
    ]

    assert count_alert_switches(history) == 3


def test_all_persistent_active():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert calculate_alert_switch_rate(history) == 0.0
    assert count_alert_switches(history) == 0


def test_all_persistent_inactive():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_switch_rate(history) == 0.0
    assert count_alert_switches(history) == 0


def test_every_transition_switches():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
    ]

    assert calculate_alert_switch_rate(history) == 1.0
    assert count_alert_switches(history) == 3


def test_single_snapshot():
    history = [
        {"alert_count": 5},
    ]

    assert calculate_alert_switch_rate(history) == 0.0
    assert count_alert_switches(history) == 0


def test_empty_history():
    with pytest.raises(ValueError):
        calculate_alert_switch_rate([])

    with pytest.raises(ValueError):
        count_alert_switches([])


def test_build_alert_switch_rate_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 0},
    ]

    result = build_alert_switch_rate_summary(history)

    assert result == {
        "snapshot_count": 5,
        "transition_count": 4,
        "switch_count": 3,
        "alert_switch_rate": pytest.approx(3 / 4),
    }


def test_build_summary_single_snapshot():
    history = [
        {"alert_count": 2},
    ]

    result = build_alert_switch_rate_summary(history)

    assert result == {
        "snapshot_count": 1,
        "transition_count": 0,
        "switch_count": 0,
        "alert_switch_rate": 0.0,
    }


def test_build_summary_empty_history():
    with pytest.raises(ValueError):
        build_alert_switch_rate_summary([])


def test_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_alert_switch_rate({})


def test_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_alert_switch_rate(
            [{"alert_count": 1}, "invalid"]
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_switch_rate(
            [{"value": 1}]
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_switch_rate(
            [{"alert_count": 1.5}]
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_switch_rate(
            [{"alert_count": True}]
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_switch_rate(
            [{"alert_count": -1}]
        )
