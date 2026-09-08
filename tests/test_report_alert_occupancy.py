from __future__ import annotations

import pytest

from src.evaluation.report_alert_occupancy import (
    build_alert_occupancy_summary,
    calculate_alert_occupancy,
    calculate_alert_vacancy,
)


def test_calculate_alert_occupancy():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    assert calculate_alert_occupancy(history) == pytest.approx(3 / 5)


def test_calculate_alert_vacancy():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    assert calculate_alert_vacancy(history) == pytest.approx(2 / 5)


def test_all_active():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert calculate_alert_occupancy(history) == 1.0
    assert calculate_alert_vacancy(history) == 0.0


def test_all_inactive():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_occupancy(history) == 0.0
    assert calculate_alert_vacancy(history) == 1.0


def test_single_active_snapshot():
    history = [{"alert_count": 5}]

    assert calculate_alert_occupancy(history) == 1.0
    assert calculate_alert_vacancy(history) == 0.0


def test_single_inactive_snapshot():
    history = [{"alert_count": 0}]

    assert calculate_alert_occupancy(history) == 0.0
    assert calculate_alert_vacancy(history) == 1.0


def test_empty_history():
    with pytest.raises(ValueError):
        calculate_alert_occupancy([])


def test_build_alert_occupancy_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    result = build_alert_occupancy_summary(history)

    assert result == {
        "snapshot_count": 5,
        "active_snapshots": 3,
        "inactive_snapshots": 2,
        "alert_occupancy": pytest.approx(3 / 5),
        "alert_vacancy": pytest.approx(2 / 5),
    }


def test_build_summary_all_active():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
    ]

    result = build_alert_occupancy_summary(history)

    assert result == {
        "snapshot_count": 2,
        "active_snapshots": 2,
        "inactive_snapshots": 0,
        "alert_occupancy": 1.0,
        "alert_vacancy": 0.0,
    }


def test_build_summary_all_inactive():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    result = build_alert_occupancy_summary(history)

    assert result == {
        "snapshot_count": 2,
        "active_snapshots": 0,
        "inactive_snapshots": 2,
        "alert_occupancy": 0.0,
        "alert_vacancy": 1.0,
    }


def test_build_summary_empty_history():
    with pytest.raises(ValueError):
        build_alert_occupancy_summary([])


def test_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_alert_occupancy({})


def test_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_alert_occupancy(
            [{"alert_count": 1}, "invalid"]
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_occupancy(
            [{"value": 1}]
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_occupancy(
            [{"alert_count": 1.5}]
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_occupancy(
            [{"alert_count": True}]
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_occupancy(
            [{"alert_count": -1}]
        )
