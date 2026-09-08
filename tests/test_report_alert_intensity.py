from __future__ import annotations

import pytest

from src.evaluation.report_alert_intensity import (
    build_alert_intensity_summary,
    calculate_average_alert_intensity,
    calculate_max_alert_intensity,
    calculate_min_alert_intensity,
    calculate_total_alerts,
)


def test_calculate_total_alerts():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    assert calculate_total_alerts(history) == 6


def test_calculate_total_alerts_empty():
    assert calculate_total_alerts([]) == 0


def test_calculate_average_alert_intensity():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    assert calculate_average_alert_intensity(history) == 6 / 5


def test_calculate_average_alert_intensity_empty():
    with pytest.raises(ValueError):
        calculate_average_alert_intensity([])


def test_calculate_max_alert_intensity():
    history = [
        {"alert_count": 1},
        {"alert_count": 4},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert calculate_max_alert_intensity(history) == 4


def test_calculate_max_alert_intensity_empty():
    with pytest.raises(ValueError):
        calculate_max_alert_intensity([])


def test_calculate_min_alert_intensity():
    history = [
        {"alert_count": 3},
        {"alert_count": 1},
        {"alert_count": 4},
        {"alert_count": 2},
    ]

    assert calculate_min_alert_intensity(history) == 1


def test_calculate_min_alert_intensity_with_zero():
    history = [
        {"alert_count": 3},
        {"alert_count": 0},
        {"alert_count": 4},
    ]

    assert calculate_min_alert_intensity(history) == 0


def test_calculate_min_alert_intensity_empty():
    with pytest.raises(ValueError):
        calculate_min_alert_intensity([])


def test_build_alert_intensity_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    result = build_alert_intensity_summary(history)

    assert result == {
        "snapshot_count": 5,
        "total_alerts": 6,
        "average_alert_intensity": 6 / 5,
        "max_alert_intensity": 3,
        "min_alert_intensity": 0,
    }


def test_build_alert_intensity_summary_single_snapshot():
    history = [
        {"alert_count": 4},
    ]

    result = build_alert_intensity_summary(history)

    assert result == {
        "snapshot_count": 1,
        "total_alerts": 4,
        "average_alert_intensity": 4.0,
        "max_alert_intensity": 4,
        "min_alert_intensity": 4,
    }


def test_build_alert_intensity_summary_empty():
    with pytest.raises(ValueError):
        build_alert_intensity_summary([])


def test_rejects_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_total_alerts({})


def test_rejects_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_total_alerts(
            [{"alert_count": 1}, "invalid"]
        )


def test_rejects_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_total_alerts(
            [{"value": 1}]
        )


def test_rejects_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_total_alerts(
            [{"alert_count": 1.5}]
        )


def test_rejects_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_total_alerts(
            [{"alert_count": True}]
        )


def test_rejects_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_total_alerts(
            [{"alert_count": -1}]
        )
