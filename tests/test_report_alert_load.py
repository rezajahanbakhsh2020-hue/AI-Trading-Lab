from __future__ import annotations

import pytest

from src.evaluation.report_alert_load import (
    build_alert_load_summary,
    calculate_alert_load,
    calculate_average_alert_load,
    calculate_peak_alert_load,
)


def test_calculate_alert_load():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    assert calculate_alert_load(history) == 6


def test_calculate_alert_load_empty():
    assert calculate_alert_load([]) == 0


def test_calculate_average_alert_load():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    assert calculate_average_alert_load(history) == 6 / 5


def test_calculate_average_alert_load_empty():
    with pytest.raises(ValueError):
        calculate_average_alert_load([])


def test_calculate_peak_alert_load():
    history = [
        {"alert_count": 1},
        {"alert_count": 4},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert calculate_peak_alert_load(history) == 4


def test_calculate_peak_alert_load_empty():
    with pytest.raises(ValueError):
        calculate_peak_alert_load([])


def test_build_alert_load_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    result = build_alert_load_summary(history)

    assert result == {
        "snapshot_count": 5,
        "total_alert_load": 6,
        "average_alert_load": 6 / 5,
        "peak_alert_load": 3,
    }


def test_build_alert_load_summary_single_snapshot():
    history = [
        {"alert_count": 4},
    ]

    result = build_alert_load_summary(history)

    assert result == {
        "snapshot_count": 1,
        "total_alert_load": 4,
        "average_alert_load": 4.0,
        "peak_alert_load": 4,
    }


def test_build_alert_load_summary_empty():
    with pytest.raises(ValueError):
        build_alert_load_summary([])


def test_rejects_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_alert_load({})


def test_rejects_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_alert_load(
            [{"alert_count": 1}, "invalid"]
        )


def test_rejects_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_load(
            [{"value": 1}]
        )


def test_rejects_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_load(
            [{"alert_count": 1.5}]
        )


def test_rejects_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_load(
            [{"alert_count": True}]
        )


def test_rejects_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_load(
            [{"alert_count": -1}]
        )
