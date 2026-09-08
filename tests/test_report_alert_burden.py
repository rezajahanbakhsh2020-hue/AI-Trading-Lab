from __future__ import annotations

import pytest

from src.evaluation.report_alert_burden import (
    build_alert_burden_summary,
    calculate_alert_burden,
    calculate_average_alert_burden,
    calculate_burden_ratio,
    calculate_burdened_snapshots,
)


def test_calculate_alert_burden():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    assert calculate_alert_burden(history) == 6


def test_calculate_alert_burden_empty():
    assert calculate_alert_burden([]) == 0


def test_calculate_average_alert_burden():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    assert calculate_average_alert_burden(history) == 6 / 5


def test_calculate_average_alert_burden_empty():
    with pytest.raises(ValueError):
        calculate_average_alert_burden([])


def test_calculate_burdened_snapshots():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    assert calculate_burdened_snapshots(history) == 3


def test_calculate_burdened_snapshots_no_alerts():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_burdened_snapshots(history) == 0


def test_calculate_burden_ratio():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
    ]

    assert calculate_burden_ratio(history) == 0.5


def test_calculate_burden_ratio_all_burdened():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
    ]

    assert calculate_burden_ratio(history) == 1.0


def test_calculate_burden_ratio_empty():
    with pytest.raises(ValueError):
        calculate_burden_ratio([])


def test_build_alert_burden_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    result = build_alert_burden_summary(history)

    assert result == {
        "snapshot_count": 5,
        "total_alert_burden": 6,
        "average_alert_burden": 6 / 5,
        "burdened_snapshots": 3,
        "burden_ratio": 3 / 5,
    }


def test_build_alert_burden_summary_single_snapshot():
    history = [
        {"alert_count": 4},
    ]

    result = build_alert_burden_summary(history)

    assert result == {
        "snapshot_count": 1,
        "total_alert_burden": 4,
        "average_alert_burden": 4.0,
        "burdened_snapshots": 1,
        "burden_ratio": 1.0,
    }


def test_build_alert_burden_summary_empty():
    with pytest.raises(ValueError):
        build_alert_burden_summary([])


def test_rejects_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_alert_burden({})


def test_rejects_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_alert_burden(
            [{"alert_count": 1}, "invalid"]
        )


def test_rejects_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_burden(
            [{"value": 1}]
        )


def test_rejects_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_burden(
            [{"alert_count": 1.5}]
        )


def test_rejects_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_burden(
            [{"alert_count": True}]
        )


def test_rejects_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_burden(
            [{"alert_count": -1}]
        )
