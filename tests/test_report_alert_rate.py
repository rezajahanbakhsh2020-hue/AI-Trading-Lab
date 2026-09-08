from __future__ import annotations

import pytest

from src.evaluation.report_alert_rate import (
    build_alert_rate_summary,
    calculate_alert_free_rate,
    calculate_alert_rate,
    count_alert_free_snapshots,
    count_alert_snapshots,
)


def test_calculate_alert_rate():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    assert calculate_alert_rate(history) == 3 / 5


def test_calculate_alert_rate_all_alerts():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert calculate_alert_rate(history) == 1.0


def test_calculate_alert_rate_no_alerts():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_rate(history) == 0.0


def test_calculate_alert_rate_empty():
    with pytest.raises(ValueError):
        calculate_alert_rate([])


def test_calculate_alert_rate_rejects_invalid_history():
    with pytest.raises(TypeError):
        calculate_alert_rate({})


def test_calculate_alert_rate_rejects_invalid_item():
    with pytest.raises(TypeError):
        calculate_alert_rate(
            [{"alert_count": 1}, "invalid"]
        )


def test_calculate_alert_rate_rejects_missing_count():
    with pytest.raises(ValueError):
        calculate_alert_rate(
            [{"value": 1}]
        )


def test_calculate_alert_rate_rejects_non_integer_count():
    with pytest.raises(ValueError):
        calculate_alert_rate(
            [{"alert_count": 1.5}]
        )


def test_calculate_alert_rate_rejects_boolean_count():
    with pytest.raises(ValueError):
        calculate_alert_rate(
            [{"alert_count": True}]
        )


def test_calculate_alert_rate_rejects_negative_count():
    with pytest.raises(ValueError):
        calculate_alert_rate(
            [{"alert_count": -1}]
        )


def test_calculate_alert_free_rate():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
    ]

    assert calculate_alert_free_rate(history) == 0.5


def test_calculate_alert_free_rate_all_alerts():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
    ]

    assert calculate_alert_free_rate(history) == 0.0


def test_calculate_alert_free_rate_no_alerts():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_free_rate(history) == 1.0


def test_count_alert_snapshots():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
    ]

    assert count_alert_snapshots(history) == 2


def test_count_alert_snapshots_empty():
    assert count_alert_snapshots([]) == 0


def test_count_alert_free_snapshots():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
    ]

    assert count_alert_free_snapshots(history) == 2


def test_count_alert_free_snapshots_all_alerts():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
    ]

    assert count_alert_free_snapshots(history) == 0


def test_count_alert_free_snapshots_empty():
    assert count_alert_free_snapshots([]) == 0


def test_build_alert_rate_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    result = build_alert_rate_summary(history)

    assert result == {
        "snapshot_count": 5,
        "alert_snapshots": 3,
        "alert_free_snapshots": 2,
        "alert_rate": 3 / 5,
        "alert_free_rate": 2 / 5,
    }


def test_build_alert_rate_summary_empty():
    with pytest.raises(ValueError):
        build_alert_rate_summary([])
