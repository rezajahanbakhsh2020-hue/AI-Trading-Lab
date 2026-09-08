from __future__ import annotations

import pytest

from src.evaluation.report_alert_volatility import (
    build_alert_volatility_summary,
    calculate_alert_change_range,
    calculate_alert_volatility,
)


def test_calculate_alert_volatility():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 2},
        {"alert_count": 6},
    ]

    changes = [2, -1, 4]
    mean_change = 5 / 3

    variance = sum(
        (change - mean_change) ** 2
        for change in changes
    ) / 3

    expected = variance ** 0.5

    assert calculate_alert_volatility(history) == pytest.approx(
        expected
    )


def test_calculate_alert_volatility_constant_changes():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 4},
    ]

    assert calculate_alert_volatility(history) == 0.0


def test_calculate_alert_volatility_single_snapshot():
    history = [
        {"alert_count": 5},
    ]

    assert calculate_alert_volatility(history) == 0.0


def test_calculate_alert_volatility_empty():
    with pytest.raises(ValueError):
        calculate_alert_volatility([])


def test_calculate_alert_change_range():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 2},
        {"alert_count": 6},
    ]

    assert calculate_alert_change_range(history) == 5


def test_calculate_alert_change_range_constant():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
    ]

    assert calculate_alert_change_range(history) == 0


def test_calculate_alert_change_range_single_snapshot():
    history = [
        {"alert_count": 5},
    ]

    assert calculate_alert_change_range(history) == 0


def test_calculate_alert_change_range_empty():
    with pytest.raises(ValueError):
        calculate_alert_change_range([])


def test_build_alert_volatility_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 2},
        {"alert_count": 6},
    ]

    result = build_alert_volatility_summary(history)

    assert result["snapshot_count"] == 4
    assert result["alert_volatility"] == pytest.approx(
        calculate_alert_volatility(history)
    )
    assert result["alert_change_range"] == 5


def test_build_alert_volatility_summary_constant():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
    ]

    result = build_alert_volatility_summary(history)

    assert result == {
        "snapshot_count": 3,
        "alert_volatility": 0.0,
        "alert_change_range": 0,
    }


def test_build_alert_volatility_summary_empty():
    with pytest.raises(ValueError):
        build_alert_volatility_summary([])


def test_rejects_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_alert_volatility({})


def test_rejects_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_alert_volatility(
            [{"alert_count": 1}, "invalid"]
        )


def test_rejects_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_volatility(
            [{"value": 1}]
        )


def test_rejects_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_volatility(
            [{"alert_count": 1.5}]
        )


def test_rejects_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_volatility(
            [{"alert_count": True}]
        )


def test_rejects_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_volatility(
            [{"alert_count": -1}]
        )
