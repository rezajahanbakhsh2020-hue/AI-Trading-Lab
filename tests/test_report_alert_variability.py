from __future__ import annotations

import pytest

from src.evaluation.report_alert_variability import (
    build_alert_variability_summary,
    calculate_alert_range,
    calculate_alert_variability,
    calculate_alert_variability_ratio,
)


def test_calculate_alert_variability():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    expected = (1.36) ** 0.5

    assert calculate_alert_variability(history) == pytest.approx(
        expected
    )


def test_calculate_alert_variability_uniform():
    history = [
        {"alert_count": 2},
        {"alert_count": 2},
        {"alert_count": 2},
    ]

    assert calculate_alert_variability(history) == 0.0


def test_calculate_alert_variability_empty():
    with pytest.raises(ValueError):
        calculate_alert_variability([])


def test_calculate_alert_range():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 4},
        {"alert_count": 2},
    ]

    assert calculate_alert_range(history) == 4


def test_calculate_alert_range_uniform():
    history = [
        {"alert_count": 3},
        {"alert_count": 3},
    ]

    assert calculate_alert_range(history) == 0


def test_calculate_alert_range_empty():
    with pytest.raises(ValueError):
        calculate_alert_range([])


def test_calculate_alert_variability_ratio():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    variability = (1.36) ** 0.5
    mean = 6 / 5

    assert calculate_alert_variability_ratio(history) == pytest.approx(
        variability / mean
    )


def test_calculate_alert_variability_ratio_zero_mean():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_variability_ratio(history) == 0.0


def test_calculate_alert_variability_ratio_empty():
    with pytest.raises(ValueError):
        calculate_alert_variability_ratio([])


def test_build_alert_variability_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    variability = (1.36) ** 0.5
    ratio = variability / (6 / 5)

    result = build_alert_variability_summary(history)

    assert result["snapshot_count"] == 5
    assert result["alert_variability"] == pytest.approx(
        variability
    )
    assert result["alert_range"] == 3
    assert result["alert_variability_ratio"] == pytest.approx(
        ratio
    )


def test_build_alert_variability_summary_single_snapshot():
    history = [
        {"alert_count": 4},
    ]

    result = build_alert_variability_summary(history)

    assert result == {
        "snapshot_count": 1,
        "alert_variability": 0.0,
        "alert_range": 0,
        "alert_variability_ratio": 0.0,
    }


def test_build_alert_variability_summary_empty():
    with pytest.raises(ValueError):
        build_alert_variability_summary([])


def test_rejects_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_alert_variability({})


def test_rejects_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_alert_variability(
            [{"alert_count": 1}, "invalid"]
        )


def test_rejects_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_variability(
            [{"value": 1}]
        )


def test_rejects_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_variability(
            [{"alert_count": 1.5}]
        )


def test_rejects_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_variability(
            [{"alert_count": True}]
        )


def test_rejects_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_variability(
            [{"alert_count": -1}]
        )
