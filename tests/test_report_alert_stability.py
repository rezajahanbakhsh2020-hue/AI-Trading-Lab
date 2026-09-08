from __future__ import annotations

import pytest

from src.evaluation.report_alert_stability import (
    build_alert_stability_summary,
    calculate_alert_instability,
    calculate_alert_stability,
)


def test_calculate_alert_stability():
    history = [
        {"alert_count": 2},
        {"alert_count": 2},
        {"alert_count": 2},
    ]

    assert calculate_alert_stability(history) == 1.0


def test_calculate_alert_stability_variable_history():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    mean = 6 / 5
    standard_deviation = (1.36) ** 0.5
    coefficient_of_variation = (
        standard_deviation / mean
    )
    expected = 1 / (
        1 + coefficient_of_variation
    )

    assert calculate_alert_stability(history) == pytest.approx(
        expected
    )


def test_calculate_alert_stability_all_zero():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_stability(history) == 1.0


def test_calculate_alert_stability_single_snapshot():
    history = [
        {"alert_count": 5},
    ]

    assert calculate_alert_stability(history) == 1.0


def test_calculate_alert_stability_empty():
    with pytest.raises(ValueError):
        calculate_alert_stability([])


def test_calculate_alert_instability():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    stability = calculate_alert_stability(history)

    assert calculate_alert_instability(history) == pytest.approx(
        1.0 - stability
    )


def test_calculate_alert_instability_perfect():
    history = [
        {"alert_count": 4},
        {"alert_count": 4},
    ]

    assert calculate_alert_instability(history) == 0.0


def test_calculate_alert_instability_all_zero():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_instability(history) == 0.0


def test_build_alert_stability_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    result = build_alert_stability_summary(history)

    stability = calculate_alert_stability(history)

    assert result["snapshot_count"] == 5
    assert result["alert_stability"] == pytest.approx(
        stability
    )
    assert result["alert_instability"] == pytest.approx(
        1.0 - stability
    )


def test_build_alert_stability_summary_perfect():
    history = [
        {"alert_count": 3},
        {"alert_count": 3},
        {"alert_count": 3},
    ]

    result = build_alert_stability_summary(history)

    assert result == {
        "snapshot_count": 3,
        "alert_stability": 1.0,
        "alert_instability": 0.0,
    }


def test_build_alert_stability_summary_all_zero():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    result = build_alert_stability_summary(history)

    assert result == {
        "snapshot_count": 2,
        "alert_stability": 1.0,
        "alert_instability": 0.0,
    }


def test_build_alert_stability_summary_empty():
    with pytest.raises(ValueError):
        build_alert_stability_summary([])


def test_rejects_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_alert_stability({})


def test_rejects_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_alert_stability(
            [{"alert_count": 1}, "invalid"]
        )


def test_rejects_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_stability(
            [{"value": 1}]
        )


def test_rejects_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_stability(
            [{"alert_count": 1.5}]
        )


def test_rejects_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_stability(
            [{"alert_count": True}]
        )


def test_rejects_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_stability(
            [{"alert_count": -1}]
        )
