from __future__ import annotations

import pytest

from src.evaluation.report_alert_density import (
    build_alert_density_summary,
    calculate_alert_active_density,
    calculate_alert_density,
    calculate_alert_free_density,
)


def test_calculate_alert_density():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    assert calculate_alert_density(history) == 6 / 5


def test_calculate_alert_density_all_alerts():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
    ]

    assert calculate_alert_density(history) == 3.0


def test_calculate_alert_density_no_alerts():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_density(history) == 0.0


def test_calculate_alert_density_empty():
    with pytest.raises(ValueError):
        calculate_alert_density([])


def test_calculate_alert_free_density():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
    ]

    assert calculate_alert_free_density(history) == 0.5


def test_calculate_alert_free_density_all_free():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_free_density(history) == 1.0


def test_calculate_alert_free_density_empty():
    with pytest.raises(ValueError):
        calculate_alert_free_density([])


def test_calculate_alert_active_density():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
    ]

    assert calculate_alert_active_density(history) == 0.5


def test_calculate_alert_active_density_all_active():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
    ]

    assert calculate_alert_active_density(history) == 1.0


def test_calculate_alert_active_density_empty():
    with pytest.raises(ValueError):
        calculate_alert_active_density([])


def test_build_alert_density_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    result = build_alert_density_summary(history)

    assert result == {
        "snapshot_count": 5,
        "alert_density": 6 / 5,
        "alert_free_density": 2 / 5,
        "alert_active_density": 3 / 5,
    }


def test_build_alert_density_summary_single_snapshot():
    history = [
        {"alert_count": 4},
    ]

    result = build_alert_density_summary(history)

    assert result == {
        "snapshot_count": 1,
        "alert_density": 4.0,
        "alert_free_density": 0.0,
        "alert_active_density": 1.0,
    }


def test_build_alert_density_summary_empty():
    with pytest.raises(ValueError):
        build_alert_density_summary([])


def test_rejects_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_alert_density({})


def test_rejects_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_alert_density(
            [{"alert_count": 1}, "invalid"]
        )


def test_rejects_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_density(
            [{"value": 1}]
        )


def test_rejects_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_density(
            [{"alert_count": 1.5}]
        )


def test_rejects_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_density(
            [{"alert_count": True}]
        )


def test_rejects_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_density(
            [{"alert_count": -1}]
        )
