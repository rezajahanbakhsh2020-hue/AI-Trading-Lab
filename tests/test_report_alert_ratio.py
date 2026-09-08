from __future__ import annotations

import pytest

from src.evaluation.report_alert_ratio import (
    build_alert_ratio_summary,
    calculate_alert_free_ratio,
    calculate_alert_ratio,
)


def test_calculate_alert_ratio():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    assert calculate_alert_ratio(history) == pytest.approx(
        3 / 5
    )


def test_calculate_alert_ratio_all_active():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert calculate_alert_ratio(history) == 1.0


def test_calculate_alert_ratio_none_active():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_ratio(history) == 0.0


def test_calculate_alert_ratio_single_active():
    history = [
        {"alert_count": 0},
        {"alert_count": 5},
    ]

    assert calculate_alert_ratio(history) == pytest.approx(
        1 / 2
    )


def test_calculate_alert_ratio_empty():
    with pytest.raises(ValueError):
        calculate_alert_ratio([])


def test_calculate_alert_free_ratio():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    assert calculate_alert_free_ratio(history) == pytest.approx(
        2 / 5
    )


def test_calculate_alert_free_ratio_all_active():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
    ]

    assert calculate_alert_free_ratio(history) == 0.0


def test_calculate_alert_free_ratio_none_active():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_free_ratio(history) == 1.0


def test_build_alert_ratio_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    result = build_alert_ratio_summary(history)

    assert result == {
        "snapshot_count": 5,
        "alert_ratio": pytest.approx(3 / 5),
        "alert_free_ratio": pytest.approx(2 / 5),
    }


def test_build_alert_ratio_summary_all_active():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
    ]

    result = build_alert_ratio_summary(history)

    assert result == {
        "snapshot_count": 2,
        "alert_ratio": 1.0,
        "alert_free_ratio": 0.0,
    }


def test_build_alert_ratio_summary_empty():
    with pytest.raises(ValueError):
        build_alert_ratio_summary([])


def test_rejects_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_alert_ratio({})


def test_rejects_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_alert_ratio(
            [{"alert_count": 1}, "invalid"]
        )


def test_rejects_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_ratio(
            [{"value": 1}]
        )


def test_rejects_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_ratio(
            [{"alert_count": 1.5}]
        )


def test_rejects_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_ratio(
            [{"alert_count": True}]
        )


def test_rejects_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_ratio(
            [{"alert_count": -1}]
        )
