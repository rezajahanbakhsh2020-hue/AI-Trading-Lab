from __future__ import annotations

import pytest

from src.evaluation.report_alert_peak_ratio import (
    build_alert_peak_ratio_summary,
    calculate_non_peak_ratio,
    calculate_peak_alert_count,
    calculate_peak_occurrence_percentage,
    calculate_peak_occurrence_ratio,
    count_peak_observations,
)


def test_calculate_peak_alert_count():
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 3},
        {"alert_count": 5},
    ]

    assert calculate_peak_alert_count(history) == 5


def test_count_peak_observations():
    history = [
        {"alert_count": 5},
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 1},
        {"alert_count": 5},
    ]

    assert count_peak_observations(history) == 3


def test_calculate_peak_occurrence_ratio():
    history = [
        {"alert_count": 5},
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 1},
        {"alert_count": 5},
    ]

    assert calculate_peak_occurrence_ratio(history) == pytest.approx(0.6)


def test_calculate_peak_occurrence_percentage():
    history = [
        {"alert_count": 5},
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 1},
        {"alert_count": 5},
    ]

    assert calculate_peak_occurrence_percentage(history) == pytest.approx(
        60.0
    )


def test_calculate_non_peak_ratio():
    history = [
        {"alert_count": 5},
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 1},
        {"alert_count": 5},
    ]

    assert calculate_non_peak_ratio(history) == pytest.approx(0.4)


def test_peak_occurs_once():
    history = [
        {"alert_count": 1},
        {"alert_count": 8},
        {"alert_count": 2},
    ]

    assert count_peak_observations(history) == 1
    assert calculate_peak_occurrence_ratio(history) == pytest.approx(
        1 / 3
    )
    assert calculate_peak_occurrence_percentage(history) == pytest.approx(
        100 / 3
    )
    assert calculate_non_peak_ratio(history) == pytest.approx(
        2 / 3
    )


def test_peak_occurs_everywhere():
    history = [
        {"alert_count": 4},
        {"alert_count": 4},
        {"alert_count": 4},
        {"alert_count": 4},
    ]

    assert count_peak_observations(history) == 4
    assert calculate_peak_occurrence_ratio(history) == 1.0
    assert calculate_peak_occurrence_percentage(history) == 100.0
    assert calculate_non_peak_ratio(history) == 0.0


def test_all_zero_history():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_peak_alert_count(history) == 0
    assert count_peak_observations(history) == 3
    assert calculate_peak_occurrence_ratio(history) == 1.0
    assert calculate_non_peak_ratio(history) == 0.0


def test_single_snapshot():
    history = [
        {"alert_count": 6},
    ]

    assert count_peak_observations(history) == 1
    assert calculate_peak_occurrence_ratio(history) == 1.0
    assert calculate_peak_occurrence_percentage(history) == 100.0
    assert calculate_non_peak_ratio(history) == 0.0


def test_build_summary():
    history = [
        {"alert_count": 5},
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 1},
        {"alert_count": 5},
    ]

    result = build_alert_peak_ratio_summary(history)

    assert result == {
        "snapshot_count": 5,
        "peak_alert_count": 5,
        "peak_observation_count": 3,
        "peak_occurrence_ratio": pytest.approx(0.6),
        "peak_occurrence_percentage": pytest.approx(60.0),
        "non_peak_ratio": pytest.approx(0.4),
    }


def test_build_summary_single_peak():
    history = [
        {"alert_count": 2},
        {"alert_count": 9},
        {"alert_count": 3},
    ]

    result = build_alert_peak_ratio_summary(history)

    assert result == {
        "snapshot_count": 3,
        "peak_alert_count": 9,
        "peak_observation_count": 1,
        "peak_occurrence_ratio": pytest.approx(1 / 3),
        "peak_occurrence_percentage": pytest.approx(100 / 3),
        "non_peak_ratio": pytest.approx(2 / 3),
    }


def test_empty_history():
    with pytest.raises(ValueError):
        calculate_peak_occurrence_ratio([])

    with pytest.raises(ValueError):
        build_alert_peak_ratio_summary([])


def test_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_peak_occurrence_ratio({})


def test_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_peak_occurrence_ratio(
            [{"alert_count": 5}, "invalid"]
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_peak_occurrence_ratio(
            [{"value": 5}]
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_peak_occurrence_ratio(
            [{"alert_count": 1.5}]
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_peak_occurrence_ratio(
            [{"alert_count": True}]
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_peak_occurrence_ratio(
            [{"alert_count": -1}]
        )
