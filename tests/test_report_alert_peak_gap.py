from __future__ import annotations

import pytest

from src.evaluation.report_alert_peak_gap import (
    build_alert_peak_gap_summary,
    calculate_average_peak_gap,
    calculate_longest_peak_gap,
    calculate_peak_alert_count,
    calculate_peak_gaps,
    find_peak_positions,
)


def test_calculate_peak_alert_count():
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 3},
        {"alert_count": 5},
    ]

    assert calculate_peak_alert_count(history) == 5


def test_find_peak_positions():
    history = [
        {"alert_count": 5},
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 1},
        {"alert_count": 5},
    ]

    assert find_peak_positions(history) == [0, 2, 4]


def test_calculate_peak_gaps():
    history = [
        {"alert_count": 5},
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 1},
        {"alert_count": 5},
    ]

    assert calculate_peak_gaps(history) == [1, 1]


def test_calculate_longest_peak_gap():
    history = [
        {"alert_count": 5},
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 5},
        {"alert_count": 4},
        {"alert_count": 5},
    ]

    assert calculate_longest_peak_gap(history) == 3


def test_calculate_average_peak_gap():
    history = [
        {"alert_count": 5},
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 4},
        {"alert_count": 3},
        {"alert_count": 5},
    ]

    assert calculate_average_peak_gap(history) == pytest.approx(2.0)


def test_adjacent_peak_observations():
    history = [
        {"alert_count": 5},
        {"alert_count": 5},
        {"alert_count": 2},
        {"alert_count": 5},
    ]

    assert find_peak_positions(history) == [0, 1, 3]
    assert calculate_peak_gaps(history) == [0, 1]


def test_peak_occurs_once():
    history = [
        {"alert_count": 1},
        {"alert_count": 8},
        {"alert_count": 2},
    ]

    assert find_peak_positions(history) == [1]
    assert calculate_peak_gaps(history) == []
    assert calculate_longest_peak_gap(history) == 0
    assert calculate_average_peak_gap(history) == 0.0


def test_peak_occurs_everywhere():
    history = [
        {"alert_count": 4},
        {"alert_count": 4},
        {"alert_count": 4},
        {"alert_count": 4},
    ]

    assert find_peak_positions(history) == [0, 1, 2, 3]
    assert calculate_peak_gaps(history) == [0, 0, 0]
    assert calculate_longest_peak_gap(history) == 0
    assert calculate_average_peak_gap(history) == 0.0


def test_peak_at_start_and_end():
    history = [
        {"alert_count": 7},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 7},
    ]

    assert find_peak_positions(history) == [0, 3]
    assert calculate_peak_gaps(history) == [2]


def test_all_zero_history():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_peak_alert_count(history) == 0
    assert find_peak_positions(history) == [0, 1, 2]
    assert calculate_peak_gaps(history) == [0, 0]


def test_single_snapshot():
    history = [
        {"alert_count": 6},
    ]

    assert find_peak_positions(history) == [0]
    assert calculate_peak_gaps(history) == []


def test_build_summary():
    history = [
        {"alert_count": 5},
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 5},
        {"alert_count": 4},
        {"alert_count": 5},
    ]

    result = build_alert_peak_gap_summary(history)

    assert result == {
        "snapshot_count": 7,
        "peak_alert_count": 5,
        "peak_occurrence_count": 3,
        "peak_positions": [0, 4, 6],
        "peak_gap_count": 2,
        "peak_gaps": [3, 1],
        "average_peak_gap": pytest.approx(2.0),
        "longest_peak_gap": 3,
    }


def test_build_summary_single_peak():
    history = [
        {"alert_count": 2},
        {"alert_count": 9},
        {"alert_count": 3},
    ]

    result = build_alert_peak_gap_summary(history)

    assert result == {
        "snapshot_count": 3,
        "peak_alert_count": 9,
        "peak_occurrence_count": 1,
        "peak_positions": [1],
        "peak_gap_count": 0,
        "peak_gaps": [],
        "average_peak_gap": 0.0,
        "longest_peak_gap": 0,
    }


def test_empty_history():
    with pytest.raises(ValueError):
        calculate_peak_gaps([])

    with pytest.raises(ValueError):
        build_alert_peak_gap_summary([])


def test_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_peak_gaps({})


def test_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_peak_gaps(
            [{"alert_count": 5}, "invalid"]
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_peak_gaps(
            [{"value": 5}]
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_peak_gaps(
            [{"alert_count": 1.5}]
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_peak_gaps(
            [{"alert_count": True}]
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_peak_gaps(
            [{"alert_count": -1}]
        )
