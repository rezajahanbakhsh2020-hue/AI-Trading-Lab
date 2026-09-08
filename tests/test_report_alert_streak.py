from __future__ import annotations

import pytest

from src.evaluation.report_alert_streak import (
    build_alert_streak_summary,
    calculate_alert_streaks,
    calculate_average_alert_streak,
    calculate_longest_alert_streak,
    calculate_single_snapshot_streaks,
    count_alert_streaks,
)


def test_calculate_alert_streaks():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 0},
        {"alert_count": 5},
    ]

    assert calculate_alert_streaks(history) == [2, 2, 1]


def test_calculate_alert_streaks_empty():
    assert calculate_alert_streaks([]) == []


def test_calculate_alert_streaks_no_alerts():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_streaks(history) == []


def test_calculate_alert_streaks_all_alerts():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert calculate_alert_streaks(history) == [3]


def test_calculate_alert_streaks_single_alert():
    assert calculate_alert_streaks(
        [{"alert_count": 1}]
    ) == [1]


def test_calculate_alert_streaks_trailing_streak():
    history = [
        {"alert_count": 0},
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert calculate_alert_streaks(history) == [3]


def test_calculate_alert_streaks_rejects_invalid_history():
    with pytest.raises(TypeError):
        calculate_alert_streaks({})


def test_calculate_alert_streaks_rejects_invalid_item():
    with pytest.raises(TypeError):
        calculate_alert_streaks(
            [{"alert_count": 1}, "invalid"]
        )


def test_calculate_alert_streaks_rejects_missing_count():
    with pytest.raises(ValueError):
        calculate_alert_streaks(
            [{"value": 1}]
        )


def test_calculate_alert_streaks_rejects_non_integer_count():
    with pytest.raises(ValueError):
        calculate_alert_streaks(
            [{"alert_count": 1.5}]
        )


def test_calculate_alert_streaks_rejects_boolean_count():
    with pytest.raises(ValueError):
        calculate_alert_streaks(
            [{"alert_count": True}]
        )


def test_calculate_alert_streaks_rejects_negative_count():
    with pytest.raises(ValueError):
        calculate_alert_streaks(
            [{"alert_count": -1}]
        )


def test_calculate_longest_alert_streak():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 0},
    ]

    assert calculate_longest_alert_streak(history) == 3


def test_calculate_longest_alert_streak_no_alerts():
    assert calculate_longest_alert_streak(
        [{"alert_count": 0}]
    ) == 0


def test_calculate_average_alert_streak():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 0},
        {"alert_count": 4},
        {"alert_count": 5},
    ]

    assert calculate_average_alert_streak(history) == 2.0


def test_calculate_average_alert_streak_no_alerts():
    assert calculate_average_alert_streak(
        [{"alert_count": 0}]
    ) == 0.0


def test_count_alert_streaks():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
    ]

    assert count_alert_streaks(history) == 3


def test_count_alert_streaks_empty():
    assert count_alert_streaks([]) == 0


def test_calculate_single_snapshot_streaks():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 0},
        {"alert_count": 4},
        {"alert_count": 0},
    ]

    assert calculate_single_snapshot_streaks(history) == 2


def test_calculate_single_snapshot_streaks_none():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
    ]

    assert calculate_single_snapshot_streaks(history) == 0


def test_build_alert_streak_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 0},
        {"alert_count": 4},
        {"alert_count": 0},
    ]

    result = build_alert_streak_summary(history)

    assert result == {
        "snapshot_count": 7,
        "streak_count": 3,
        "streaks": [1, 2, 1],
        "total_alert_snapshots": 4,
        "longest_streak": 2,
        "average_streak": 4 / 3,
        "single_snapshot_streaks": 2,
    }


def test_build_alert_streak_summary_empty():
    result = build_alert_streak_summary([])

    assert result == {
        "snapshot_count": 0,
        "streak_count": 0,
        "streaks": [],
        "total_alert_snapshots": 0,
        "longest_streak": 0,
        "average_streak": 0.0,
        "single_snapshot_streaks": 0,
    }
