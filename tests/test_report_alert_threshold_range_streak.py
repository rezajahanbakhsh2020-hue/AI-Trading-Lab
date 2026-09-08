from __future__ import annotations

import pytest

from src.evaluation.report_alert_threshold_range_streak import (
    build_outside_range_streak_summary,
    calculate_longest_outside_range_streak,
    calculate_mean_outside_range_streak,
    calculate_total_outside_range_duration,
    count_outside_range_streaks,
    find_outside_range_streaks,
    is_outside_range,
)


def test_is_outside_range():
    assert is_outside_range(1, 3, 6) is True
    assert is_outside_range(9, 3, 6) is True


def test_is_outside_range_inside():
    assert is_outside_range(3, 3, 6) is False
    assert is_outside_range(4, 3, 6) is False
    assert is_outside_range(6, 3, 6) is False


def test_find_outside_range_streaks():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 8},
        {"alert_count": 4},
        {"alert_count": 1},
    ]

    assert find_outside_range_streaks(
        history,
        3,
        6,
    ) == [2, 2, 1]


def test_count_outside_range_streaks():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 8},
        {"alert_count": 4},
        {"alert_count": 1},
    ]

    assert count_outside_range_streaks(
        history,
        3,
        6,
    ) == 3


def test_calculate_total_outside_range_duration():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 8},
        {"alert_count": 4},
        {"alert_count": 1},
    ]

    assert calculate_total_outside_range_duration(
        history,
        3,
        6,
    ) == 5


def test_calculate_mean_outside_range_streak():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 8},
        {"alert_count": 4},
        {"alert_count": 1},
    ]

    assert calculate_mean_outside_range_streak(
        history,
        3,
        6,
    ) == pytest.approx(5 / 3)


def test_calculate_longest_outside_range_streak():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 8},
        {"alert_count": 4},
        {"alert_count": 1},
    ]

    assert calculate_longest_outside_range_streak(
        history,
        3,
        6,
    ) == 2


def test_no_outside_range_streaks():
    history = [
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 6},
    ]

    assert find_outside_range_streaks(
        history,
        3,
        6,
    ) == []

    assert count_outside_range_streaks(
        history,
        3,
        6,
    ) == 0

    assert calculate_total_outside_range_duration(
        history,
        3,
        6,
    ) == 0

    assert calculate_mean_outside_range_streak(
        history,
        3,
        6,
    ) == 0.0

    assert calculate_longest_outside_range_streak(
        history,
        3,
        6,
    ) == 0


def test_empty_history():
    assert find_outside_range_streaks(
        [],
        3,
        6,
    ) == []

    assert build_outside_range_streak_summary(
        [],
        3,
        6,
    ) == {
        "snapshot_count": 0,
        "lower_bound": 3,
        "upper_bound": 6,
        "streak_count": 0,
        "streaks": [],
        "total_outside_duration": 0,
        "mean_streak": 0.0,
        "longest_streak": 0,
    }


def test_outside_streak_at_start():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 4},
    ]

    assert find_outside_range_streaks(
        history,
        3,
        6,
    ) == [2]


def test_outside_streak_at_end():
    history = [
        {"alert_count": 4},
        {"alert_count": 7},
        {"alert_count": 8},
    ]

    assert find_outside_range_streaks(
        history,
        3,
        6,
    ) == [2]


def test_all_history_outside():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 8},
        {"alert_count": 9},
    ]

    assert find_outside_range_streaks(
        history,
        3,
        6,
    ) == [4]


def test_zero_width_range():
    history = [
        {"alert_count": 3},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 4},
    ]

    assert find_outside_range_streaks(
        history,
        3,
        3,
    ) == [1, 1]


def test_build_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 8},
        {"alert_count": 4},
        {"alert_count": 1},
    ]

    assert build_outside_range_streak_summary(
        history,
        3,
        6,
    ) == {
        "snapshot_count": 7,
        "lower_bound": 3,
        "upper_bound": 6,
        "streak_count": 3,
        "streaks": [2, 2, 1],
        "total_outside_duration": 5,
        "mean_streak": pytest.approx(5 / 3),
        "longest_streak": 2,
    }


def test_invalid_history_type():
    with pytest.raises(TypeError):
        find_outside_range_streaks(
            {},
            3,
            6,
        )


def test_invalid_history_item():
    with pytest.raises(TypeError):
        find_outside_range_streaks(
            [{"alert_count": 3}, "invalid"],
            3,
            6,
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        find_outside_range_streaks(
            [{"value": 3}],
            3,
            6,
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        find_outside_range_streaks(
            [{"alert_count": 3.5}],
            3,
            6,
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        find_outside_range_streaks(
            [{"alert_count": True}],
            3,
            6,
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        find_outside_range_streaks(
            [{"alert_count": -1}],
            3,
            6,
        )


def test_invalid_lower_bound_type():
    with pytest.raises(TypeError):
        find_outside_range_streaks(
            [{"alert_count": 4}],
            3.5,
            6,
        )


def test_invalid_upper_bound_type():
    with pytest.raises(TypeError):
        find_outside_range_streaks(
            [{"alert_count": 4}],
            3,
            6.5,
        )


def test_boolean_lower_bound():
    with pytest.raises(TypeError):
        find_outside_range_streaks(
            [{"alert_count": 4}],
            True,
            6,
        )


def test_boolean_upper_bound():
    with pytest.raises(TypeError):
        find_outside_range_streaks(
            [{"alert_count": 4}],
            3,
            False,
        )


def test_negative_lower_bound():
    with pytest.raises(ValueError):
        find_outside_range_streaks(
            [{"alert_count": 4}],
            -1,
            6,
        )


def test_negative_upper_bound():
    with pytest.raises(ValueError):
        find_outside_range_streaks(
            [{"alert_count": 4}],
            3,
            -1,
        )


def test_lower_bound_greater_than_upper_bound():
    with pytest.raises(ValueError):
        find_outside_range_streaks(
            [{"alert_count": 4}],
            6,
            3,
        )


def test_direct_invalid_alert_count():
    with pytest.raises(TypeError):
        is_outside_range(
            4.5,
            3,
            6,
        )


def test_direct_boolean_alert_count():
    with pytest.raises(TypeError):
        is_outside_range(
            True,
            3,
            6,
        )


def test_direct_negative_alert_count():
    with pytest.raises(ValueError):
        is_outside_range(
            -1,
            3,
            6,
        )
