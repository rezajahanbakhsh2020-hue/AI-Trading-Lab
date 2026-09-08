from __future__ import annotations

import pytest

from src.evaluation.report_alert_threshold_breach import (
    build_alert_threshold_breach_summary,
    calculate_longest_breach_streak,
    calculate_threshold_breach_ratio,
    count_threshold_breaches,
    find_threshold_breach_positions,
)


def test_count_threshold_breaches():
    history = [
        {"alert_count": 1},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 3},
    ]

    assert count_threshold_breaches(history, 3) == 2


def test_find_threshold_breach_positions():
    history = [
        {"alert_count": 1},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 3},
        {"alert_count": 5},
    ]

    assert find_threshold_breach_positions(history, 3) == [1, 2, 4]


def test_threshold_is_strictly_exceeded():
    history = [
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 3},
    ]

    assert count_threshold_breaches(history, 3) == 1
    assert find_threshold_breach_positions(history, 3) == [1]


def test_calculate_threshold_breach_ratio():
    history = [
        {"alert_count": 1},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 3},
        {"alert_count": 5},
    ]

    assert calculate_threshold_breach_ratio(
        history,
        3,
    ) == pytest.approx(0.6)


def test_calculate_longest_breach_streak():
    history = [
        {"alert_count": 5},
        {"alert_count": 6},
        {"alert_count": 2},
        {"alert_count": 7},
        {"alert_count": 8},
        {"alert_count": 9},
        {"alert_count": 1},
    ]

    assert calculate_longest_breach_streak(
        history,
        4,
    ) == 3


def test_no_breaches():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert count_threshold_breaches(history, 3) == 0
    assert find_threshold_breach_positions(history, 3) == []
    assert calculate_threshold_breach_ratio(
        history,
        3,
    ) == 0.0
    assert calculate_longest_breach_streak(
        history,
        3,
    ) == 0


def test_every_observation_breaches():
    history = [
        {"alert_count": 5},
        {"alert_count": 6},
        {"alert_count": 7},
    ]

    assert count_threshold_breaches(history, 4) == 3
    assert find_threshold_breach_positions(
        history,
        4,
    ) == [0, 1, 2]
    assert calculate_threshold_breach_ratio(
        history,
        4,
    ) == 1.0
    assert calculate_longest_breach_streak(
        history,
        4,
    ) == 3


def test_zero_threshold():
    history = [
        {"alert_count": 0},
        {"alert_count": 1},
        {"alert_count": 2},
    ]

    assert count_threshold_breaches(history, 0) == 2
    assert find_threshold_breach_positions(
        history,
        0,
    ) == [1, 2]


def test_threshold_above_all_values():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert count_threshold_breaches(history, 10) == 0
    assert calculate_threshold_breach_ratio(
        history,
        10,
    ) == 0.0


def test_single_snapshot_breach():
    history = [
        {"alert_count": 5},
    ]

    assert count_threshold_breaches(history, 3) == 1
    assert find_threshold_breach_positions(
        history,
        3,
    ) == [0]
    assert calculate_threshold_breach_ratio(
        history,
        3,
    ) == 1.0
    assert calculate_longest_breach_streak(
        history,
        3,
    ) == 1


def test_build_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 6},
        {"alert_count": 2},
        {"alert_count": 7},
    ]

    result = build_alert_threshold_breach_summary(
        history,
        3,
    )

    assert result == {
        "snapshot_count": 5,
        "threshold": 3,
        "breach_count": 3,
        "breach_positions": [1, 2, 4],
        "breach_ratio": pytest.approx(0.6),
        "breach_percentage": pytest.approx(60.0),
        "longest_breach_streak": 2,
    }


def test_build_summary_without_breaches():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    result = build_alert_threshold_breach_summary(
        history,
        3,
    )

    assert result == {
        "snapshot_count": 3,
        "threshold": 3,
        "breach_count": 0,
        "breach_positions": [],
        "breach_ratio": 0.0,
        "breach_percentage": 0.0,
        "longest_breach_streak": 0,
    }


def test_empty_history():
    assert count_threshold_breaches([], 3) == 0
    assert find_threshold_breach_positions([], 3) == []
    assert calculate_longest_breach_streak([], 3) == 0

    with pytest.raises(ValueError):
        calculate_threshold_breach_ratio([], 3)

    with pytest.raises(ValueError):
        build_alert_threshold_breach_summary([], 3)


def test_invalid_history_type():
    with pytest.raises(TypeError):
        count_threshold_breaches({}, 3)


def test_invalid_history_item():
    with pytest.raises(TypeError):
        count_threshold_breaches(
            [{"alert_count": 5}, "invalid"],
            3,
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        count_threshold_breaches(
            [{"value": 5}],
            3,
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        count_threshold_breaches(
            [{"alert_count": 1.5}],
            3,
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        count_threshold_breaches(
            [{"alert_count": True}],
            3,
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        count_threshold_breaches(
            [{"alert_count": -1}],
            3,
        )


def test_invalid_threshold_type():
    history = [{"alert_count": 5}]

    with pytest.raises(TypeError):
        count_threshold_breaches(history, 3.5)


def test_boolean_threshold():
    history = [{"alert_count": 5}]

    with pytest.raises(TypeError):
        count_threshold_breaches(history, True)


def test_negative_threshold():
    history = [{"alert_count": 5}]

    with pytest.raises(ValueError):
        count_threshold_breaches(history, -1)
