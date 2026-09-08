from __future__ import annotations

import pytest

from src.evaluation.report_alert_threshold_summary import (
    build_alert_threshold_summary,
    calculate_average_alert_count,
    calculate_breach_percentage,
    calculate_breach_ratio,
    calculate_max_alert_count,
    count_breaches,
)


def test_count_breaches():
    history = [
        {"alert_count": 1},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 3},
    ]

    assert count_breaches(history, 3) == 2


def test_threshold_is_strictly_exceeded():
    history = [
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 3},
    ]

    assert count_breaches(history, 3) == 1


def test_calculate_breach_ratio():
    history = [
        {"alert_count": 1},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 3},
        {"alert_count": 5},
    ]

    assert calculate_breach_ratio(
        history,
        3,
    ) == pytest.approx(0.6)


def test_calculate_breach_percentage():
    history = [
        {"alert_count": 1},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 3},
        {"alert_count": 5},
    ]

    assert calculate_breach_percentage(
        history,
        3,
    ) == pytest.approx(60.0)


def test_calculate_max_alert_count():
    history = [
        {"alert_count": 2},
        {"alert_count": 7},
        {"alert_count": 4},
        {"alert_count": 5},
    ]

    assert calculate_max_alert_count(history) == 7


def test_calculate_average_alert_count():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 5},
    ]

    assert calculate_average_alert_count(
        history,
    ) == pytest.approx(3.0)


def test_no_breaches():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert count_breaches(history, 3) == 0
    assert calculate_breach_ratio(history, 3) == 0.0
    assert calculate_breach_percentage(history, 3) == 0.0


def test_every_observation_breaches():
    history = [
        {"alert_count": 5},
        {"alert_count": 6},
        {"alert_count": 7},
    ]

    assert count_breaches(history, 4) == 3
    assert calculate_breach_ratio(history, 4) == 1.0
    assert calculate_breach_percentage(history, 4) == 100.0


def test_empty_history():
    assert count_breaches([], 3) == 0
    assert calculate_breach_ratio([], 3) == 0.0
    assert calculate_breach_percentage([], 3) == 0.0
    assert calculate_max_alert_count([]) == 0
    assert calculate_average_alert_count([]) == 0.0

    assert build_alert_threshold_summary([], 3) == {
        "snapshot_count": 0,
        "threshold": 3,
        "breach_count": 0,
        "breach_ratio": 0.0,
        "breach_percentage": 0.0,
        "max_alert_count": 0,
        "average_alert_count": 0.0,
    }


def test_zero_threshold():
    history = [
        {"alert_count": 0},
        {"alert_count": 1},
        {"alert_count": 2},
    ]

    assert count_breaches(history, 0) == 2


def test_build_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 6},
        {"alert_count": 2},
        {"alert_count": 7},
    ]

    result = build_alert_threshold_summary(
        history,
        3,
    )

    assert result == {
        "snapshot_count": 5,
        "threshold": 3,
        "breach_count": 3,
        "breach_ratio": pytest.approx(0.6),
        "breach_percentage": pytest.approx(60.0),
        "max_alert_count": 7,
        "average_alert_count": pytest.approx(4.2),
    }


def test_invalid_history_type():
    with pytest.raises(TypeError):
        count_breaches({}, 3)


def test_invalid_history_item():
    with pytest.raises(TypeError):
        count_breaches(
            [{"alert_count": 5}, "invalid"],
            3,
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        count_breaches(
            [{"value": 5}],
            3,
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        count_breaches(
            [{"alert_count": 1.5}],
            3,
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        count_breaches(
            [{"alert_count": True}],
            3,
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        count_breaches(
            [{"alert_count": -1}],
            3,
        )


def test_invalid_threshold_type():
    history = [{"alert_count": 5}]

    with pytest.raises(TypeError):
        count_breaches(history, 3.5)


def test_boolean_threshold():
    history = [{"alert_count": 5}]

    with pytest.raises(TypeError):
        count_breaches(history, True)


def test_negative_threshold():
    history = [{"alert_count": 5}]

    with pytest.raises(ValueError):
        count_breaches(history, -1)
