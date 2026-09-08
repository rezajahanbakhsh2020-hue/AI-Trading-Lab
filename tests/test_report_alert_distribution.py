from __future__ import annotations

import pytest

from src.evaluation.report_alert_distribution import (
    build_alert_distribution_summary,
    calculate_alert_count_share,
    calculate_alert_distribution,
    find_most_common_alert_count,
)


def test_calculate_alert_distribution():
    history = [
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 2},
    ]

    result = calculate_alert_distribution(history)

    assert result == {
        0: 2,
        1: 1,
        2: 3,
    }


def test_calculate_alert_distribution_empty():
    assert calculate_alert_distribution([]) == {}


def test_calculate_alert_distribution_rejects_invalid_history():
    with pytest.raises(TypeError):
        calculate_alert_distribution({})


def test_calculate_alert_distribution_rejects_invalid_item():
    with pytest.raises(TypeError):
        calculate_alert_distribution(
            [{"alert_count": 1}, "invalid"]
        )


def test_calculate_alert_distribution_rejects_missing_count():
    with pytest.raises(ValueError):
        calculate_alert_distribution(
            [{"value": 1}]
        )


def test_calculate_alert_distribution_rejects_non_numeric_count():
    with pytest.raises(ValueError):
        calculate_alert_distribution(
            [{"alert_count": "invalid"}]
        )


def test_calculate_alert_distribution_rejects_negative_count():
    with pytest.raises(ValueError):
        calculate_alert_distribution(
            [{"alert_count": -1}]
        )


def test_calculate_alert_distribution_rejects_fractional_count():
    with pytest.raises(ValueError):
        calculate_alert_distribution(
            [{"alert_count": 1.5}]
        )


def test_find_most_common_alert_count():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 2},
    ]

    assert find_most_common_alert_count(history) == 2


def test_find_most_common_alert_count_tie_returns_smallest():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 1},
        {"alert_count": 2},
    ]

    assert find_most_common_alert_count(history) == 1


def test_find_most_common_alert_count_empty():
    with pytest.raises(ValueError):
        find_most_common_alert_count([])


def test_calculate_alert_count_share():
    history = [
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 2},
        {"alert_count": 1},
    ]

    assert calculate_alert_count_share(
        history,
        2,
    ) == 0.5


def test_calculate_alert_count_share_missing_count():
    history = [
        {"alert_count": 0},
        {"alert_count": 1},
    ]

    assert calculate_alert_count_share(
        history,
        5,
    ) == 0.0


def test_calculate_alert_count_share_empty():
    with pytest.raises(ValueError):
        calculate_alert_count_share([], 0)


def test_calculate_alert_count_share_rejects_invalid_count():
    with pytest.raises(TypeError):
        calculate_alert_count_share(
            [{"alert_count": 1}],
            1.5,
        )


def test_calculate_alert_count_share_rejects_negative_count():
    with pytest.raises(ValueError):
        calculate_alert_count_share(
            [{"alert_count": 1}],
            -1,
        )


def test_build_alert_distribution_summary():
    history = [
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 2},
        {"alert_count": 1},
        {"alert_count": 2},
    ]

    result = build_alert_distribution_summary(history)

    assert result == {
        "distribution": {
            0: 1,
            1: 1,
            2: 3,
        },
        "most_common_alert_count": 2,
        "most_common_share": 0.6,
    }


def test_build_alert_distribution_summary_empty():
    with pytest.raises(ValueError):
        build_alert_distribution_summary([])
