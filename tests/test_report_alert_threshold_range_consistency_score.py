from __future__ import annotations

import pytest

from src.evaluation.report_alert_threshold_range_consistency_score import (
    build_consistency_score_summary,
    calculate_consistency_percentage,
    calculate_consistency_score,
    calculate_deviation_percentage,
    calculate_deviation_score,
)


def test_calculate_consistency_score():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 9},
    ]

    assert calculate_consistency_score(
        history,
        3,
        6,
    ) == pytest.approx(3 / 5)


def test_calculate_deviation_score():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 9},
    ]

    assert calculate_deviation_score(
        history,
        3,
        6,
    ) == pytest.approx(2 / 5)


def test_consistency_percentage():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 9},
    ]

    assert calculate_consistency_percentage(
        history,
        3,
        6,
    ) == pytest.approx(60.0)


def test_deviation_percentage():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 9},
    ]

    assert calculate_deviation_percentage(
        history,
        3,
        6,
    ) == pytest.approx(40.0)


def test_all_values_consistent():
    history = [
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 6},
    ]

    assert calculate_consistency_score(
        history,
        3,
        6,
    ) == 1.0

    assert calculate_deviation_score(
        history,
        3,
        6,
    ) == 0.0

    assert calculate_consistency_percentage(
        history,
        3,
        6,
    ) == 100.0

    assert calculate_deviation_percentage(
        history,
        3,
        6,
    ) == 0.0


def test_all_values_deviate():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 7},
        {"alert_count": 8},
    ]

    assert calculate_consistency_score(
        history,
        3,
        6,
    ) == 0.0

    assert calculate_deviation_score(
        history,
        3,
        6,
    ) == 1.0

    assert calculate_consistency_percentage(
        history,
        3,
        6,
    ) == 0.0

    assert calculate_deviation_percentage(
        history,
        3,
        6,
    ) == 100.0


def test_empty_history():
    assert calculate_consistency_score(
        [],
        3,
        6,
    ) == 0.0

    assert calculate_deviation_score(
        [],
        3,
        6,
    ) == 1.0

    assert calculate_consistency_percentage(
        [],
        3,
        6,
    ) == 0.0

    assert calculate_deviation_percentage(
        [],
        3,
        6,
    ) == 100.0

    assert build_consistency_score_summary(
        [],
        3,
        6,
    ) == {
        "snapshot_count": 0,
        "lower_bound": 3,
        "upper_bound": 6,
        "consistency_score": 0.0,
        "deviation_score": 1.0,
        "consistency_percentage": 0.0,
        "deviation_percentage": 100.0,
    }


def test_zero_width_range():
    history = [
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 3},
    ]

    assert calculate_consistency_score(
        history,
        3,
        3,
    ) == pytest.approx(2 / 3)

    assert calculate_deviation_score(
        history,
        3,
        3,
    ) == pytest.approx(1 / 3)


def test_build_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 9},
    ]

    assert build_consistency_score_summary(
        history,
        3,
        6,
    ) == {
        "snapshot_count": 5,
        "lower_bound": 3,
        "upper_bound": 6,
        "consistency_score": pytest.approx(3 / 5),
        "deviation_score": pytest.approx(2 / 5),
        "consistency_percentage": pytest.approx(60.0),
        "deviation_percentage": pytest.approx(40.0),
    }


def test_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_consistency_score(
            {},
            3,
            6,
        )


def test_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_consistency_score(
            [{"alert_count": 3}, "invalid"],
            3,
            6,
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_consistency_score(
            [{"value": 3}],
            3,
            6,
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_consistency_score(
            [{"alert_count": 3.5}],
            3,
            6,
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_consistency_score(
            [{"alert_count": True}],
            3,
            6,
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_consistency_score(
            [{"alert_count": -1}],
            3,
            6,
        )


def test_invalid_lower_bound_type():
    with pytest.raises(TypeError):
        calculate_consistency_score(
            [{"alert_count": 4}],
            3.5,
            6,
        )


def test_invalid_upper_bound_type():
    with pytest.raises(TypeError):
        calculate_consistency_score(
            [{"alert_count": 4}],
            3,
            6.5,
        )


def test_boolean_lower_bound():
    with pytest.raises(TypeError):
        calculate_consistency_score(
            [{"alert_count": 4}],
            True,
            6,
        )


def test_boolean_upper_bound():
    with pytest.raises(TypeError):
        calculate_consistency_score(
            [{"alert_count": 4}],
            3,
            False,
        )


def test_negative_lower_bound():
    with pytest.raises(ValueError):
        calculate_consistency_score(
            [{"alert_count": 4}],
            -1,
            6,
        )


def test_negative_upper_bound():
    with pytest.raises(ValueError):
        calculate_consistency_score(
            [{"alert_count": 4}],
            3,
            -1,
        )


def test_lower_bound_greater_than_upper_bound():
    with pytest.raises(ValueError):
        calculate_consistency_score(
            [{"alert_count": 4}],
            6,
            3,
        )
