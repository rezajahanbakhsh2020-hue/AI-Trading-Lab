import pytest

from src.evaluation.report_alert_threshold_range_consistency_score_summary import (
    build_consistency_score_summary,
    calculate_consistency_percentage,
    calculate_consistency_score,
    calculate_deviation_percentage,
    calculate_deviation_score,
)


def test_consistency_score_within_range():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 5},
        {"alert_count": 7},
    ]

    score = calculate_consistency_score(
        history,
        2,
        5,
    )

    assert score == pytest.approx(0.75)


def test_deviation_score_within_range():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 5},
        {"alert_count": 7},
    ]

    score = calculate_deviation_score(
        history,
        2,
        5,
    )

    assert score == pytest.approx(0.25)


def test_consistency_percentage():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    percentage = calculate_consistency_percentage(
        history,
        1,
        3,
    )

    assert percentage == pytest.approx(75.0)


def test_deviation_percentage():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    percentage = calculate_deviation_percentage(
        history,
        1,
        3,
    )

    assert percentage == pytest.approx(25.0)


def test_build_consistency_score_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    summary = build_consistency_score_summary(
        history,
        1,
        3,
    )

    assert summary == {
        "snapshot_count": 4,
        "lower_bound": 1,
        "upper_bound": 3,
        "consistency_score": pytest.approx(0.75),
        "deviation_score": pytest.approx(0.25),
        "consistency_percentage": pytest.approx(75.0),
        "deviation_percentage": pytest.approx(25.0),
    }


def test_empty_history_returns_zero_scores():
    history = []

    assert calculate_consistency_score(
        history,
        2,
        5,
    ) == 0.0

    assert calculate_deviation_score(
        history,
        2,
        5,
    ) == 1.0

    summary = build_consistency_score_summary(
        history,
        2,
        5,
    )

    assert summary == {
        "snapshot_count": 0,
        "lower_bound": 2,
        "upper_bound": 5,
        "consistency_score": 0.0,
        "deviation_score": 1.0,
        "consistency_percentage": 0.0,
        "deviation_percentage": 100.0,
    }


def test_invalid_history_type_raises():
    with pytest.raises(TypeError):
        calculate_consistency_score(
            None,
            1,
            3,
        )


def test_invalid_history_item_raises():
    with pytest.raises(TypeError):
        calculate_consistency_score(
            [{"alert_count": 1}, "invalid"],
            1,
            3,
        )


def test_missing_alert_count_raises():
    with pytest.raises(ValueError):
        calculate_consistency_score(
            [{"other": 1}],
            1,
            3,
        )


def test_negative_alert_count_raises():
    with pytest.raises(ValueError):
        calculate_consistency_score(
            [{"alert_count": -1}],
            1,
            3,
        )


def test_boolean_alert_count_raises():
    with pytest.raises(ValueError):
        calculate_consistency_score(
            [{"alert_count": True}],
            1,
            3,
        )


def test_invalid_bound_type_raises():
    with pytest.raises(TypeError):
        calculate_consistency_score(
            [{"alert_count": 2}],
            1.0,
            3,
        )


def test_boolean_bound_raises():
    with pytest.raises(TypeError):
        calculate_consistency_score(
            [{"alert_count": 2}],
            True,
            3,
        )


def test_negative_bound_raises():
    with pytest.raises(ValueError):
        calculate_consistency_score(
            [{"alert_count": 2}],
            -1,
            3,
        )


def test_reversed_bounds_raise():
    with pytest.raises(ValueError):
        calculate_consistency_score(
            [{"alert_count": 2}],
            5,
            2,
        )


def test_boundary_values_are_included():
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
    ]

    assert calculate_consistency_score(
        history,
        2,
        5,
    ) == pytest.approx(1.0)
