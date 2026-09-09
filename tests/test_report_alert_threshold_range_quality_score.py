import pytest

from src.evaluation.report_alert_threshold_range_quality_score import (
    build_quality_score_summary,
    calculate_boundary_penalty,
    calculate_in_range_score,
    calculate_quality_percentage,
    calculate_quality_score,
)


def test_in_range_score_counts_values_inside_inclusive_range():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 7},
    ]

    assert calculate_in_range_score(history, 3, 4) == pytest.approx(0.5)


def test_boundary_values_are_included():
    history = [
        {"alert_count": 3},
        {"alert_count": 4},
    ]

    assert calculate_in_range_score(history, 3, 4) == 1.0


def test_quality_score_matches_in_range_score():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 5},
        {"alert_count": 6},
    ]

    assert calculate_quality_score(history, 3, 5) == pytest.approx(0.5)


def test_quality_percentage_is_score_times_one_hundred():
    history = [
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 8},
        {"alert_count": 9},
    ]

    assert calculate_quality_percentage(
        history,
        3,
        4,
    ) == pytest.approx(50.0)


def test_boundary_penalty_is_complement_of_quality_score():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 8},
    ]

    assert calculate_boundary_penalty(
        history,
        3,
        4,
    ) == pytest.approx(0.5)


def test_empty_history_returns_zero_quality_and_full_penalty():
    assert calculate_quality_score([], 2, 5) == 0.0
    assert calculate_quality_percentage([], 2, 5) == 0.0
    assert calculate_boundary_penalty([], 2, 5) == 1.0


def test_summary_contains_expected_values():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 7},
    ]

    summary = build_quality_score_summary(
        history,
        3,
        4,
    )

    assert summary == {
        "snapshot_count": 4,
        "lower_bound": 3,
        "upper_bound": 4,
        "quality_score": pytest.approx(0.5),
        "quality_percentage": pytest.approx(50.0),
        "boundary_penalty": pytest.approx(0.5),
    }


def test_all_values_inside_range_produce_perfect_score():
    history = [
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 3},
        {"alert_count": 4},
    ]

    assert calculate_quality_score(
        history,
        3,
        4,
    ) == 1.0


def test_all_values_outside_range_produce_zero_score():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 8},
    ]

    assert calculate_quality_score(
        history,
        3,
        5,
    ) == 0.0


def test_history_must_be_a_list():
    with pytest.raises(TypeError, match="history must be a list"):
        calculate_quality_score((), 1, 3)


def test_history_items_must_be_mappings():
    with pytest.raises(
        TypeError,
        match="Each history item must be a mapping",
    ):
        calculate_quality_score([1], 1, 3)


def test_missing_alert_count_is_rejected():
    with pytest.raises(
        ValueError,
        match="integer alert_count",
    ):
        calculate_quality_score(
            [{}],
            1,
            3,
        )


def test_non_integer_alert_count_is_rejected():
    with pytest.raises(
        ValueError,
        match="integer alert_count",
    ):
        calculate_quality_score(
            [{"alert_count": 2.5}],
            1,
            3,
        )


def test_boolean_alert_count_is_rejected():
    with pytest.raises(
        ValueError,
        match="integer alert_count",
    ):
        calculate_quality_score(
            [{"alert_count": True}],
            1,
            3,
        )


def test_negative_alert_count_is_rejected():
    with pytest.raises(
        ValueError,
        match="must not be negative",
    ):
        calculate_quality_score(
            [{"alert_count": -1}],
            1,
            3,
        )


def test_non_integer_lower_bound_is_rejected():
    with pytest.raises(
        TypeError,
        match="lower_bound must be an integer",
    ):
        calculate_quality_score(
            [{"alert_count": 2}],
            1.0,
            3,
        )


def test_non_integer_upper_bound_is_rejected():
    with pytest.raises(
        TypeError,
        match="upper_bound must be an integer",
    ):
        calculate_quality_score(
            [{"alert_count": 2}],
            1,
            3.0,
        )


def test_boolean_bound_is_rejected():
    with pytest.raises(
        TypeError,
        match="upper_bound must be an integer",
    ):
        calculate_quality_score(
            [{"alert_count": 2}],
            1,
            True,
        )


def test_negative_lower_bound_is_rejected():
    with pytest.raises(
        ValueError,
        match="lower_bound must not be negative",
    ):
        calculate_quality_score(
            [{"alert_count": 2}],
            -1,
            3,
        )


def test_negative_upper_bound_is_rejected():
    with pytest.raises(
        ValueError,
        match="upper_bound must not be negative",
    ):
        calculate_quality_score(
            [{"alert_count": 2}],
            1,
            -3,
        )


def test_lower_bound_cannot_exceed_upper_bound():
    with pytest.raises(
        ValueError,
        match="lower_bound must not be greater than upper_bound",
    ):
        calculate_quality_score(
            [{"alert_count": 2}],
            5,
            3,
        )


def test_single_point_range_is_supported():
    history = [
        {"alert_count": 2},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert calculate_quality_score(
        history,
        2,
        2,
    ) == pytest.approx(2 / 3)
