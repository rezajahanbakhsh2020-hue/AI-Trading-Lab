import pytest

from src.evaluation.report_alert_threshold_range_quality_summary import (
    build_quality_summary,
    calculate_consistency_ratio,
    calculate_quality_percentage,
    calculate_quality_score,
    calculate_range_ratio,
    calculate_stability_ratio,
)


def test_calculate_range_ratio_for_values_inside_and_outside_range():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 5},
        {"alert_count": 8},
    ]

    assert calculate_range_ratio(history, 3, 5) == pytest.approx(0.5)


def test_calculate_stability_ratio_matches_range_ratio():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 7},
    ]

    assert calculate_stability_ratio(history, 3, 4) == pytest.approx(0.5)


def test_calculate_consistency_ratio_matches_range_ratio():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 5},
        {"alert_count": 6},
    ]

    assert calculate_consistency_ratio(history, 3, 5) == pytest.approx(0.5)


def test_quality_score_is_average_of_stability_and_consistency():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 8},
    ]

    assert calculate_quality_score(history, 3, 4) == pytest.approx(0.5)


def test_quality_percentage_converts_score_to_percent():
    history = [
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 8},
        {"alert_count": 9},
    ]

    assert calculate_quality_percentage(history, 3, 4) == pytest.approx(50.0)


def test_quality_summary_contains_expected_values():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 7},
    ]

    summary = build_quality_summary(history, 3, 4)

    assert summary == {
        "snapshot_count": 4,
        "lower_bound": 3,
        "upper_bound": 4,
        "range_ratio": pytest.approx(0.5),
        "stability_ratio": pytest.approx(0.5),
        "consistency_ratio": pytest.approx(0.5),
        "quality_score": pytest.approx(0.5),
        "quality_percentage": pytest.approx(50.0),
    }


def test_empty_history_returns_zero_ratios_and_score():
    history = []

    assert calculate_range_ratio(history, 2, 5) == 0.0
    assert calculate_stability_ratio(history, 2, 5) == 0.0
    assert calculate_consistency_ratio(history, 2, 5) == 0.0
    assert calculate_quality_score(history, 2, 5) == 0.0
    assert calculate_quality_percentage(history, 2, 5) == 0.0


def test_quality_summary_for_empty_history():
    summary = build_quality_summary([], 2, 5)

    assert summary == {
        "snapshot_count": 0,
        "lower_bound": 2,
        "upper_bound": 5,
        "range_ratio": 0.0,
        "stability_ratio": 0.0,
        "consistency_ratio": 0.0,
        "quality_score": 0.0,
        "quality_percentage": 0.0,
    }


def test_history_must_be_a_list():
    with pytest.raises(TypeError, match="history must be a list"):
        calculate_range_ratio((), 1, 3)


def test_history_items_must_be_mappings():
    with pytest.raises(
        TypeError,
        match="Each history item must be a mapping",
    ):
        calculate_range_ratio([1], 1, 3)


def test_alert_count_must_be_an_integer():
    with pytest.raises(
        ValueError,
        match="integer alert_count",
    ):
        calculate_range_ratio(
            [{"alert_count": 2.5}],
            1,
            3,
        )


def test_boolean_alert_count_is_rejected():
    with pytest.raises(
        ValueError,
        match="integer alert_count",
    ):
        calculate_range_ratio(
            [{"alert_count": True}],
            1,
            3,
        )


def test_negative_alert_count_is_rejected():
    with pytest.raises(
        ValueError,
        match="must not be negative",
    ):
        calculate_range_ratio(
            [{"alert_count": -1}],
            1,
            3,
        )


def test_bounds_must_be_integers():
    with pytest.raises(
        TypeError,
        match="lower_bound must be an integer",
    ):
        calculate_range_ratio(
            [{"alert_count": 2}],
            1.0,
            3,
        )


def test_boolean_bound_is_rejected():
    with pytest.raises(
        TypeError,
        match="upper_bound must be an integer",
    ):
        calculate_range_ratio(
            [{"alert_count": 2}],
            1,
            True,
        )


def test_negative_lower_bound_is_rejected():
    with pytest.raises(
        ValueError,
        match="lower_bound must not be negative",
    ):
        calculate_range_ratio(
            [{"alert_count": 2}],
            -1,
            3,
        )


def test_negative_upper_bound_is_rejected():
    with pytest.raises(
        ValueError,
        match="upper_bound must not be negative",
    ):
        calculate_range_ratio(
            [{"alert_count": 2}],
            1,
            -3,
        )


def test_lower_bound_cannot_exceed_upper_bound():
    with pytest.raises(
        ValueError,
        match="lower_bound must not be greater than upper_bound",
    ):
        calculate_range_ratio(
            [{"alert_count": 2}],
            5,
            3,
        )


def test_boundary_values_are_included():
    history = [
        {"alert_count": 3},
        {"alert_count": 5},
    ]

    assert calculate_range_ratio(history, 3, 5) == 1.0


def test_all_values_outside_range_produce_zero_quality():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 8},
    ]

    assert calculate_quality_score(history, 3, 5) == 0.0
    assert calculate_quality_percentage(history, 3, 5) == 0.0
