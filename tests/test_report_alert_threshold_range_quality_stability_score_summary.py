import pytest

from src.evaluation.report_alert_threshold_range_quality_stability_score_summary import (
    build_quality_stability_score_summary,
    calculate_quality_ratio,
    calculate_quality_stability_score,
    calculate_score_percentage,
    calculate_stability_ratio,
    classify_score,
)


def test_quality_ratio():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 5},
        {"alert_count": 4},
    ]

    assert calculate_quality_ratio(history, 2, 4) == 0.75


def test_stability_ratio():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
        {"alert_count": 7},
    ]

    assert calculate_stability_ratio(history, 2, 4) == pytest.approx(2 / 3)


def test_quality_stability_score():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
        {"alert_count": 7},
    ]

    expected = (0.5 + (2 / 3)) / 2

    assert calculate_quality_stability_score(
        history,
        2,
        4,
    ) == pytest.approx(expected)


def test_score_percentage():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 4},
    ]

    assert calculate_score_percentage(
        history,
        2,
        4,
    ) == 100.0


def test_classify_stable_score():
    assert classify_score(0.80) == "stable"
    assert classify_score(0.95) == "stable"


def test_classify_acceptable_score():
    assert classify_score(0.60) == "acceptable"
    assert classify_score(0.79) == "acceptable"


def test_classify_unstable_score():
    assert classify_score(0.59) == "unstable"
    assert classify_score(0.0) == "unstable"


def test_classify_custom_thresholds():
    assert classify_score(
        0.70,
        stable_threshold=0.75,
        acceptable_threshold=0.65,
    ) == "acceptable"


def test_classify_score_rejects_non_numeric_score():
    with pytest.raises(TypeError, match="score must be numeric"):
        classify_score("0.80")


def test_stable_threshold_must_be_numeric():
    with pytest.raises(
        TypeError,
        match="stable_threshold must be numeric",
    ):
        classify_score(0.8, "0.8")


def test_acceptable_threshold_must_be_numeric():
    with pytest.raises(
        TypeError,
        match="acceptable_threshold must be numeric",
    ):
        classify_score(0.8, 0.8, "0.6")


def test_stable_threshold_must_be_between_zero_and_one():
    with pytest.raises(
        ValueError,
        match="stable_threshold must be between 0 and 1",
    ):
        classify_score(0.8, 1.1, 0.6)


def test_acceptable_threshold_must_be_between_zero_and_one():
    with pytest.raises(
        ValueError,
        match="acceptable_threshold must be between 0 and 1",
    ):
        classify_score(0.8, 0.9, -0.1)


def test_acceptable_threshold_cannot_exceed_stable_threshold():
    with pytest.raises(
        ValueError,
        match="acceptable_threshold must not exceed",
    ):
        classify_score(
            0.8,
            stable_threshold=0.60,
            acceptable_threshold=0.80,
        )


def test_empty_history_score_is_unstable():
    assert calculate_quality_stability_score([], 2, 4) == 0.0
    assert classify_score(
        calculate_quality_stability_score([], 2, 4)
    ) == "unstable"


def test_single_observation_inside_range_is_stable():
    history = [{"alert_count": 3}]

    assert calculate_quality_ratio(history, 2, 4) == 1.0
    assert calculate_stability_ratio(history, 2, 4) == 1.0
    assert calculate_quality_stability_score(
        history,
        2,
        4,
    ) == 1.0


def test_build_summary():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
        {"alert_count": 7},
    ]

    expected_score = (0.5 + (2 / 3)) / 2

    summary = build_quality_stability_score_summary(
        history,
        2,
        4,
    )

    assert summary["snapshot_count"] == 4
    assert summary["lower_bound"] == 2
    assert summary["upper_bound"] == 4
    assert summary["quality_ratio"] == pytest.approx(0.5)
    assert summary["stability_ratio"] == pytest.approx(2 / 3)
    assert summary["quality_stability_score"] == pytest.approx(
        expected_score
    )
    assert summary["quality_stability_percentage"] == pytest.approx(
        expected_score * 100.0
    )
    assert summary["classification"] == "unstable"


def test_history_must_be_a_list():
    with pytest.raises(TypeError, match="history must be a list"):
        calculate_quality_ratio((), 2, 4)


def test_history_item_must_be_mapping():
    with pytest.raises(
        TypeError,
        match="Each history item must be a mapping",
    ):
        calculate_quality_ratio([3], 2, 4)


def test_missing_alert_count_is_rejected():
    with pytest.raises(
        ValueError,
        match="integer alert_count",
    ):
        calculate_quality_ratio([{}], 2, 4)


def test_non_integer_alert_count_is_rejected():
    with pytest.raises(
        ValueError,
        match="integer alert_count",
    ):
        calculate_quality_ratio(
            [{"alert_count": 2.5}],
            2,
            4,
        )


def test_boolean_alert_count_is_rejected():
    with pytest.raises(
        ValueError,
        match="integer alert_count",
    ):
        calculate_quality_ratio(
            [{"alert_count": True}],
            2,
            4,
        )


def test_negative_alert_count_is_rejected():
    with pytest.raises(
        ValueError,
        match="alert_count must not be negative",
    ):
        calculate_quality_ratio(
            [{"alert_count": -1}],
            0,
            4,
        )


def test_lower_bound_must_be_integer():
    with pytest.raises(
        TypeError,
        match="lower_bound must be an integer",
    ):
        calculate_quality_ratio(
            [{"alert_count": 2}],
            2.0,
            4,
        )


def test_upper_bound_must_be_integer():
    with pytest.raises(
        TypeError,
        match="upper_bound must be an integer",
    ):
        calculate_quality_ratio(
            [{"alert_count": 2}],
            2,
            4.0,
        )


def test_negative_lower_bound_is_rejected():
    with pytest.raises(
        ValueError,
        match="lower_bound must not be negative",
    ):
        calculate_quality_ratio(
            [{"alert_count": 2}],
            -1,
            4,
        )


def test_negative_upper_bound_is_rejected():
    with pytest.raises(
        ValueError,
        match="upper_bound must not be negative",
    ):
        calculate_quality_ratio(
            [{"alert_count": 2}],
            0,
            -1,
        )


def test_lower_bound_cannot_exceed_upper_bound():
    with pytest.raises(
        ValueError,
        match="lower_bound must not be greater",
    ):
        calculate_quality_ratio(
            [{"alert_count": 2}],
            5,
            4,
        )


def test_range_boundaries_are_inclusive():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
    ]

    assert calculate_quality_ratio(history, 2, 4) == 1.0
