import pytest

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score import (
    build_quality_score_summary,
    calculate_consistency_ratio,
    calculate_quality_percentage,
    calculate_quality_score,
    calculate_stability_consistency_score,
    calculate_stability_ratio,
)


def test_stability_ratio():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    assert calculate_stability_ratio(
        history,
        1,
        3,
    ) == pytest.approx(0.75)


def test_consistency_ratio():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    assert calculate_consistency_ratio(
        history,
        1,
        3,
    ) == pytest.approx(0.75)


def test_stability_consistency_score():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    assert calculate_stability_consistency_score(
        history,
        1,
        3,
    ) == pytest.approx(0.75)


def test_quality_score():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    assert calculate_quality_score(
        history,
        1,
        3,
    ) == pytest.approx(0.75)


def test_quality_percentage():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    assert calculate_quality_percentage(
        history,
        1,
        3,
    ) == pytest.approx(75.0)


def test_build_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    summary = build_quality_score_summary(
        history,
        1,
        3,
    )

    assert summary == {
        "snapshot_count": 4,
        "lower_bound": 1,
        "upper_bound": 3,
        "stability_ratio": pytest.approx(0.75),
        "consistency_ratio": pytest.approx(0.75),
        "quality_score": pytest.approx(0.75),
        "quality_percentage": pytest.approx(75.0),
    }


def test_empty_history():
    assert calculate_quality_score(
        [],
        1,
        3,
    ) == 0.0

    assert calculate_quality_percentage(
        [],
        1,
        3,
    ) == 0.0


def test_empty_history_summary():
    summary = build_quality_score_summary(
        [],
        1,
        3,
    )

    assert summary == {
        "snapshot_count": 0,
        "lower_bound": 1,
        "upper_bound": 3,
        "stability_ratio": 0.0,
        "consistency_ratio": 0.0,
        "quality_score": 0.0,
        "quality_percentage": 0.0,
    }


def test_boundary_values_are_included():
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
    ]

    assert calculate_quality_score(
        history,
        2,
        5,
    ) == pytest.approx(1.0)


def test_no_values_inside_range():
    history = [
        {"alert_count": 1},
        {"alert_count": 6},
    ]

    assert calculate_quality_score(
        history,
        2,
        5,
    ) == pytest.approx(0.0)


def test_invalid_history_type_raises():
    with pytest.raises(TypeError):
        calculate_quality_score(
            None,
            1,
            3,
        )


def test_invalid_history_item_raises():
    with pytest.raises(TypeError):
        calculate_quality_score(
            [{"alert_count": 1}, "invalid"],
            1,
            3,
        )


def test_missing_alert_count_raises():
    with pytest.raises(ValueError):
        calculate_quality_score(
            [{"other": 1}],
            1,
            3,
        )


def test_negative_alert_count_raises():
    with pytest.raises(ValueError):
        calculate_quality_score(
            [{"alert_count": -1}],
            1,
            3,
        )


def test_boolean_alert_count_raises():
    with pytest.raises(ValueError):
        calculate_quality_score(
            [{"alert_count": True}],
            1,
            3,
        )


def test_invalid_bound_type_raises():
    with pytest.raises(TypeError):
        calculate_quality_score(
            [{"alert_count": 2}],
            1.0,
            3,
        )


def test_boolean_bound_raises():
    with pytest.raises(TypeError):
        calculate_quality_score(
            [{"alert_count": 2}],
            True,
            3,
        )


def test_negative_bound_raises():
    with pytest.raises(ValueError):
        calculate_quality_score(
            [{"alert_count": 2}],
            -1,
            3,
        )


def test_reversed_bounds_raise():
    with pytest.raises(ValueError):
        calculate_quality_score(
            [{"alert_count": 2}],
            5,
            2,
        )
