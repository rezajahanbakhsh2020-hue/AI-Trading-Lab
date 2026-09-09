import pytest

from src.evaluation.report_alert_threshold_range_stability_score_summary import (
    build_stability_score_summary,
    calculate_instability_percentage,
    calculate_instability_score,
    calculate_stability_percentage,
    calculate_stability_score,
)


def test_stability_score_within_range():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 5},
        {"alert_count": 7},
    ]

    score = calculate_stability_score(
        history,
        2,
        5,
    )

    assert score == pytest.approx(0.75)


def test_instability_score_within_range():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 5},
        {"alert_count": 7},
    ]

    score = calculate_instability_score(
        history,
        2,
        5,
    )

    assert score == pytest.approx(0.25)


def test_stability_percentage():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    percentage = calculate_stability_percentage(
        history,
        1,
        3,
    )

    assert percentage == pytest.approx(75.0)


def test_instability_percentage():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    percentage = calculate_instability_percentage(
        history,
        1,
        3,
    )

    assert percentage == pytest.approx(25.0)


def test_build_stability_score_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    summary = build_stability_score_summary(
        history,
        1,
        3,
    )

    assert summary == {
        "snapshot_count": 4,
        "lower_bound": 1,
        "upper_bound": 3,
        "stability_score": pytest.approx(0.75),
        "instability_score": pytest.approx(0.25),
        "stability_percentage": pytest.approx(75.0),
        "instability_percentage": pytest.approx(25.0),
    }


def test_empty_history_returns_zero_stability():
    history = []

    assert calculate_stability_score(
        history,
        2,
        5,
    ) == 0.0

    assert calculate_instability_score(
        history,
        2,
        5,
    ) == 1.0

    summary = build_stability_score_summary(
        history,
        2,
        5,
    )

    assert summary == {
        "snapshot_count": 0,
        "lower_bound": 2,
        "upper_bound": 5,
        "stability_score": 0.0,
        "instability_score": 1.0,
        "stability_percentage": 0.0,
        "instability_percentage": 100.0,
    }


def test_invalid_history_type_raises():
    with pytest.raises(TypeError):
        calculate_stability_score(
            None,
            1,
            3,
        )


def test_invalid_history_item_raises():
    with pytest.raises(TypeError):
        calculate_stability_score(
            [{"alert_count": 1}, "invalid"],
            1,
            3,
        )


def test_missing_alert_count_raises():
    with pytest.raises(ValueError):
        calculate_stability_score(
            [{"other": 1}],
            1,
            3,
        )


def test_negative_alert_count_raises():
    with pytest.raises(ValueError):
        calculate_stability_score(
            [{"alert_count": -1}],
            1,
            3,
        )


def test_boolean_alert_count_raises():
    with pytest.raises(ValueError):
        calculate_stability_score(
            [{"alert_count": True}],
            1,
            3,
        )


def test_invalid_bound_type_raises():
    with pytest.raises(TypeError):
        calculate_stability_score(
            [{"alert_count": 2}],
            1.0,
            3,
        )


def test_boolean_bound_raises():
    with pytest.raises(TypeError):
        calculate_stability_score(
            [{"alert_count": 2}],
            True,
            3,
        )


def test_negative_bound_raises():
    with pytest.raises(ValueError):
        calculate_stability_score(
            [{"alert_count": 2}],
            -1,
            3,
        )


def test_reversed_bounds_raise():
    with pytest.raises(ValueError):
        calculate_stability_score(
            [{"alert_count": 2}],
            5,
            2,
        )


def test_boundary_values_are_included():
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
    ]

    assert calculate_stability_score(
        history,
        2,
        5,
    ) == pytest.approx(1.0)
