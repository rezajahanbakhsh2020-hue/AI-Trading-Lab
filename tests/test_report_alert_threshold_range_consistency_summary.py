import pytest

from src.evaluation.report_alert_threshold_range_consistency_summary import (
    build_consistency_summary,
    calculate_consistency_ratio,
    calculate_consistent_count,
    calculate_inconsistency_ratio,
    calculate_inconsistent_count,
)


def test_consistent_count():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    assert calculate_consistent_count(
        history,
        1,
        3,
    ) == 3


def test_inconsistent_count():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    assert calculate_inconsistent_count(
        history,
        1,
        3,
    ) == 1


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


def test_inconsistency_ratio():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    assert calculate_inconsistency_ratio(
        history,
        1,
        3,
    ) == pytest.approx(0.25)


def test_build_consistency_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    summary = build_consistency_summary(
        history,
        1,
        3,
    )

    assert summary == {
        "snapshot_count": 4,
        "lower_bound": 1,
        "upper_bound": 3,
        "consistent_count": 3,
        "inconsistent_count": 1,
        "consistency_ratio": pytest.approx(0.75),
        "inconsistency_ratio": pytest.approx(0.25),
        "consistency_percentage": pytest.approx(75.0),
        "inconsistency_percentage": pytest.approx(25.0),
    }


def test_empty_history():
    history = []

    assert calculate_consistent_count(
        history,
        1,
        3,
    ) == 0

    assert calculate_inconsistent_count(
        history,
        1,
        3,
    ) == 0

    assert calculate_consistency_ratio(
        history,
        1,
        3,
    ) == 0.0

    assert calculate_inconsistency_ratio(
        history,
        1,
        3,
    ) == 0.0


def test_empty_history_summary():
    summary = build_consistency_summary(
        [],
        1,
        3,
    )

    assert summary == {
        "snapshot_count": 0,
        "lower_bound": 1,
        "upper_bound": 3,
        "consistent_count": 0,
        "inconsistent_count": 0,
        "consistency_ratio": 0.0,
        "inconsistency_ratio": 0.0,
        "consistency_percentage": 0.0,
        "inconsistency_percentage": 0.0,
    }


def test_boundary_values_are_consistent():
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
    ]

    assert calculate_consistent_count(
        history,
        2,
        5,
    ) == 2


def test_invalid_history_type_raises():
    with pytest.raises(TypeError):
        calculate_consistent_count(
            None,
            1,
            3,
        )


def test_invalid_history_item_raises():
    with pytest.raises(TypeError):
        calculate_consistent_count(
            [{"alert_count": 1}, "invalid"],
            1,
            3,
        )


def test_missing_alert_count_raises():
    with pytest.raises(ValueError):
        calculate_consistent_count(
            [{"other": 1}],
            1,
            3,
        )


def test_negative_alert_count_raises():
    with pytest.raises(ValueError):
        calculate_consistent_count(
            [{"alert_count": -1}],
            1,
            3,
        )


def test_boolean_alert_count_raises():
    with pytest.raises(ValueError):
        calculate_consistent_count(
            [{"alert_count": True}],
            1,
            3,
        )


def test_invalid_bound_type_raises():
    with pytest.raises(TypeError):
        calculate_consistent_count(
            [{"alert_count": 2}],
            1.0,
            3,
        )


def test_boolean_bound_raises():
    with pytest.raises(TypeError):
        calculate_consistent_count(
            [{"alert_count": 2}],
            True,
            3,
        )


def test_negative_bound_raises():
    with pytest.raises(ValueError):
        calculate_consistent_count(
            [{"alert_count": 2}],
            -1,
            3,
        )


def test_reversed_bounds_raise():
    with pytest.raises(ValueError):
        calculate_consistent_count(
            [{"alert_count": 2}],
            5,
            2,
        )
