import pytest

from src.evaluation.report_alert_threshold_range_stability_summary import (
    build_stability_summary,
    calculate_instability_ratio,
    calculate_stable_count,
    calculate_stability_ratio,
    calculate_unstable_count,
)


def test_stable_count():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    assert calculate_stable_count(
        history,
        1,
        3,
    ) == 3


def test_unstable_count():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    assert calculate_unstable_count(
        history,
        1,
        3,
    ) == 1


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


def test_instability_ratio():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    assert calculate_instability_ratio(
        history,
        1,
        3,
    ) == pytest.approx(0.25)


def test_build_stability_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    summary = build_stability_summary(
        history,
        1,
        3,
    )

    assert summary == {
        "snapshot_count": 4,
        "lower_bound": 1,
        "upper_bound": 3,
        "stable_count": 3,
        "unstable_count": 1,
        "stability_ratio": pytest.approx(0.75),
        "instability_ratio": pytest.approx(0.25),
        "stability_percentage": pytest.approx(75.0),
        "instability_percentage": pytest.approx(25.0),
    }


def test_empty_history():
    history = []

    assert calculate_stable_count(
        history,
        1,
        3,
    ) == 0

    assert calculate_unstable_count(
        history,
        1,
        3,
    ) == 0

    assert calculate_stability_ratio(
        history,
        1,
        3,
    ) == 0.0

    assert calculate_instability_ratio(
        history,
        1,
        3,
    ) == 0.0


def test_empty_history_summary():
    summary = build_stability_summary(
        [],
        1,
        3,
    )

    assert summary == {
        "snapshot_count": 0,
        "lower_bound": 1,
        "upper_bound": 3,
        "stable_count": 0,
        "unstable_count": 0,
        "stability_ratio": 0.0,
        "instability_ratio": 0.0,
        "stability_percentage": 0.0,
        "instability_percentage": 0.0,
    }


def test_boundary_values_are_stable():
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
    ]

    assert calculate_stable_count(
        history,
        2,
        5,
    ) == 2


def test_invalid_history_type_raises():
    with pytest.raises(TypeError):
        calculate_stable_count(
            None,
            1,
            3,
        )


def test_invalid_history_item_raises():
    with pytest.raises(TypeError):
        calculate_stable_count(
            [{"alert_count": 1}, "invalid"],
            1,
            3,
        )


def test_missing_alert_count_raises():
    with pytest.raises(ValueError):
        calculate_stable_count(
            [{"other": 1}],
            1,
            3,
        )


def test_negative_alert_count_raises():
    with pytest.raises(ValueError):
        calculate_stable_count(
            [{"alert_count": -1}],
            1,
            3,
        )


def test_boolean_alert_count_raises():
    with pytest.raises(ValueError):
        calculate_stable_count(
            [{"alert_count": True}],
            1,
            3,
        )


def test_invalid_bound_type_raises():
    with pytest.raises(TypeError):
        calculate_stable_count(
            [{"alert_count": 2}],
            1.0,
            3,
        )


def test_boolean_bound_raises():
    with pytest.raises(TypeError):
        calculate_stable_count(
            [{"alert_count": 2}],
            True,
            3,
        )


def test_negative_bound_raises():
    with pytest.raises(ValueError):
        calculate_stable_count(
            [{"alert_count": 2}],
            -1,
            3,
        )


def test_reversed_bounds_raise():
    with pytest.raises(ValueError):
        calculate_stable_count(
            [{"alert_count": 2}],
            5,
            2,
        )
