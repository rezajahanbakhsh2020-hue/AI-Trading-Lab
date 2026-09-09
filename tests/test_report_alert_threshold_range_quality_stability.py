import pytest

from src.evaluation.report_alert_threshold_range_quality_stability import (
    build_quality_stability_summary,
    calculate_quality_ratio,
    calculate_quality_stability_percentage,
    calculate_quality_stability_score,
    calculate_stability_ratio,
)


def test_quality_ratio_for_all_in_range_history():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 4},
    ]

    assert calculate_quality_ratio(history, 2, 4) == 1.0


def test_quality_ratio_for_partially_in_range_history():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 3},
    ]

    assert calculate_quality_ratio(history, 2, 4) == 0.5


def test_quality_ratio_for_empty_history():
    assert calculate_quality_ratio([], 2, 4) == 0.0


def test_stability_ratio_for_single_observation():
    history = [{"alert_count": 3}]

    assert calculate_stability_ratio(history, 2, 4) == 1.0


def test_stability_ratio_for_empty_history():
    assert calculate_stability_ratio([], 2, 4) == 0.0


def test_stability_ratio_when_all_states_are_stable():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 2},
    ]

    assert calculate_stability_ratio(history, 2, 4) == 1.0


def test_stability_ratio_counts_state_changes():
    history = [
        {"alert_count": 3},
        {"alert_count": 5},
        {"alert_count": 2},
        {"alert_count": 6},
    ]

    assert calculate_stability_ratio(history, 2, 4) == 0.0


def test_stability_ratio_for_mixed_transitions():
    history = [
        {"alert_count": 3},
        {"alert_count": 2},
        {"alert_count": 6},
        {"alert_count": 7},
    ]

    assert calculate_stability_ratio(history, 2, 4) == pytest.approx(1 / 3)


def test_quality_stability_score_combines_quality_and_stability():
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 3},
    ]

    quality = 2 / 3
    stability = 0.0

    assert calculate_quality_stability_score(
        history,
        2,
        4,
    ) == pytest.approx((quality + stability) / 2)


def test_quality_stability_percentage():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 4},
    ]

    assert calculate_quality_stability_percentage(
        history,
        2,
        4,
    ) == 100.0


def test_build_quality_stability_summary():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 5},
    ]

    summary = build_quality_stability_summary(
        history,
        2,
        4,
    )

    assert summary == {
        "snapshot_count": 3,
        "lower_bound": 2,
        "upper_bound": 4,
        "quality_ratio": pytest.approx(2 / 3),
        "stability_ratio": pytest.approx(0.5),
        "quality_stability_score": pytest.approx((2 / 3 + 0.5) / 2),
        "quality_stability_percentage": pytest.approx(
            ((2 / 3 + 0.5) / 2) * 100.0
        ),
    }


def test_history_must_be_a_list():
    with pytest.raises(TypeError, match="history must be a list"):
        calculate_quality_ratio((), 2, 4)


def test_history_items_must_be_mappings():
    with pytest.raises(
        TypeError,
        match="Each history item must be a mapping",
    ):
        calculate_quality_ratio([3], 2, 4)


def test_alert_count_must_be_integer():
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


def test_boundary_values_are_inclusive():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
    ]

    assert calculate_quality_ratio(history, 2, 4) == 1.0
