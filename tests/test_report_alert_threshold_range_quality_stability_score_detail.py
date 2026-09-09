import pytest

from src.evaluation.report_alert_threshold_range_quality_stability_score_detail import (
    build_quality_stability_score_detail,
    calculate_quality_ratio,
    calculate_quality_stability_percentage,
    calculate_quality_stability_score,
    calculate_stability_ratio,
)


def test_quality_ratio_for_mixed_history():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 4},
    ]

    assert calculate_quality_ratio(history, 2, 4) == 0.75


def test_stability_ratio_for_mixed_transitions():
    history = [
        {"alert_count": 3},
        {"alert_count": 2},
        {"alert_count": 6},
        {"alert_count": 7},
    ]

    assert calculate_stability_ratio(history, 2, 4) == pytest.approx(2 / 3)


def test_quality_stability_score_is_average_of_two_ratios():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 4},
    ]

    expected = (0.75 + (2 / 3)) / 2

    assert calculate_quality_stability_score(
        history,
        2,
        4,
    ) == pytest.approx(expected)


def test_quality_stability_percentage():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 4},
    ]

    expected_score = (0.75 + (2 / 3)) / 2

    assert calculate_quality_stability_percentage(
        history,
        2,
        4,
    ) == pytest.approx(expected_score * 100)


def test_build_detail_returns_complete_result():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 4},
    ]

    result = build_quality_stability_score_detail(
        history,
        2,
        4,
    )

    expected_score = (0.75 + (2 / 3)) / 2

    assert result["snapshot_count"] == 4
    assert result["lower_bound"] == 2
    assert result["upper_bound"] == 4
    assert result["quality_ratio"] == pytest.approx(0.75)
    assert result["stability_ratio"] == pytest.approx(2 / 3)
    assert result["quality_stability_score"] == pytest.approx(
        expected_score
    )
    assert result["quality_stability_percentage"] == pytest.approx(
        expected_score * 100
    )


def test_empty_history():
    history = []

    assert calculate_quality_ratio(history, 2, 4) == 0.0
    assert calculate_stability_ratio(history, 2, 4) == 0.0
    assert calculate_quality_stability_score(history, 2, 4) == 0.0
    assert calculate_quality_stability_percentage(history, 2, 4) == 0.0


def test_single_observation_is_fully_stable():
    history = [{"alert_count": 3}]

    assert calculate_quality_ratio(history, 2, 4) == 1.0
    assert calculate_stability_ratio(history, 2, 4) == 1.0
    assert calculate_quality_stability_score(history, 2, 4) == 1.0


def test_range_bounds_are_inclusive():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
    ]

    assert calculate_quality_ratio(history, 2, 4) == 1.0


def test_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_quality_ratio((), 2, 4)


def test_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_quality_ratio([{"alert_count": 2}, 3], 2, 4)


def test_boolean_alert_count_is_rejected():
    with pytest.raises(TypeError):
        calculate_quality_ratio([{"alert_count": True}], 2, 4)


def test_negative_alert_count_is_rejected():
    with pytest.raises(ValueError):
        calculate_quality_ratio([{"alert_count": -1}], 2, 4)


def test_invalid_lower_bound_type():
    with pytest.raises(TypeError):
        calculate_quality_ratio([{"alert_count": 2}], "2", 4)


def test_invalid_upper_bound_type():
    with pytest.raises(TypeError):
        calculate_quality_ratio([{"alert_count": 2}], 2, "4")


def test_negative_bounds_are_rejected():
    with pytest.raises(ValueError):
        calculate_quality_ratio([{"alert_count": 2}], -1, 4)


def test_reversed_bounds_are_rejected():
    with pytest.raises(ValueError):
        calculate_quality_ratio([{"alert_count": 2}], 5, 2)
