import pytest

from src.evaluation.report_alert_threshold_range_quality_stability_score_trend import (
    build_quality_stability_score_trend,
    calculate_quality_ratio,
    calculate_quality_stability_score,
    calculate_stability_ratio,
)


def test_quality_ratio_for_history() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 7},
    ]

    result = calculate_quality_ratio(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result == pytest.approx(2 / 3)


def test_stability_ratio_for_history() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 7},
        {"alert_count": 8},
    ]

    result = calculate_stability_ratio(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result == pytest.approx(2 / 3)


def test_quality_stability_score_is_average_of_two_ratios() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 7},
        {"alert_count": 8},
    ]

    quality = 0.5
    stability = 2 / 3

    result = calculate_quality_stability_score(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result == pytest.approx((quality + stability) / 2)


def test_empty_history_returns_zero_quality_and_stability() -> None:
    history: list[dict[str, int]] = []

    assert calculate_quality_ratio(history, 2, 4) == 0.0
    assert calculate_stability_ratio(history, 2, 4) == 0.0
    assert calculate_quality_stability_score(history, 2, 4) == 0.0


def test_single_snapshot_has_full_stability() -> None:
    history = [{"alert_count": 3}]

    assert calculate_quality_ratio(history, 2, 4) == 1.0
    assert calculate_stability_ratio(history, 2, 4) == 1.0
    assert calculate_quality_stability_score(history, 2, 4) == 1.0


def test_trend_series_are_cumulative() -> None:
    history = [
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 4},
    ]

    result = build_quality_stability_score_trend(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result["quality_series"] == pytest.approx(
        [1.0, 0.5, 2 / 3]
    )
    assert result["stability_series"] == pytest.approx(
        [1.0, 0.0, 0.0]
    )
    assert result["score_series"] == pytest.approx(
        [1.0, 0.25, 1 / 3]
    )


def test_trend_direction_is_calculated_from_first_and_last_values() -> None:
    history = [
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 4},
    ]

    result = build_quality_stability_score_trend(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result["quality_trend"] == "down"
    assert result["stability_trend"] == "down"
    assert result["score_trend"] == "down"


def test_trend_change_values_are_returned() -> None:
    history = [
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 4},
    ]

    result = build_quality_stability_score_trend(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result["quality_change"] == pytest.approx(-1 / 3)
    assert result["stability_change"] == pytest.approx(-1.0)
    assert result["score_change"] == pytest.approx(-2 / 3)


def test_empty_history_returns_empty_series_and_flat_trends() -> None:
    result = build_quality_stability_score_trend(
        [],
        lower_bound=2,
        upper_bound=4,
    )

    assert result["snapshot_count"] == 0
    assert result["quality_series"] == []
    assert result["stability_series"] == []
    assert result["score_series"] == []
    assert result["quality_trend"] == "flat"
    assert result["stability_trend"] == "flat"
    assert result["score_trend"] == "flat"
    assert result["quality_change"] == 0.0
    assert result["stability_change"] == 0.0
    assert result["score_change"] == 0.0


def test_single_snapshot_has_flat_trends() -> None:
    result = build_quality_stability_score_trend(
        [{"alert_count": 3}],
        lower_bound=2,
        upper_bound=4,
    )

    assert result["quality_trend"] == "flat"
    assert result["stability_trend"] == "flat"
    assert result["score_trend"] == "flat"


def test_lower_bound_greater_than_upper_bound_is_rejected() -> None:
    with pytest.raises(ValueError, match="lower_bound"):
        calculate_quality_ratio(
            [{"alert_count": 3}],
            lower_bound=5,
            upper_bound=2,
        )


def test_negative_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        calculate_quality_ratio(
            [{"alert_count": 3}],
            lower_bound=-1,
            upper_bound=4,
        )


def test_boolean_alert_count_is_rejected() -> None:
    with pytest.raises(TypeError, match="alert_count"):
        calculate_quality_ratio(
            [{"alert_count": True}],
            lower_bound=2,
            upper_bound=4,
        )


def test_non_integer_alert_count_is_rejected() -> None:
    with pytest.raises(TypeError, match="alert_count"):
        calculate_quality_ratio(
            [{"alert_count": 3.5}],
            lower_bound=2,
            upper_bound=4,
        )


def test_non_mapping_history_item_is_rejected() -> None:
    with pytest.raises(TypeError, match="mapping"):
        calculate_quality_ratio(
            [{"alert_count": 3}, 4],  # type: ignore[list-item]
            lower_bound=2,
            upper_bound=4,
        )


def test_boolean_bounds_are_rejected() -> None:
    with pytest.raises(TypeError, match="lower_bound"):
        calculate_quality_ratio(
            [{"alert_count": 3}],
            lower_bound=True,  # type: ignore[arg-type]
            upper_bound=4,
        )


def test_complete_result_contains_expected_fields() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    result = build_quality_stability_score_trend(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    expected_keys = {
        "snapshot_count",
        "lower_bound",
        "upper_bound",
        "quality_series",
        "stability_series",
        "score_series",
        "quality_trend",
        "stability_trend",
        "score_trend",
        "quality_change",
        "stability_change",
        "score_change",
    }

    assert set(result) == expected_keys
    assert result["snapshot_count"] == 3
    assert result["lower_bound"] == 2
    assert result["upper_bound"] == 4
