from __future__ import annotations

import pytest

from src.evaluation.report_alert_threshold_band import (
    build_threshold_band_summary,
    calculate_band_ratio,
    calculate_lower_bound,
    calculate_upper_bound,
    count_within_band,
    find_outside_band_positions,
    find_within_band_positions,
    is_within_band,
)


def test_calculate_lower_bound():
    assert calculate_lower_bound(4, 2) == 2
    assert calculate_lower_bound(4, 5) == 0


def test_calculate_upper_bound():
    assert calculate_upper_bound(4, 2) == 6
    assert calculate_upper_bound(4, 5) == 9


def test_is_within_band():
    assert is_within_band(2, 4, 2) is True
    assert is_within_band(4, 4, 2) is True
    assert is_within_band(6, 4, 2) is True
    assert is_within_band(1, 4, 2) is False
    assert is_within_band(7, 4, 2) is False


def test_band_boundaries_are_inclusive():
    assert is_within_band(2, 4, 2) is True
    assert is_within_band(6, 4, 2) is True


def test_count_within_band():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 7},
    ]

    assert count_within_band(
        history,
        4,
        2,
    ) == 3


def test_calculate_band_ratio():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 7},
    ]

    assert calculate_band_ratio(
        history,
        4,
        2,
    ) == pytest.approx(0.6)


def test_find_within_band_positions():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 7},
    ]

    assert find_within_band_positions(
        history,
        4,
        2,
    ) == [1, 2, 3]


def test_find_outside_band_positions():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 7},
    ]

    assert find_outside_band_positions(
        history,
        4,
        2,
    ) == [0, 4]


def test_zero_band():
    history = [
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 5},
    ]

    assert count_within_band(
        history,
        4,
        0,
    ) == 1

    assert find_within_band_positions(
        history,
        4,
        0,
    ) == [1]


def test_empty_history():
    assert count_within_band([], 4, 2) == 0
    assert calculate_band_ratio([], 4, 2) == 0.0
    assert find_within_band_positions([], 4, 2) == []
    assert find_outside_band_positions([], 4, 2) == []

    assert build_threshold_band_summary(
        [],
        4,
        2,
    ) == {
        "snapshot_count": 0,
        "threshold": 4,
        "band": 2,
        "lower_bound": 2,
        "upper_bound": 6,
        "within_band_count": 0,
        "band_ratio": 0.0,
        "within_band_positions": [],
        "outside_band_positions": [],
    }


def test_build_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 7},
    ]

    result = build_threshold_band_summary(
        history,
        4,
        2,
    )

    assert result == {
        "snapshot_count": 5,
        "threshold": 4,
        "band": 2,
        "lower_bound": 2,
        "upper_bound": 6,
        "within_band_count": 3,
        "band_ratio": pytest.approx(0.6),
        "within_band_positions": [1, 2, 3],
        "outside_band_positions": [0, 4],
    }


def test_invalid_history_type():
    with pytest.raises(TypeError):
        count_within_band({}, 4, 2)


def test_invalid_history_item():
    with pytest.raises(TypeError):
        count_within_band(
            [{"alert_count": 4}, "invalid"],
            4,
            2,
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        count_within_band(
            [{"value": 4}],
            4,
            2,
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        count_within_band(
            [{"alert_count": 2.5}],
            4,
            2,
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        count_within_band(
            [{"alert_count": True}],
            4,
            2,
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        count_within_band(
            [{"alert_count": -1}],
            4,
            2,
        )


def test_invalid_threshold_type():
    with pytest.raises(TypeError):
        count_within_band(
            [{"alert_count": 4}],
            4.5,
            2,
        )


def test_boolean_threshold():
    with pytest.raises(TypeError):
        count_within_band(
            [{"alert_count": 4}],
            True,
            2,
        )


def test_negative_threshold():
    with pytest.raises(ValueError):
        count_within_band(
            [{"alert_count": 4}],
            -1,
            2,
        )


def test_invalid_band_type():
    with pytest.raises(TypeError):
        count_within_band(
            [{"alert_count": 4}],
            4,
            2.5,
        )


def test_boolean_band():
    with pytest.raises(TypeError):
        count_within_band(
            [{"alert_count": 4}],
            4,
            True,
        )


def test_negative_band():
    with pytest.raises(ValueError):
        count_within_band(
            [{"alert_count": 4}],
            4,
            -1,
        )


def test_invalid_alert_count_for_direct_calculation():
    with pytest.raises(TypeError):
        is_within_band(4.5, 4, 2)


def test_boolean_alert_count_for_direct_calculation():
    with pytest.raises(TypeError):
        is_within_band(True, 4, 2)


def test_negative_alert_count_for_direct_calculation():
    with pytest.raises(ValueError):
        is_within_band(-1, 4, 2)
