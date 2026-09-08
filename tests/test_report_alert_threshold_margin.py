from __future__ import annotations

import pytest

from src.evaluation.report_alert_threshold_margin import (
    build_threshold_margin_summary,
    calculate_margin,
    count_above_margin,
    count_below_margin,
    find_above_margin_positions,
    find_below_margin_positions,
    is_above_margin,
    is_below_margin,
)


def test_calculate_margin_above_threshold():
    assert calculate_margin(7, 4) == 3


def test_calculate_margin_below_threshold():
    assert calculate_margin(2, 4) == -2


def test_calculate_margin_at_threshold():
    assert calculate_margin(4, 4) == 0


def test_is_above_margin():
    assert is_above_margin(7, 4, 3) is True
    assert is_above_margin(6, 4, 3) is False


def test_is_below_margin():
    assert is_below_margin(1, 4, 3) is True
    assert is_below_margin(2, 4, 3) is False


def test_zero_margin():
    assert is_above_margin(4, 4, 0) is True
    assert is_below_margin(4, 4, 0) is True


def test_count_above_margin():
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 7},
        {"alert_count": 8},
    ]

    assert count_above_margin(
        history,
        4,
        3,
    ) == 2


def test_count_below_margin():
    history = [
        {"alert_count": 0},
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 5},
    ]

    assert count_below_margin(
        history,
        4,
        3,
    ) == 2


def test_find_above_margin_positions():
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 7},
        {"alert_count": 8},
    ]

    assert find_above_margin_positions(
        history,
        4,
        3,
    ) == [2, 3]


def test_find_below_margin_positions():
    history = [
        {"alert_count": 0},
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 5},
    ]

    assert find_below_margin_positions(
        history,
        4,
        3,
    ) == [0, 1]


def test_build_summary():
    history = [
        {"alert_count": 0},
        {"alert_count": 1},
        {"alert_count": 4},
        {"alert_count": 7},
        {"alert_count": 8},
    ]

    result = build_threshold_margin_summary(
        history,
        4,
        3,
    )

    assert result == {
        "snapshot_count": 5,
        "threshold": 4,
        "margin": 3,
        "above_margin_count": 2,
        "below_margin_count": 2,
        "above_margin_positions": [3, 4],
        "below_margin_positions": [0, 1],
    }


def test_empty_history():
    assert count_above_margin([], 4, 3) == 0
    assert count_below_margin([], 4, 3) == 0
    assert find_above_margin_positions([], 4, 3) == []
    assert find_below_margin_positions([], 4, 3) == []

    assert build_threshold_margin_summary(
        [],
        4,
        3,
    ) == {
        "snapshot_count": 0,
        "threshold": 4,
        "margin": 3,
        "above_margin_count": 0,
        "below_margin_count": 0,
        "above_margin_positions": [],
        "below_margin_positions": [],
    }


def test_invalid_history_type():
    with pytest.raises(TypeError):
        count_above_margin({}, 4, 3)


def test_invalid_history_item():
    with pytest.raises(TypeError):
        count_above_margin(
            [{"alert_count": 5}, "invalid"],
            4,
            3,
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        count_above_margin(
            [{"value": 5}],
            4,
            3,
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        count_above_margin(
            [{"alert_count": 2.5}],
            4,
            3,
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        count_above_margin(
            [{"alert_count": True}],
            4,
            3,
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        count_above_margin(
            [{"alert_count": -1}],
            4,
            3,
        )


def test_invalid_threshold_type():
    with pytest.raises(TypeError):
        count_above_margin(
            [{"alert_count": 5}],
            4.5,
            3,
        )


def test_boolean_threshold():
    with pytest.raises(TypeError):
        count_above_margin(
            [{"alert_count": 5}],
            True,
            3,
        )


def test_negative_threshold():
    with pytest.raises(ValueError):
        count_above_margin(
            [{"alert_count": 5}],
            -1,
            3,
        )


def test_invalid_margin_type():
    with pytest.raises(TypeError):
        count_above_margin(
            [{"alert_count": 5}],
            4,
            3.5,
        )


def test_boolean_margin():
    with pytest.raises(TypeError):
        count_above_margin(
            [{"alert_count": 5}],
            4,
            True,
        )


def test_negative_margin():
    with pytest.raises(ValueError):
        count_above_margin(
            [{"alert_count": 5}],
            4,
            -1,
        )


def test_invalid_alert_count_for_direct_calculation():
    with pytest.raises(TypeError):
        calculate_margin(4.5, 4)


def test_boolean_alert_count_for_direct_calculation():
    with pytest.raises(TypeError):
        calculate_margin(True, 4)


def test_negative_alert_count_for_direct_calculation():
    with pytest.raises(ValueError):
        calculate_margin(-1, 4)
