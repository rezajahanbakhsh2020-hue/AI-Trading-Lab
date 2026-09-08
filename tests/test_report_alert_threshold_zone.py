from __future__ import annotations

import pytest

from src.evaluation.report_alert_threshold_zone import (
    build_threshold_zone_summary,
    calculate_lower_zone,
    calculate_upper_zone,
    classify_alert_zone,
    count_alert_zones,
    find_zone_positions,
)


def test_calculate_lower_zone():
    assert calculate_lower_zone(4, 2) == 2
    assert calculate_lower_zone(4, 5) == 0


def test_calculate_upper_zone():
    assert calculate_upper_zone(4, 2) == 6
    assert calculate_upper_zone(4, 5) == 9


def test_classify_alert_zone_below():
    assert classify_alert_zone(1, 4, 2) == "below"


def test_classify_alert_zone_inside():
    assert classify_alert_zone(2, 4, 2) == "inside"
    assert classify_alert_zone(4, 4, 2) == "inside"
    assert classify_alert_zone(6, 4, 2) == "inside"


def test_classify_alert_zone_above():
    assert classify_alert_zone(7, 4, 2) == "above"


def test_zone_boundaries_are_inside():
    assert classify_alert_zone(2, 4, 2) == "inside"
    assert classify_alert_zone(6, 4, 2) == "inside"


def test_count_alert_zones():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 7},
    ]

    assert count_alert_zones(
        history,
        4,
        2,
    ) == {
        "below": 1,
        "inside": 3,
        "above": 1,
    }


def test_find_zone_positions():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 7},
    ]

    assert find_zone_positions(
        history,
        4,
        2,
    ) == {
        "below": [0],
        "inside": [1, 2, 3],
        "above": [4],
    }


def test_zero_width():
    history = [
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 5},
    ]

    assert count_alert_zones(
        history,
        4,
        0,
    ) == {
        "below": 1,
        "inside": 1,
        "above": 1,
    }


def test_empty_history():
    assert count_alert_zones([], 4, 2) == {
        "below": 0,
        "inside": 0,
        "above": 0,
    }

    assert find_zone_positions([], 4, 2) == {
        "below": [],
        "inside": [],
        "above": [],
    }

    assert build_threshold_zone_summary(
        [],
        4,
        2,
    ) == {
        "snapshot_count": 0,
        "threshold": 4,
        "width": 2,
        "lower_zone": 2,
        "upper_zone": 6,
        "zone_counts": {
            "below": 0,
            "inside": 0,
            "above": 0,
        },
        "zone_positions": {
            "below": [],
            "inside": [],
            "above": [],
        },
    }


def test_build_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 7},
    ]

    result = build_threshold_zone_summary(
        history,
        4,
        2,
    )

    assert result == {
        "snapshot_count": 5,
        "threshold": 4,
        "width": 2,
        "lower_zone": 2,
        "upper_zone": 6,
        "zone_counts": {
            "below": 1,
            "inside": 3,
            "above": 1,
        },
        "zone_positions": {
            "below": [0],
            "inside": [1, 2, 3],
            "above": [4],
        },
    }


def test_invalid_history_type():
    with pytest.raises(TypeError):
        count_alert_zones({}, 4, 2)


def test_invalid_history_item():
    with pytest.raises(TypeError):
        count_alert_zones(
            [{"alert_count": 4}, "invalid"],
            4,
            2,
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        count_alert_zones(
            [{"value": 4}],
            4,
            2,
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        count_alert_zones(
            [{"alert_count": 2.5}],
            4,
            2,
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        count_alert_zones(
            [{"alert_count": True}],
            4,
            2,
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        count_alert_zones(
            [{"alert_count": -1}],
            4,
            2,
        )


def test_invalid_threshold_type():
    with pytest.raises(TypeError):
        count_alert_zones(
            [{"alert_count": 4}],
            4.5,
            2,
        )


def test_boolean_threshold():
    with pytest.raises(TypeError):
        count_alert_zones(
            [{"alert_count": 4}],
            True,
            2,
        )


def test_negative_threshold():
    with pytest.raises(ValueError):
        count_alert_zones(
            [{"alert_count": 4}],
            -1,
            2,
        )


def test_invalid_width_type():
    with pytest.raises(TypeError):
        count_alert_zones(
            [{"alert_count": 4}],
            4,
            2.5,
        )


def test_boolean_width():
    with pytest.raises(TypeError):
        count_alert_zones(
            [{"alert_count": 4}],
            4,
            True,
        )


def test_negative_width():
    with pytest.raises(ValueError):
        count_alert_zones(
            [{"alert_count": 4}],
            4,
            -1,
        )


def test_invalid_alert_count_for_direct_calculation():
    with pytest.raises(TypeError):
        classify_alert_zone(4.5, 4, 2)


def test_boolean_alert_count_for_direct_calculation():
    with pytest.raises(TypeError):
        classify_alert_zone(True, 4, 2)


def test_negative_alert_count_for_direct_calculation():
    with pytest.raises(ValueError):
        classify_alert_zone(-1, 4, 2)
