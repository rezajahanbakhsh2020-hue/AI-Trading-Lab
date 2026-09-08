from __future__ import annotations

import pytest

from src.evaluation.report_alert_threshold_range_stability import (
    build_range_stability_summary,
    calculate_range_instability,
    calculate_range_stability,
    calculate_stability_score,
    calculate_stable_snapshot_count,
    calculate_unstable_snapshot_count,
)


def test_calculate_range_stability():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 9},
    ]

    assert calculate_range_stability(
        history,
        3,
        6,
    ) == pytest.approx(3 / 5)


def test_calculate_range_instability():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 9},
    ]

    assert calculate_range_instability(
        history,
        3,
        6,
    ) == pytest.approx(2 / 5)


def test_stable_snapshot_count():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 9},
    ]

    assert calculate_stable_snapshot_count(
        history,
        3,
        6,
    ) == 3


def test_unstable_snapshot_count():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 9},
    ]

    assert calculate_unstable_snapshot_count(
        history,
        3,
        6,
    ) == 2


def test_stability_score():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 9},
    ]

    assert calculate_stability_score(
        history,
        3,
        6,
    ) == pytest.approx(0.6)


def test_all_snapshots_stable():
    history = [
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 6},
    ]

    assert calculate_range_stability(
        history,
        3,
        6,
    ) == 1.0

    assert calculate_range_instability(
        history,
        3,
        6,
    ) == 0.0

    assert calculate_stable_snapshot_count(
        history,
        3,
        6,
    ) == 3

    assert calculate_unstable_snapshot_count(
        history,
        3,
        6,
    ) == 0


def test_all_snapshots_unstable():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 7},
        {"alert_count": 8},
    ]

    assert calculate_range_stability(
        history,
        3,
        6,
    ) == 0.0

    assert calculate_range_instability(
        history,
        3,
        6,
    ) == 1.0

    assert calculate_stable_snapshot_count(
        history,
        3,
        6,
    ) == 0

    assert calculate_unstable_snapshot_count(
        history,
        3,
        6,
    ) == 4


def test_empty_history():
    assert calculate_range_stability(
        [],
        3,
        6,
    ) == 0.0

    assert calculate_range_instability(
        [],
        3,
        6,
    ) == 1.0

    assert calculate_stable_snapshot_count(
        [],
        3,
        6,
    ) == 0

    assert calculate_unstable_snapshot_count(
        [],
        3,
        6,
    ) == 0

    assert calculate_stability_score(
        [],
        3,
        6,
    ) == 0.0


def test_zero_width_range():
    history = [
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 3},
    ]

    assert calculate_range_stability(
        history,
        3,
        3,
    ) == pytest.approx(2 / 3)

    assert calculate_range_instability(
        history,
        3,
        3,
    ) == pytest.approx(1 / 3)


def test_build_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 9},
    ]

    assert build_range_stability_summary(
        history,
        3,
        6,
    ) == {
        "snapshot_count": 5,
        "lower_bound": 3,
        "upper_bound": 6,
        "stable_snapshot_count": 3,
        "unstable_snapshot_count": 2,
        "stability": pytest.approx(3 / 5),
        "instability": pytest.approx(2 / 5),
        "stability_score": pytest.approx(3 / 5),
    }


def test_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_range_stability(
            {},
            3,
            6,
        )


def test_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_range_stability(
            [{"alert_count": 3}, "invalid"],
            3,
            6,
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_range_stability(
            [{"value": 3}],
            3,
            6,
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_range_stability(
            [{"alert_count": 3.5}],
            3,
            6,
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_range_stability(
            [{"alert_count": True}],
            3,
            6,
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_range_stability(
            [{"alert_count": -1}],
            3,
            6,
        )


def test_invalid_lower_bound_type():
    with pytest.raises(TypeError):
        calculate_range_stability(
            [{"alert_count": 4}],
            3.5,
            6,
        )


def test_invalid_upper_bound_type():
    with pytest.raises(TypeError):
        calculate_range_stability(
            [{"alert_count": 4}],
            3,
            6.5,
        )


def test_boolean_lower_bound():
    with pytest.raises(TypeError):
        calculate_range_stability(
            [{"alert_count": 4}],
            True,
            6,
        )


def test_boolean_upper_bound():
    with pytest.raises(TypeError):
        calculate_range_stability(
            [{"alert_count": 4}],
            3,
            False,
        )


def test_negative_lower_bound():
    with pytest.raises(ValueError):
        calculate_range_stability(
            [{"alert_count": 4}],
            -1,
            6,
        )


def test_negative_upper_bound():
    with pytest.raises(ValueError):
        calculate_range_stability(
            [{"alert_count": 4}],
            3,
            -1,
        )


def test_lower_bound_greater_than_upper_bound():
    with pytest.raises(ValueError):
        calculate_range_stability(
            [{"alert_count": 4}],
            6,
            3,
        )
