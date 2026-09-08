from __future__ import annotations

import pytest

from src.evaluation.report_alert_balance_ratio import (
    build_alert_balance_ratio_summary,
    calculate_alert_balance_gap,
    calculate_alert_balance_ratio,
)


def test_calculate_alert_balance_ratio():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
    ]

    expected = 2 / 4

    assert calculate_alert_balance_ratio(history) == pytest.approx(
        expected
    )


def test_calculate_alert_balance_ratio_perfect():
    history = [
        {"alert_count": 3},
        {"alert_count": 3},
        {"alert_count": 3},
    ]

    assert calculate_alert_balance_ratio(history) == 1.0


def test_calculate_alert_balance_ratio_all_zero():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_balance_ratio(history) == 1.0


def test_calculate_alert_balance_ratio_single_snapshot():
    history = [
        {"alert_count": 5},
    ]

    assert calculate_alert_balance_ratio(history) == 1.0


def test_calculate_alert_balance_ratio_empty():
    with pytest.raises(ValueError):
        calculate_alert_balance_ratio([])


def test_calculate_alert_balance_gap():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
    ]

    expected = 1 - (2 / 4)

    assert calculate_alert_balance_gap(history) == pytest.approx(
        expected
    )


def test_calculate_alert_balance_gap_perfect():
    history = [
        {"alert_count": 4},
        {"alert_count": 4},
    ]

    assert calculate_alert_balance_gap(history) == 0.0


def test_calculate_alert_balance_gap_all_zero():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_balance_gap(history) == 0.0


def test_build_alert_balance_ratio_summary():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
    ]

    result = build_alert_balance_ratio_summary(history)

    assert result["snapshot_count"] == 3
    assert result["alert_balance_ratio"] == pytest.approx(
        2 / 4
    )
    assert result["alert_balance_gap"] == pytest.approx(
        1 - (2 / 4)
    )


def test_build_alert_balance_ratio_summary_perfect():
    history = [
        {"alert_count": 3},
        {"alert_count": 3},
    ]

    result = build_alert_balance_ratio_summary(history)

    assert result == {
        "snapshot_count": 2,
        "alert_balance_ratio": 1.0,
        "alert_balance_gap": 0.0,
    }


def test_build_alert_balance_ratio_summary_empty():
    with pytest.raises(ValueError):
        build_alert_balance_ratio_summary([])


def test_rejects_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_alert_balance_ratio({})


def test_rejects_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_alert_balance_ratio(
            [{"alert_count": 1}, "invalid"]
        )


def test_rejects_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_balance_ratio(
            [{"value": 1}]
        )


def test_rejects_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_balance_ratio(
            [{"alert_count": 1.5}]
        )


def test_rejects_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_balance_ratio(
            [{"alert_count": True}]
        )


def test_rejects_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_balance_ratio(
            [{"alert_count": -1}]
        )
