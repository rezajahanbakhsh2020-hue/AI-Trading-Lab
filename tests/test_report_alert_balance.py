from __future__ import annotations

import pytest

from src.evaluation.report_alert_balance import (
    build_alert_balance_summary,
    calculate_alert_balance,
    calculate_alert_balance_range,
    calculate_alert_imbalance,
)


def test_calculate_alert_balance():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
    ]

    assert calculate_alert_balance(history) == pytest.approx(
        2 / 6
    )


def test_calculate_alert_balance_perfect():
    history = [
        {"alert_count": 3},
        {"alert_count": 3},
        {"alert_count": 3},
    ]

    assert calculate_alert_balance(history) == 1.0


def test_calculate_alert_balance_all_zero():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_balance(history) == 1.0


def test_calculate_alert_balance_empty():
    with pytest.raises(ValueError):
        calculate_alert_balance([])


def test_calculate_alert_imbalance():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
    ]

    assert calculate_alert_imbalance(history) == pytest.approx(
        1 - (2 / 6)
    )


def test_calculate_alert_imbalance_perfect():
    history = [
        {"alert_count": 5},
        {"alert_count": 5},
    ]

    assert calculate_alert_imbalance(history) == 0.0


def test_calculate_alert_balance_range():
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 3},
    ]

    assert calculate_alert_balance_range(history) == 4


def test_calculate_alert_balance_range_uniform():
    history = [
        {"alert_count": 4},
        {"alert_count": 4},
    ]

    assert calculate_alert_balance_range(history) == 0


def test_calculate_alert_balance_range_empty():
    with pytest.raises(ValueError):
        calculate_alert_balance_range([])


def test_build_alert_balance_summary():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
    ]

    result = build_alert_balance_summary(history)

    assert result["snapshot_count"] == 3
    assert result["alert_balance"] == pytest.approx(2 / 6)
    assert result["alert_imbalance"] == pytest.approx(
        1 - (2 / 6)
    )
    assert result["alert_balance_range"] == 4


def test_build_alert_balance_summary_perfect():
    history = [
        {"alert_count": 2},
        {"alert_count": 2},
        {"alert_count": 2},
    ]

    result = build_alert_balance_summary(history)

    assert result == {
        "snapshot_count": 3,
        "alert_balance": 1.0,
        "alert_imbalance": 0.0,
        "alert_balance_range": 0,
    }


def test_build_alert_balance_summary_all_zero():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    result = build_alert_balance_summary(history)

    assert result == {
        "snapshot_count": 2,
        "alert_balance": 1.0,
        "alert_imbalance": 0.0,
        "alert_balance_range": 0,
    }


def test_build_alert_balance_summary_single_snapshot():
    history = [
        {"alert_count": 7},
    ]

    result = build_alert_balance_summary(history)

    assert result == {
        "snapshot_count": 1,
        "alert_balance": 1.0,
        "alert_imbalance": 0.0,
        "alert_balance_range": 0,
    }


def test_build_alert_balance_summary_empty():
    with pytest.raises(ValueError):
        build_alert_balance_summary([])


def test_rejects_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_alert_balance({})


def test_rejects_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_alert_balance(
            [{"alert_count": 1}, "invalid"]
        )


def test_rejects_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_balance(
            [{"value": 1}]
        )


def test_rejects_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_balance(
            [{"alert_count": 1.5}]
        )


def test_rejects_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_balance(
            [{"alert_count": True}]
        )


def test_rejects_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_balance(
            [{"alert_count": -1}]
        )
