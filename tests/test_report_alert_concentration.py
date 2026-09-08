from __future__ import annotations

import pytest

from src.evaluation.report_alert_concentration import (
    build_alert_concentration_summary,
    calculate_alert_concentration,
    calculate_alert_concentration_gap,
    calculate_peak_alert_share,
)


def test_calculate_alert_concentration():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    assert calculate_alert_concentration(history) == 3 / 6


def test_calculate_alert_concentration_all_alerts_in_one_snapshot():
    history = [
        {"alert_count": 6},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_concentration(history) == 1.0


def test_calculate_alert_concentration_no_alerts():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_concentration(history) == 0.0


def test_calculate_alert_concentration_empty():
    with pytest.raises(ValueError):
        calculate_alert_concentration([])


def test_calculate_peak_alert_share():
    history = [
        {"alert_count": 2},
        {"alert_count": 1},
        {"alert_count": 3},
    ]

    assert calculate_peak_alert_share(history) == 3 / 6


def test_calculate_alert_concentration_gap():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    assert calculate_alert_concentration_gap(history) == (
        3 / 6 - 1 / 5
    )


def test_calculate_alert_concentration_gap_uniform_history():
    history = [
        {"alert_count": 2},
        {"alert_count": 2},
    ]

    assert calculate_alert_concentration_gap(history) == 0.0


def test_calculate_alert_concentration_gap_empty():
    with pytest.raises(ValueError):
        calculate_alert_concentration_gap([])


def test_build_alert_concentration_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    result = build_alert_concentration_summary(history)

    assert result == {
        "snapshot_count": 5,
        "total_alerts": 6,
        "peak_alerts": 3,
        "alert_concentration": 3 / 6,
        "peak_alert_share": 3 / 6,
        "concentration_gap": 3 / 6 - 1 / 5,
    }


def test_build_alert_concentration_summary_no_alerts():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    result = build_alert_concentration_summary(history)

    assert result == {
        "snapshot_count": 3,
        "total_alerts": 0,
        "peak_alerts": 0,
        "alert_concentration": 0.0,
        "peak_alert_share": 0.0,
        "concentration_gap": -1 / 3,
    }


def test_build_alert_concentration_summary_empty():
    with pytest.raises(ValueError):
        build_alert_concentration_summary([])


def test_rejects_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_alert_concentration({})


def test_rejects_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_alert_concentration(
            [{"alert_count": 1}, "invalid"]
        )


def test_rejects_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_concentration(
            [{"value": 1}]
        )


def test_rejects_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_concentration(
            [{"alert_count": 1.5}]
        )


def test_rejects_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_concentration(
            [{"alert_count": True}]
        )


def test_rejects_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_concentration(
            [{"alert_count": -1}]
        )
