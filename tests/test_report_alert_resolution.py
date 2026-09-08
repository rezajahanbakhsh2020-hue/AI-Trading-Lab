from __future__ import annotations

import pytest

from src.evaluation.report_alert_resolution import (
    build_alert_resolution_summary,
    calculate_alert_change,
    calculate_new_alerts,
    calculate_resolved_alerts,
)


def test_calculate_resolved_alerts():
    previous = {"alert_count": 5}
    current = {"alert_count": 2}

    assert calculate_resolved_alerts(
        previous,
        current,
    ) == 3


def test_calculate_resolved_alerts_no_reduction():
    previous = {"alert_count": 2}
    current = {"alert_count": 5}

    assert calculate_resolved_alerts(
        previous,
        current,
    ) == 0


def test_calculate_resolved_alerts_same_count():
    previous = {"alert_count": 3}
    current = {"alert_count": 3}

    assert calculate_resolved_alerts(
        previous,
        current,
    ) == 0


def test_calculate_new_alerts():
    previous = {"alert_count": 2}
    current = {"alert_count": 5}

    assert calculate_new_alerts(
        previous,
        current,
    ) == 3


def test_calculate_new_alerts_no_increase():
    previous = {"alert_count": 5}
    current = {"alert_count": 2}

    assert calculate_new_alerts(
        previous,
        current,
    ) == 0


def test_calculate_alert_change_increase():
    assert calculate_alert_change(
        {"alert_count": 2},
        {"alert_count": 5},
    ) == 3


def test_calculate_alert_change_decrease():
    assert calculate_alert_change(
        {"alert_count": 5},
        {"alert_count": 2},
    ) == -3


def test_calculate_alert_change_stable():
    assert calculate_alert_change(
        {"alert_count": 3},
        {"alert_count": 3},
    ) == 0


def test_rejects_invalid_previous_mapping():
    with pytest.raises(TypeError):
        calculate_alert_change(
            [],
            {"alert_count": 1},
        )


def test_rejects_invalid_current_mapping():
    with pytest.raises(TypeError):
        calculate_alert_change(
            {"alert_count": 1},
            [],
        )


def test_rejects_missing_previous_count():
    with pytest.raises(ValueError):
        calculate_alert_change(
            {},
            {"alert_count": 1},
        )


def test_rejects_missing_current_count():
    with pytest.raises(ValueError):
        calculate_alert_change(
            {"alert_count": 1},
            {},
        )


def test_rejects_non_integer_count():
    with pytest.raises(ValueError):
        calculate_alert_change(
            {"alert_count": 1.5},
            {"alert_count": 1},
        )


def test_rejects_boolean_count():
    with pytest.raises(ValueError):
        calculate_alert_change(
            {"alert_count": True},
            {"alert_count": 1},
        )


def test_rejects_negative_previous_count():
    with pytest.raises(ValueError):
        calculate_alert_change(
            {"alert_count": -1},
            {"alert_count": 1},
        )


def test_rejects_negative_current_count():
    with pytest.raises(ValueError):
        calculate_alert_change(
            {"alert_count": 1},
            {"alert_count": -1},
        )


def test_build_alert_resolution_summary_decrease():
    result = build_alert_resolution_summary(
        {"alert_count": 6},
        {"alert_count": 2},
    )

    assert result == {
        "previous_alert_count": 6,
        "current_alert_count": 2,
        "change": -4,
        "resolved_alerts": 4,
        "new_alerts": 0,
    }


def test_build_alert_resolution_summary_increase():
    result = build_alert_resolution_summary(
        {"alert_count": 2},
        {"alert_count": 5},
    )

    assert result == {
        "previous_alert_count": 2,
        "current_alert_count": 5,
        "change": 3,
        "resolved_alerts": 0,
        "new_alerts": 3,
    }


def test_build_alert_resolution_summary_stable():
    result = build_alert_resolution_summary(
        {"alert_count": 3},
        {"alert_count": 3},
    )

    assert result == {
        "previous_alert_count": 3,
        "current_alert_count": 3,
        "change": 0,
        "resolved_alerts": 0,
        "new_alerts": 0,
    }
