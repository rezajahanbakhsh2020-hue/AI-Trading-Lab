from __future__ import annotations

import pytest

from src.evaluation.report_alert_severity import (
    build_alert_severity_summary,
    classify_alert_history_severity,
    classify_alert_severity,
    count_alert_severities,
)


def test_classify_alert_severity_normal():
    assert classify_alert_severity(0) == "normal"


def test_classify_alert_severity_warning():
    assert classify_alert_severity(1) == "warning"
    assert classify_alert_severity(2) == "warning"


def test_classify_alert_severity_critical():
    assert classify_alert_severity(3) == "critical"
    assert classify_alert_severity(5) == "critical"


def test_classify_alert_severity_custom_limits():
    assert classify_alert_severity(
        2,
        warning_limit=2,
        critical_limit=5,
    ) == "warning"

    assert classify_alert_severity(
        5,
        warning_limit=2,
        critical_limit=5,
    ) == "critical"


def test_classify_alert_severity_rejects_invalid_alert_count():
    with pytest.raises(TypeError):
        classify_alert_severity(1.5)


def test_classify_alert_severity_rejects_bool():
    with pytest.raises(TypeError):
        classify_alert_severity(True)


def test_classify_alert_severity_rejects_negative_count():
    with pytest.raises(ValueError):
        classify_alert_severity(-1)


def test_classify_alert_severity_rejects_invalid_warning_limit():
    with pytest.raises(TypeError):
        classify_alert_severity(
            1,
            warning_limit=1.5,
        )


def test_classify_alert_severity_rejects_invalid_critical_limit():
    with pytest.raises(TypeError):
        classify_alert_severity(
            1,
            critical_limit=1.5,
        )


def test_classify_alert_severity_rejects_invalid_limits():
    with pytest.raises(ValueError):
        classify_alert_severity(
            1,
            warning_limit=4,
            critical_limit=3,
        )


def test_classify_alert_history_severity():
    history = [
        {"alert_count": 0},
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert classify_alert_history_severity(history) == [
        "normal",
        "warning",
        "warning",
        "critical",
    ]


def test_classify_alert_history_severity_empty():
    assert classify_alert_history_severity([]) == []


def test_classify_alert_history_severity_rejects_invalid_history():
    with pytest.raises(TypeError):
        classify_alert_history_severity({})


def test_classify_alert_history_severity_rejects_invalid_item():
    with pytest.raises(TypeError):
        classify_alert_history_severity(
            [{"alert_count": 1}, "invalid"]
        )


def test_classify_alert_history_severity_rejects_invalid_count():
    with pytest.raises(ValueError):
        classify_alert_history_severity(
            [{"alert_count": 1.5}]
        )


def test_count_alert_severities():
    severities = [
        "normal",
        "warning",
        "critical",
        "warning",
        "normal",
    ]

    assert count_alert_severities(severities) == {
        "normal": 2,
        "warning": 2,
        "critical": 1,
    }


def test_count_alert_severities_empty():
    assert count_alert_severities([]) == {
        "normal": 0,
        "warning": 0,
        "critical": 0,
    }


def test_count_alert_severities_rejects_invalid_list():
    with pytest.raises(TypeError):
        count_alert_severities("warning")


def test_count_alert_severities_rejects_invalid_severity():
    with pytest.raises(ValueError):
        count_alert_severities(
            ["normal", "unknown"]
        )


def test_count_alert_severities_rejects_non_string():
    with pytest.raises(TypeError):
        count_alert_severities(
            ["normal", 1]
        )


def test_build_alert_severity_summary():
    history = [
        {"alert_count": 0},
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
    ]

    result = build_alert_severity_summary(history)

    assert result == {
        "snapshot_count": 4,
        "severities": [
            "normal",
            "warning",
            "critical",
            "critical",
        ],
        "severity_counts": {
            "normal": 1,
            "warning": 1,
            "critical": 2,
        },
    }


def test_build_alert_severity_summary_empty():
    result = build_alert_severity_summary([])

    assert result == {
        "snapshot_count": 0,
        "severities": [],
        "severity_counts": {
            "normal": 0,
            "warning": 0,
            "critical": 0,
        },
    }
