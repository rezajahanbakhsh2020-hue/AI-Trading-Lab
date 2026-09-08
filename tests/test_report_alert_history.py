from __future__ import annotations

import pytest

from src.evaluation.report_alert_history import (
    build_alert_history_summary,
    count_alert_events,
    get_alert_history_count,
    get_latest_alert_snapshot,
    record_alert_history,
)


def test_record_alert_history_appends_snapshot():
    history = [
        {
            "alert_count": 1,
            "level": "low",
        }
    ]

    report = {
        "alert_count": 2,
        "level": "medium",
    }

    result = record_alert_history(
        history,
        report,
    )

    assert result == [
        {
            "alert_count": 1,
            "level": "low",
        },
        {
            "alert_count": 2,
            "level": "medium",
        },
    ]


def test_record_alert_history_does_not_modify_original():
    history = [
        {
            "alert_count": 1,
        }
    ]

    original = [
        {
            "alert_count": 1,
        }
    ]

    record_alert_history(
        history,
        {
            "alert_count": 2,
        },
    )

    assert history == original


def test_record_alert_history_empty_history():
    result = record_alert_history(
        [],
        {
            "alert_count": 0,
        },
    )

    assert result == [
        {
            "alert_count": 0,
        }
    ]


def test_record_alert_history_rejects_invalid_history():
    with pytest.raises(TypeError):
        record_alert_history(
            {},
            {"alert_count": 1},
        )


def test_record_alert_history_rejects_invalid_report():
    with pytest.raises(TypeError):
        record_alert_history(
            [],
            [],
        )


def test_record_alert_history_rejects_invalid_history_item():
    with pytest.raises(TypeError):
        record_alert_history(
            [{"alert_count": 1}, "invalid"],
            {"alert_count": 2},
        )


def test_get_alert_history_count():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
    ]

    assert get_alert_history_count(history) == 3


def test_get_alert_history_count_empty():
    assert get_alert_history_count([]) == 0


def test_get_alert_history_count_rejects_invalid_history():
    with pytest.raises(TypeError):
        get_alert_history_count({})


def test_get_latest_alert_snapshot():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert get_latest_alert_snapshot(history) == {
        "alert_count": 3,
    }


def test_get_latest_alert_snapshot_empty():
    assert get_latest_alert_snapshot([]) is None


def test_get_latest_alert_snapshot_rejects_invalid_item():
    with pytest.raises(TypeError):
        get_latest_alert_snapshot(
            [{"alert_count": 1}, "invalid"]
        )


def test_count_alert_events():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    assert count_alert_events(history) == 4


def test_count_alert_events_missing_count():
    history = [
        {"level": "low"},
        {"alert_count": 2},
    ]

    assert count_alert_events(history) == 2


def test_count_alert_events_rejects_invalid_count():
    with pytest.raises(ValueError):
        count_alert_events(
            [
                {
                    "alert_count": "invalid",
                }
            ]
        )


def test_count_alert_events_rejects_negative_count():
    with pytest.raises(ValueError):
        count_alert_events(
            [
                {
                    "alert_count": -1,
                }
            ]
        )


def test_build_alert_history_summary():
    history = [
        {
            "alert_count": 1,
            "level": "low",
        },
        {
            "alert_count": 2,
            "level": "medium",
        },
        {
            "alert_count": 0,
            "level": "none",
        },
    ]

    result = build_alert_history_summary(history)

    assert result == {
        "snapshot_count": 3,
        "total_alerts": 3,
        "latest": {
            "alert_count": 0,
            "level": "none",
        },
    }


def test_build_alert_history_summary_empty():
    result = build_alert_history_summary([])

    assert result == {
        "snapshot_count": 0,
        "total_alerts": 0,
        "latest": None,
    }
