from __future__ import annotations

from datetime import datetime, timezone

from live_snapshot_history import append_live_snapshot
from live_snapshot_history_health import (
    assess_live_snapshot_history,
    is_live_snapshot_history_fresh,
)


def sample_snapshot(
    *,
    signal: int = 0,
    signal_label: str = "NO TRADE",
    timestamp: str = "2026-09-09T15:30:00+00:00",
) -> dict:
    return {
        "symbol": "XAUUSD",
        "interval": "5m",
        "signal": signal,
        "signal_label": signal_label,
        "trend": "UP",
        "strategy": "momentum",
        "momentum": -0.0053,
        "entry_price": 4395.596,
        "stop_loss": None,
        "take_profit": None,
        "risk_reward_ratio": None,
        "timestamp": timestamp,
    }


def test_empty_history_is_unhealthy(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text("", encoding="utf-8")

    result = assess_live_snapshot_history(path)

    assert result["healthy"] is False
    assert result["fresh"] is False
    assert result["records"] == 0
    assert result["latest_snapshot"] is None
    assert result["reason"] == "empty_history"


def test_recent_snapshot_is_fresh(tmp_path):
    path = tmp_path / "history.jsonl"

    append_live_snapshot(
        sample_snapshot(
            timestamp="2026-09-09T15:58:00+00:00",
        ),
        path,
    )

    now = datetime(2026, 9, 9, 16, 0, tzinfo=timezone.utc)

    result = assess_live_snapshot_history(
        path,
        now=now,
        max_age_seconds=300,
    )

    assert result["healthy"] is True
    assert result["fresh"] is True
    assert result["records"] == 1
    assert result["timestamp_valid"] is True
    assert result["age_seconds"] == 120.0
    assert result["reason"] == "fresh"


def test_old_snapshot_is_stale(tmp_path):
    path = tmp_path / "history.jsonl"

    append_live_snapshot(
        sample_snapshot(
            timestamp="2026-09-09T15:00:00+00:00",
        ),
        path,
    )

    now = datetime(2026, 9, 9, 16, 0, tzinfo=timezone.utc)

    result = assess_live_snapshot_history(
        path,
        now=now,
        max_age_seconds=300,
    )

    assert result["healthy"] is True
    assert result["fresh"] is False
    assert result["age_seconds"] == 3600.0
    assert result["reason"] == "stale"


def test_latest_snapshot_is_used(tmp_path):
    path = tmp_path / "history.jsonl"

    append_live_snapshot(
        sample_snapshot(
            signal=1,
            signal_label="BUY",
            timestamp="2026-09-09T15:00:00+00:00",
        ),
        path,
    )

    append_live_snapshot(
        sample_snapshot(
            signal=0,
            signal_label="NO TRADE",
            timestamp="2026-09-09T15:59:00+00:00",
        ),
        path,
    )

    now = datetime(2026, 9, 9, 16, 0, tzinfo=timezone.utc)

    result = assess_live_snapshot_history(
        path,
        now=now,
        max_age_seconds=300,
    )

    assert result["records"] == 2
    assert result["latest_snapshot"]["signal_label"] == "NO TRADE"
    assert result["latest_timestamp"] == "2026-09-09T15:59:00+00:00"
    assert result["age_seconds"] == 60.0
    assert result["fresh"] is True


def test_invalid_latest_timestamp_is_unhealthy(tmp_path):
    path = tmp_path / "history.jsonl"

    append_live_snapshot(
        sample_snapshot(
            timestamp="not-a-timestamp",
        ),
        path,
    )

    result = assess_live_snapshot_history(path)

    assert result["healthy"] is False
    assert result["fresh"] is False
    assert result["timestamp_valid"] is False
    assert result["reason"] == "invalid_latest_timestamp"


def test_future_latest_timestamp_is_unhealthy(tmp_path):
    path = tmp_path / "history.jsonl"

    append_live_snapshot(
        sample_snapshot(
            timestamp="2026-09-09T17:00:00+00:00",
        ),
        path,
    )

    now = datetime(2026, 9, 9, 16, 0, tzinfo=timezone.utc)

    result = assess_live_snapshot_history(
        path,
        now=now,
        max_age_seconds=300,
    )

    assert result["healthy"] is False
    assert result["fresh"] is False
    assert result["timestamp_valid"] is True
    assert result["age_seconds"] == -3600.0
    assert result["reason"] == "latest_timestamp_in_future"


def test_freshness_helper_returns_boolean(tmp_path):
    path = tmp_path / "history.jsonl"

    append_live_snapshot(
        sample_snapshot(
            timestamp="2026-09-09T15:59:00+00:00",
        ),
        path,
    )

    now = datetime(2026, 9, 9, 16, 0, tzinfo=timezone.utc)

    assert (
        is_live_snapshot_history_fresh(
            path,
            now=now,
            max_age_seconds=300,
        )
        is True
    )

    assert (
        is_live_snapshot_history_fresh(
            path,
            now=now,
            max_age_seconds=30,
        )
        is False
    )


def test_negative_max_age_is_rejected(tmp_path):
    path = tmp_path / "history.jsonl"

    try:
        assess_live_snapshot_history(
            path,
            max_age_seconds=-1,
        )
    except ValueError as exc:
        assert str(exc) == "max_age_seconds must be >= 0"
    else:
        raise AssertionError("Expected ValueError")
