"""Validate freshness and basic health of persisted live snapshot history."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from live_snapshot_history import (
    DEFAULT_HISTORY_PATH,
    load_live_snapshot_history,
)


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp into an aware UTC datetime."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def assess_live_snapshot_history(
    path: str | Path = DEFAULT_HISTORY_PATH,
    *,
    now: datetime | None = None,
    max_age_seconds: float = 300.0,
) -> dict[str, Any]:
    """
    Assess whether persisted live snapshot history is healthy and fresh.

    The function is intentionally read-only. It does not modify the history
    file and does not create a new snapshot.

    Returns:
        A dictionary containing record count, latest snapshot metadata,
        age information, freshness status, and basic history health.
    """
    if max_age_seconds < 0:
        raise ValueError("max_age_seconds must be >= 0")

    records = load_live_snapshot_history(path)

    if not records:
        return {
            "healthy": False,
            "fresh": False,
            "records": 0,
            "latest_snapshot": None,
            "latest_timestamp": None,
            "age_seconds": None,
            "max_age_seconds": float(max_age_seconds),
            "timestamp_valid": False,
            "reason": "empty_history",
        }

    latest = records[-1]
    latest_timestamp = latest.get("timestamp")
    parsed_latest = _parse_timestamp(latest_timestamp)

    if parsed_latest is None:
        return {
            "healthy": False,
            "fresh": False,
            "records": len(records),
            "latest_snapshot": latest,
            "latest_timestamp": (
                str(latest_timestamp)
                if latest_timestamp is not None
                else None
            ),
            "age_seconds": None,
            "max_age_seconds": float(max_age_seconds),
            "timestamp_valid": False,
            "reason": "invalid_latest_timestamp",
        }

    reference_now = now or datetime.now(timezone.utc)

    if reference_now.tzinfo is None:
        reference_now = reference_now.replace(tzinfo=timezone.utc)

    reference_now = reference_now.astimezone(timezone.utc)

    age_seconds = (reference_now - parsed_latest).total_seconds()

    if age_seconds < 0:
        return {
            "healthy": False,
            "fresh": False,
            "records": len(records),
            "latest_snapshot": latest,
            "latest_timestamp": parsed_latest.isoformat(),
            "age_seconds": age_seconds,
            "max_age_seconds": float(max_age_seconds),
            "timestamp_valid": True,
            "reason": "latest_timestamp_in_future",
        }

    fresh = age_seconds <= max_age_seconds

    return {
        "healthy": True,
        "fresh": fresh,
        "records": len(records),
        "latest_snapshot": latest,
        "latest_timestamp": parsed_latest.isoformat(),
        "age_seconds": age_seconds,
        "max_age_seconds": float(max_age_seconds),
        "timestamp_valid": True,
        "reason": "fresh" if fresh else "stale",
    }


def is_live_snapshot_history_fresh(
    path: str | Path = DEFAULT_HISTORY_PATH,
    *,
    now: datetime | None = None,
    max_age_seconds: float = 300.0,
) -> bool:
    """Return True only when the latest live snapshot is fresh and valid."""
    result = assess_live_snapshot_history(
        path,
        now=now,
        max_age_seconds=max_age_seconds,
    )
    return bool(result["healthy"] and result["fresh"])
