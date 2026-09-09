"""Check temporal continuity of persisted live snapshot history."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from live_snapshot_history import (
    DEFAULT_HISTORY_PATH,
    load_live_snapshot_history,
)


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp into UTC."""
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


def assess_live_snapshot_continuity(
    path: str | Path = DEFAULT_HISTORY_PATH,
    *,
    expected_interval_seconds: float = 300.0,
    max_gap_seconds: float | None = None,
) -> dict[str, Any]:
    """
    Assess temporal continuity of live snapshot history.

    The history remains read-only. The function reports the largest observed
    gap between consecutive valid timestamps and whether that gap exceeds
    the configured threshold.
    """
    if expected_interval_seconds <= 0:
        raise ValueError("expected_interval_seconds must be > 0")

    if max_gap_seconds is None:
        max_gap_seconds = expected_interval_seconds * 2.0

    if max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be > 0")

    records = load_live_snapshot_history(path)

    if len(records) < 2:
        return {
            "healthy": True,
            "continuous": True,
            "records": len(records),
            "valid_timestamps": sum(
                _parse_timestamp(record.get("timestamp")) is not None
                for record in records
            ),
            "intervals_checked": 0,
            "expected_interval_seconds": float(expected_interval_seconds),
            "max_gap_seconds": float(max_gap_seconds),
            "largest_gap_seconds": None,
            "largest_gap_index": None,
            "gap_count": 0,
            "invalid_timestamp_count": len(records)
            - sum(
                _parse_timestamp(record.get("timestamp")) is not None
                for record in records
            ),
            "reason": "insufficient_records",
        }

    parsed: list[tuple[int, datetime | None]] = [
        (index, _parse_timestamp(record.get("timestamp")))
        for index, record in enumerate(records)
    ]

    invalid_timestamp_count = sum(
        timestamp is None for _, timestamp in parsed
    )

    valid = [
        (index, timestamp)
        for index, timestamp in parsed
        if timestamp is not None
    ]

    if len(valid) < 2:
        return {
            "healthy": False,
            "continuous": False,
            "records": len(records),
            "valid_timestamps": len(valid),
            "intervals_checked": 0,
            "expected_interval_seconds": float(expected_interval_seconds),
            "max_gap_seconds": float(max_gap_seconds),
            "largest_gap_seconds": None,
            "largest_gap_index": None,
            "gap_count": 0,
            "invalid_timestamp_count": invalid_timestamp_count,
            "reason": "insufficient_valid_timestamps",
        }

    gaps: list[tuple[int, float]] = []

    for position in range(1, len(valid)):
        previous_index, previous_timestamp = valid[position - 1]
        current_index, current_timestamp = valid[position]

        gap_seconds = (
            current_timestamp - previous_timestamp
        ).total_seconds()

        gaps.append((current_index, gap_seconds))

    largest_gap_index, largest_gap_seconds = max(
        gaps,
        key=lambda item: item[1],
    )

    gap_count = sum(
        gap_seconds > max_gap_seconds
        for _, gap_seconds in gaps
    )

    negative_gap_count = sum(
        gap_seconds < 0
        for _, gap_seconds in gaps
    )

    if negative_gap_count:
        return {
            "healthy": False,
            "continuous": False,
            "records": len(records),
            "valid_timestamps": len(valid),
            "intervals_checked": len(gaps),
            "expected_interval_seconds": float(expected_interval_seconds),
            "max_gap_seconds": float(max_gap_seconds),
            "largest_gap_seconds": largest_gap_seconds,
            "largest_gap_index": largest_gap_index,
            "gap_count": gap_count,
            "negative_gap_count": negative_gap_count,
            "invalid_timestamp_count": invalid_timestamp_count,
            "reason": "timestamps_out_of_order",
        }

    continuous = gap_count == 0 and invalid_timestamp_count == 0

    return {
        "healthy": True,
        "continuous": continuous,
        "records": len(records),
        "valid_timestamps": len(valid),
        "intervals_checked": len(gaps),
        "expected_interval_seconds": float(expected_interval_seconds),
        "max_gap_seconds": float(max_gap_seconds),
        "largest_gap_seconds": largest_gap_seconds,
        "largest_gap_index": largest_gap_index,
        "gap_count": gap_count,
        "negative_gap_count": negative_gap_count,
        "invalid_timestamp_count": invalid_timestamp_count,
        "reason": "continuous" if continuous else "gap_detected",
    }


def is_live_snapshot_history_continuous(
    path: str | Path = DEFAULT_HISTORY_PATH,
    *,
    expected_interval_seconds: float = 300.0,
    max_gap_seconds: float | None = None,
) -> bool:
    """Return True when snapshot history has no invalid or excessive gaps."""
    result = assess_live_snapshot_continuity(
        path,
        expected_interval_seconds=expected_interval_seconds,
        max_gap_seconds=max_gap_seconds,
    )
    return bool(result["healthy"] and result["continuous"])
