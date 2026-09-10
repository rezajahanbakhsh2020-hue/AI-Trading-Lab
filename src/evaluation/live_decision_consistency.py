from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp must be a non-empty string")

    normalized = value.strip()

    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            f"invalid timestamp: {value}"
        ) from exc


def check_live_decision_consistency(
    history: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Check structural and temporal consistency of live decisions.

    This function does not recalculate signals, risk levels, or performance.
    """
    if isinstance(history, (str, bytes)) or not isinstance(
        history, Iterable
    ):
        raise TypeError("history must be iterable")

    records = list(history)

    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("each history item must be a mapping")

    issues: list[str] = []
    timestamps: list[datetime] = []

    for index, record in enumerate(records):
        timestamp = record.get("timestamp")

        try:
            parsed = _parse_timestamp(timestamp)
        except ValueError as exc:
            issues.append(
                f"record {index}: {exc}"
            )
            continue

        timestamps.append(parsed)

        if index > 0 and parsed < timestamps[-2]:
            issues.append(
                f"record {index}: timestamp is earlier than "
                "the previous record"
            )

    duplicate_timestamps = 0

    for index in range(1, len(timestamps)):
        if timestamps[index] == timestamps[index - 1]:
            duplicate_timestamps += 1

    if duplicate_timestamps:
        issues.append(
            f"duplicate consecutive timestamps: "
            f"{duplicate_timestamps}"
        )

    return {
        "consistent": not issues,
        "record_count": len(records),
        "issue_count": len(issues),
        "issues": issues,
        "duplicate_consecutive_timestamps": duplicate_timestamps,
    }


def validate_live_decision_consistency(
    result: Mapping[str, Any],
) -> bool:
    """
    Validate the structure of a consistency-check result.
    """
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")

    required_fields = (
        "consistent",
        "record_count",
        "issue_count",
        "issues",
        "duplicate_consecutive_timestamps",
    )

    missing = [
        field
        for field in required_fields
        if field not in result
    ]

    if missing:
        raise ValueError(
            "consistency result is missing required fields: "
            + ", ".join(missing)
        )

    if not isinstance(result["consistent"], bool):
        raise ValueError("consistent must be boolean")

    for field in (
        "record_count",
        "issue_count",
        "duplicate_consecutive_timestamps",
    ):
        if not isinstance(result[field], int):
            raise ValueError(
                f"{field} must be an integer"
            )

    if not isinstance(result["issues"], list):
        raise ValueError("issues must be a list")

    if result["issue_count"] != len(result["issues"]):
        raise ValueError(
            "issue_count must match the number of issues"
        )

    return True
