from __future__ import annotations

from typing import Any, Mapping


def record_alert_history(
    history: list[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    Append a report alert snapshot to alert history.

    The original history list is not modified.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    for item in history:
        if not isinstance(item, Mapping):
            raise TypeError(
                "Each history item must be a mapping."
            )

    updated_history = [
        dict(item)
        for item in history
    ]

    updated_history.append(dict(report))

    return updated_history


def get_alert_history_count(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Return the number of recorded alert snapshots.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    for item in history:
        if not isinstance(item, Mapping):
            raise TypeError(
                "Each history item must be a mapping."
            )

    return len(history)


def get_latest_alert_snapshot(
    history: list[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """
    Return the latest alert snapshot, or None for empty history.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    for item in history:
        if not isinstance(item, Mapping):
            raise TypeError(
                "Each history item must be a mapping."
            )

    if not history:
        return None

    return history[-1]


def count_alert_events(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Count total alerts recorded across alert history.

    Each history item may contain an ``alert_count`` field.
    Missing or non-numeric counts are treated as zero.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    total = 0

    for item in history:
        if not isinstance(item, Mapping):
            raise TypeError(
                "Each history item must be a mapping."
            )

        value = item.get("alert_count", 0)

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(
                "alert_count must be numeric."
            )

        if value < 0:
            raise ValueError(
                "alert_count must not be negative."
            )

        total += int(value)

    return total


def build_alert_history_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a compact summary of alert history.
    """
    count = get_alert_history_count(history)
    latest = get_latest_alert_snapshot(history)
    total_alerts = count_alert_events(history)

    return {
        "snapshot_count": count,
        "total_alerts": total_alerts,
        "latest": latest,
    }
