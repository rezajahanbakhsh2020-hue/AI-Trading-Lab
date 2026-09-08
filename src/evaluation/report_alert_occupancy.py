from __future__ import annotations

from typing import Any, Mapping


def _validate_history(
    history: list[Mapping[str, Any]],
) -> None:
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    for item in history:
        if not isinstance(item, Mapping):
            raise TypeError(
                "Each history item must be a mapping."
            )

        value = item.get("alert_count")

        if (
            not isinstance(value, int)
            or isinstance(value, bool)
        ):
            raise ValueError(
                "Each history item must contain an integer alert_count."
            )

        if value < 0:
            raise ValueError(
                "alert_count must not be negative."
            )


def calculate_alert_occupancy(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the proportion of snapshots occupied by active alerts.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    active_snapshots = sum(
        item["alert_count"] > 0
        for item in history
    )

    return active_snapshots / len(history)


def calculate_alert_vacancy(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the proportion of snapshots without active alerts.
    """
    return 1.0 - calculate_alert_occupancy(history)


def build_alert_occupancy_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-occupancy summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    active_snapshots = sum(
        item["alert_count"] > 0
        for item in history
    )

    return {
        "snapshot_count": len(history),
        "active_snapshots": active_snapshots,
        "inactive_snapshots": len(history) - active_snapshots,
        "alert_occupancy": calculate_alert_occupancy(history),
        "alert_vacancy": calculate_alert_vacancy(history),
    }
