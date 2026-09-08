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


def calculate_inactive_ratio(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the proportion of snapshots with no active alerts.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    inactive_snapshots = sum(
        item["alert_count"] == 0
        for item in history
    )

    return inactive_snapshots / len(history)


def calculate_active_ratio(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the proportion of snapshots with active alerts.
    """
    return 1.0 - calculate_inactive_ratio(history)


def build_alert_inactive_ratio_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete inactive-alert-ratio summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    inactive_snapshots = sum(
        item["alert_count"] == 0
        for item in history
    )

    return {
        "snapshot_count": len(history),
        "inactive_snapshots": inactive_snapshots,
        "active_snapshots": len(history) - inactive_snapshots,
        "inactive_ratio": calculate_inactive_ratio(history),
        "active_ratio": calculate_active_ratio(history),
    }
