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


def count_alert_events(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Count the total number of alerts recorded.
    """
    _validate_history(history)

    return sum(
        item["alert_count"]
        for item in history
    )


def calculate_alert_frequency(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate average alert events per snapshot.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return count_alert_events(history) / len(history)


def calculate_alert_event_frequency(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the proportion of total alert events
    relative to the number of snapshots.
    """
    return calculate_alert_frequency(history)


def build_alert_frequency_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-frequency summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    total_alerts = count_alert_events(history)
    frequency = total_alerts / len(history)

    return {
        "snapshot_count": len(history),
        "total_alert_events": total_alerts,
        "alert_frequency": frequency,
        "alerts_per_snapshot": frequency,
    }
