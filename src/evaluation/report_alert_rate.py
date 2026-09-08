from __future__ import annotations

from typing import Any, Mapping


def calculate_alert_rate(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the fraction of snapshots that contain at least one alert.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    if not history:
        raise ValueError(
            "No alert history available."
        )

    active_alerts = 0

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

        if value > 0:
            active_alerts += 1

    return active_alerts / len(history)


def calculate_alert_free_rate(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the fraction of snapshots that contain no alerts.
    """
    return 1.0 - calculate_alert_rate(history)


def count_alert_snapshots(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Count snapshots containing at least one alert.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    count = 0

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

        if value > 0:
            count += 1

    return count


def count_alert_free_snapshots(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Count snapshots containing zero alerts.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    return len(history) - count_alert_snapshots(history)


def build_alert_rate_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-rate summary.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    if not history:
        raise ValueError(
            "No alert history available."
        )

    alert_snapshots = count_alert_snapshots(history)
    alert_free_snapshots = (
        len(history) - alert_snapshots
    )

    return {
        "snapshot_count": len(history),
        "alert_snapshots": alert_snapshots,
        "alert_free_snapshots": alert_free_snapshots,
        "alert_rate": alert_snapshots / len(history),
        "alert_free_rate": alert_free_snapshots / len(history),
    }
