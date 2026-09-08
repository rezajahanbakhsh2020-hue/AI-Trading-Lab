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


def calculate_total_alerts(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Calculate the total number of alerts across all snapshots.
    """
    _validate_history(history)

    return sum(
        item["alert_count"]
        for item in history
    )


def calculate_average_alert_intensity(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the average number of alerts per snapshot.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return calculate_total_alerts(history) / len(history)


def calculate_max_alert_intensity(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Calculate the maximum number of simultaneous alerts
    recorded in a single snapshot.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return max(
        item["alert_count"]
        for item in history
    )


def calculate_min_alert_intensity(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Calculate the minimum number of alerts recorded
    in a single snapshot.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return min(
        item["alert_count"]
        for item in history
    )


def build_alert_intensity_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-intensity summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    total_alerts = calculate_total_alerts(history)

    return {
        "snapshot_count": len(history),
        "total_alerts": total_alerts,
        "average_alert_intensity": (
            total_alerts / len(history)
        ),
        "max_alert_intensity": (
            calculate_max_alert_intensity(history)
        ),
        "
