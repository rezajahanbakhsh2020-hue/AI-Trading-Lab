from __future__ import annotations

from typing import Any, Mapping


def calculate_alert_durations(
    history: list[Mapping[str, Any]],
) -> list[int]:
    """
    Calculate the duration of consecutive alert periods.

    A snapshot with alert_count > 0 is considered active.
    Each returned value represents the length of one consecutive
    active-alert period.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    durations: list[int] = []
    current_duration = 0

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
            current_duration += 1
        elif current_duration > 0:
            durations.append(current_duration)
            current_duration = 0

    if current_duration > 0:
        durations.append(current_duration)

    return durations


def calculate_total_alert_duration(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Calculate the total number of snapshots containing alerts.
    """
    durations = calculate_alert_durations(history)

    return sum(durations)


def calculate_longest_alert_duration(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Return the longest consecutive alert period.
    """
    durations = calculate_alert_durations(history)

    if not durations:
        return 0

    return max(durations)


def calculate_average_alert_duration(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the average duration of alert periods.

    Returns zero when no alert period exists.
    """
    durations = calculate_alert_durations(history)

    if not durations:
        return 0.0

    return sum(durations) / len(durations)


def build_alert_duration_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-duration summary.
    """
    durations = calculate_alert_durations(history)

    total_duration = sum(durations)
    longest_duration = (
        max(durations)
        if durations
        else 0
    )
    average_duration = (
        sum(durations) / len(durations)
        if durations
        else 0.0
    )

    return {
        "snapshot_count": len(history),
        "alert_period_count": len(durations),
        "durations": durations,
        "total_alert_duration": total_duration,
        "longest_alert_duration": longest_duration,
        "average_alert_duration": average_duration,
    }
