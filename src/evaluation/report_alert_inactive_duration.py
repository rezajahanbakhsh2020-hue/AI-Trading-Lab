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


def calculate_inactive_durations(
    history: list[Mapping[str, Any]],
) -> list[int]:
    """
    Return the lengths of consecutive inactive-alert periods.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    durations: list[int] = []
    current_duration = 0

    for item in history:
        if item["alert_count"] == 0:
            current_duration += 1
        elif current_duration > 0:
            durations.append(current_duration)
            current_duration = 0

    if current_duration > 0:
        durations.append(current_duration)

    return durations


def calculate_average_inactive_duration(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the average length of consecutive inactive-alert periods.
    """
    durations = calculate_inactive_durations(history)

    if not durations:
        return 0.0

    return sum(durations) / len(durations)


def calculate_longest_inactive_duration(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Calculate the longest consecutive inactive-alert period.
    """
    durations = calculate_inactive_durations(history)

    if not durations:
        return 0

    return max(durations)


def build_alert_inactive_duration_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a summary of inactive-alert durations.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    durations = calculate_inactive_durations(history)

    return {
        "snapshot_count": len(history),
        "inactive_period_count": len(durations),
        "inactive_durations": durations,
        "average_inactive_duration": (
            calculate_average_inactive_duration(history)
        ),
        "longest_inactive_duration": (
            calculate_longest_inactive_duration(history)
        ),
    }
