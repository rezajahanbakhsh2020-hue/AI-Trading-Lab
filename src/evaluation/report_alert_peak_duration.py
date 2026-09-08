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


def calculate_peak_alert_count(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Calculate the maximum alert count observed.
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


def calculate_peak_durations(
    history: list[Mapping[str, Any]],
) -> list[int]:
    """
    Return consecutive durations during which the peak alert
    count was maintained.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    peak = calculate_peak_alert_count(history)
    durations: list[int] = []
    current_duration = 0

    for item in history:
        if item["alert_count"] == peak:
            current_duration += 1
        elif current_duration > 0:
            durations.append(current_duration)
            current_duration = 0

    if current_duration > 0:
        durations.append(current_duration)

    return durations


def calculate_longest_peak_duration(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Calculate the longest consecutive period at the peak alert count.
    """
    durations = calculate_peak_durations(history)

    if not durations:
        return 0

    return max(durations)


def calculate_average_peak_duration(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the average consecutive duration at the peak alert count.
    """
    durations = calculate_peak_durations(history)

    if not durations:
        return 0.0

    return sum(durations) / len(durations)


def build_alert_peak_duration_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete peak-duration summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    peak = calculate_peak_alert_count(history)
    durations = calculate_peak_durations(history)

    return {
        "snapshot_count": len(history),
        "peak_alert_count": peak,
        "peak_period_count": len(durations),
        "peak_durations": durations,
        "average_peak_duration": (
            calculate_average_peak_duration(history)
        ),
        "longest_peak_duration": (
            calculate_longest_peak_duration(history)
        ),
    }
