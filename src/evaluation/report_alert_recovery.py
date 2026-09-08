from __future__ import annotations

from typing import Any, Mapping


def calculate_recovery_durations(
    history: list[Mapping[str, Any]],
) -> list[int]:
    """
    Calculate the number of snapshots required to recover from
    each alert period.

    A snapshot with alert_count > 0 is considered an active alert.
    Recovery occurs when a later snapshot has alert_count == 0.

    Each returned value represents the number of transitions from
    the first active-alert snapshot to the first zero-alert snapshot.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    durations: list[int] = []
    alert_start_index: int | None = None

    for index, item in enumerate(history):
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
            if alert_start_index is None:
                alert_start_index = index
        elif alert_start_index is not None:
            durations.append(index - alert_start_index)
            alert_start_index = None

    return durations


def calculate_total_recovery_duration(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Calculate the total duration of all completed alert recoveries.
    """
    return sum(
        calculate_recovery_durations(history)
    )


def calculate_average_recovery_duration(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the average duration of completed alert recoveries.

    Returns zero when no completed recovery exists.
    """
    durations = calculate_recovery_durations(history)

    if not durations:
        return 0.0

    return sum(durations) / len(durations)


def calculate_fastest_recovery_duration(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Return the shortest completed alert recovery duration.
    """
    durations = calculate_recovery_durations(history)

    if not durations:
        return 0

    return min(durations)


def calculate_slowest_recovery_duration(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Return the longest completed alert recovery duration.
    """
    durations = calculate_recovery_durations(history)

    if not durations:
        return 0

    return max(durations)


def count_recovered_alert_periods(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Count alert periods that have fully recovered.
    """
    return len(
        calculate_recovery_durations(history)
    )


def build_alert_recovery_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-recovery summary.

    An alert period that remains active at the end of history is
    considered unresolved and is not included in recovery durations.
    """
    durations = calculate_recovery_durations(history)

    total_duration = sum(durations)
    recovery_count = len(durations)

    average_duration = (
        total_duration / recovery_count
        if recovery_count
        else 0.0
    )

    fastest_duration = (
        min(durations)
        if durations
        else 0
    )

    slowest_duration = (
        max(durations)
        if durations
        else 0
    )

    return {
        "snapshot_count": len(history),
        "recovered_period_count": recovery_count,
        "recovery_durations": durations,
        "total_recovery_duration": total_duration,
        "average_recovery_duration": average_duration,
        "fastest_recovery_duration": fastest_duration,
        "slowest_recovery_duration": slowest_duration,
    }
