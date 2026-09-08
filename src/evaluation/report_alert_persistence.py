from __future__ import annotations

from typing import Any, Mapping


def calculate_alert_persistence(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the fraction of consecutive snapshots that both
    contain at least one alert.

    For N snapshots there are N-1 consecutive transitions.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    if not history:
        raise ValueError(
            "No alert history available."
        )

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

    if len(history) < 2:
        return 0.0

    persistent_transitions = 0

    for previous, current in zip(
        history,
        history[1:],
    ):
        if (
            previous["alert_count"] > 0
            and current["alert_count"] > 0
        ):
            persistent_transitions += 1

    return persistent_transitions / (len(history) - 1)


def count_persistent_alert_transitions(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Count consecutive transitions where alerts remain present.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    if not history:
        return 0

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

    if len(history) < 2:
        return 0

    return sum(
        1
        for previous, current in zip(
            history,
            history[1:],
        )
        if (
            previous["alert_count"] > 0
            and current["alert_count"] > 0
        )
    )


def has_persistent_alerts(
    history: list[Mapping[str, Any]],
) -> bool:
    """
    Return True when at least one consecutive alert transition persists.
    """
    return (
        count_persistent_alert_transitions(history) > 0
    )


def build_alert_persistence_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-persistence summary.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    if not history:
        return {
            "snapshot_count": 0,
            "transition_count": 0,
            "persistent_transitions": 0,
            "persistence_ratio": 0.0,
            "has_persistent_alerts": False,
        }

    persistence = calculate_alert_persistence(history)
    transitions = count_persistent_alert_transitions(
        history,
    )

    transition_count = max(
        len(history) - 1,
        0,
    )

    return {
        "snapshot_count": len(history),
        "transition_count": transition_count,
        "persistent_transitions": transitions,
        "persistence_ratio": persistence,
        "has_persistent_alerts": transitions > 0,
    }
