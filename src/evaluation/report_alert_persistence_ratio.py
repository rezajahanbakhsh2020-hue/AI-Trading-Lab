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


def calculate_alert_persistence_ratio(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the ratio of snapshots that preserve the previous
    alert state.

    A transition is considered persistent when two consecutive
    snapshots are both active or both inactive.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    values = [
        item["alert_count"] > 0
        for item in history
    ]

    if len(values) < 2:
        return 1.0

    persistent_transitions = sum(
        values[index] == values[index - 1]
        for index in range(1, len(values))
    )

    return persistent_transitions / (len(values) - 1)


def calculate_alert_change_ratio(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the ratio of transitions that change alert state.
    """
    return 1.0 - calculate_alert_persistence_ratio(history)


def build_alert_persistence_ratio_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-persistence-ratio summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return {
        "snapshot_count": len(history),
        "alert_persistence_ratio": (
            calculate_alert_persistence_ratio(history)
        ),
        "alert_change_ratio": (
            calculate_alert_change_ratio(history)
        ),
    }
