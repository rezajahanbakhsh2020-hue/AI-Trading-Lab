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


def calculate_alert_continuity(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the continuity ratio of the alert state.

    Continuity is the proportion of consecutive transitions that
    preserve the same active/inactive state.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    if len(history) < 2:
        return 1.0

    states = [
        item["alert_count"] > 0
        for item in history
    ]

    continuous_transitions = sum(
        states[index] == states[index - 1]
        for index in range(1, len(states))
    )

    return continuous_transitions / (len(states) - 1)


def calculate_alert_discontinuity(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the proportion of transitions that break continuity.
    """
    return 1.0 - calculate_alert_continuity(history)


def build_alert_continuity_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-continuity summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return {
        "snapshot_count": len(history),
        "transition_count": max(len(history) - 1, 0),
        "alert_continuity": calculate_alert_continuity(history),
        "alert_discontinuity": calculate_alert_discontinuity(history),
    }
