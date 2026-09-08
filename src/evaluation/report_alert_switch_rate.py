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


def calculate_alert_switch_rate(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the rate at which the alert state switches.

    A switch occurs when consecutive snapshots change between
    active and inactive states.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    if len(history) < 2:
        return 0.0

    states = [
        item["alert_count"] > 0
        for item in history
    ]

    switches = sum(
        states[index] != states[index - 1]
        for index in range(1, len(states))
    )

    return switches / (len(states) - 1)


def count_alert_switches(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Count the number of active/inactive state switches.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    if len(history) < 2:
        return 0

    states = [
        item["alert_count"] > 0
        for item in history
    ]

    return sum(
        states[index] != states[index - 1]
        for index in range(1, len(states))
    )


def build_alert_switch_rate_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-switch-rate summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return {
        "snapshot_count": len(history),
        "transition_count": max(len(history) - 1, 0),
        "switch_count": count_alert_switches(history),
        "alert_switch_rate": calculate_alert_switch_rate(history),
    }
