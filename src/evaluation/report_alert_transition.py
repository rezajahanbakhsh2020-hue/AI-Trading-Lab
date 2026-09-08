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


def calculate_alert_transitions(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Count transitions between consecutive alert states.

    A transition occurs whenever the alert count changes from one
    snapshot to the next.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    values = [
        item["alert_count"]
        for item in history
    ]

    if len(values) < 2:
        return 0

    return sum(
        values[index] != values[index - 1]
        for index in range(1, len(values))
    )


def calculate_alert_increases(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Count transitions where the alert count increases.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    values = [
        item["alert_count"]
        for item in history
    ]

    if len(values) < 2:
        return 0

    return sum(
        values[index] > values[index - 1]
        for index in range(1, len(values))
    )


def calculate_alert_decreases(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Count transitions where the alert count decreases.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    values = [
        item["alert_count"]
        for item in history
    ]

    if len(values) < 2:
        return 0

    return sum(
        values[index] < values[index - 1]
        for index in range(1, len(values))
    )


def build_alert_transition_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-transition summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return {
        "snapshot_count": len(history),
        "alert_transitions": calculate_alert_transitions(
            history
        ),
        "alert_increases": calculate_alert_increases(
            history
        ),
        "alert_decreases": calculate_alert_decreases(
            history
        ),
    }
