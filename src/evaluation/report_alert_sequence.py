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


def calculate_alert_sequences(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Count consecutive sequences of snapshots with alerts.

    A sequence begins when alert_count becomes greater than zero
    after a snapshot with zero alerts.
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

    sequences = 0
    in_sequence = False

    for value in values:
        active = value > 0

        if active and not in_sequence:
            sequences += 1

        in_sequence = active

    return sequences


def calculate_longest_alert_sequence(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Calculate the longest consecutive sequence containing alerts.
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

    longest = 0
    current = 0

    for value in values:
        if value > 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def build_alert_sequence_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-sequence summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return {
        "snapshot_count": len(history),
        "alert_sequences": calculate_alert_sequences(
            history
        ),
        "longest_alert_sequence": (
            calculate_longest_alert_sequence(history)
        ),
    }
