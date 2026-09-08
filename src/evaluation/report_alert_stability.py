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


def calculate_alert_stability(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate alert stability using the inverse of normalized variability.

    Stability is defined as:

        1 / (1 + coefficient_of_variation)

    where coefficient_of_variation is the population standard deviation
    divided by the mean alert count.

    A perfectly constant non-zero alert history has stability 1.0.
    An all-zero history is also considered perfectly stable.
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

    mean = sum(values) / len(values)

    if mean == 0:
        return 1.0

    variance = sum(
        (value - mean) ** 2
        for value in values
    ) / len(values)

    standard_deviation = variance ** 0.5
    coefficient_of_variation = (
        standard_deviation / mean
    )

    return 1.0 / (
        1.0 + coefficient_of_variation
    )


def calculate_alert_instability(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate alert instability as the complement of stability.
    """
    return 1.0 - calculate_alert_stability(history)


def build_alert_stability_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-stability summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    stability = calculate_alert_stability(history)

    return {
        "snapshot_count": len(history),
        "alert_stability": stability,
        "alert_instability": (
            calculate_alert_instability(history)
        ),
    }
