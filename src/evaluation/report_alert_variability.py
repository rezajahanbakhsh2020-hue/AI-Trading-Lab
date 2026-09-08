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


def calculate_alert_variability(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the population standard deviation of alert counts.
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

    variance = sum(
        (value - mean) ** 2
        for value in values
    ) / len(values)

    return variance ** 0.5


def calculate_alert_range(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Calculate the range between maximum and minimum alert counts.
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

    return max(values) - min(values)


def calculate_alert_variability_ratio(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate variability relative to the average alert count.
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
        return 0.0

    return calculate_alert_variability(history) / mean


def build_alert_variability_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-variability summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    variability = calculate_alert_variability(history)

    return {
        "snapshot_count": len(history),
        "alert_variability": variability,
        "alert_range": calculate_alert_range(history),
        "alert_variability_ratio": (
            calculate_alert_variability_ratio(history)
        ),
    }
