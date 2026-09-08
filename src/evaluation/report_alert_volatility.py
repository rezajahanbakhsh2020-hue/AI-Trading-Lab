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


def calculate_alert_volatility(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the population standard deviation of changes
    between consecutive alert-count snapshots.
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
        return 0.0

    changes = [
        values[index] - values[index - 1]
        for index in range(1, len(values))
    ]

    mean_change = sum(changes) / len(changes)

    variance = sum(
        (change - mean_change) ** 2
        for change in changes
    ) / len(changes)

    return variance ** 0.5


def calculate_alert_change_range(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Calculate the range of consecutive alert-count changes.
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

    changes = [
        values[index] - values[index - 1]
        for index in range(1, len(values))
    ]

    return max(changes) - min(changes)


def build_alert_volatility_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-volatility summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return {
        "snapshot_count": len(history),
        "alert_volatility": calculate_alert_volatility(
            history
        ),
        "alert_change_range": calculate_alert_change_range(
            history
        ),
    }
