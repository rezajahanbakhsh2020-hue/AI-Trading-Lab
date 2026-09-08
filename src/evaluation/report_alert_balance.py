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


def calculate_alert_balance(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the balance score of alert counts across snapshots.

    The score is defined as:
        minimum_alert_count / maximum_alert_count

    A score of 1.0 means all snapshots have the same alert count.
    If the maximum alert count is zero, the history is perfectly balanced
    and the function returns 1.0.
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

    maximum = max(values)

    if maximum == 0:
        return 1.0

    return min(values) / maximum


def calculate_alert_imbalance(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate alert imbalance as the complement of the balance score.
    """
    return 1.0 - calculate_alert_balance(history)


def calculate_alert_balance_range(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Calculate the absolute difference between maximum and minimum alerts.
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


def build_alert_balance_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-balance summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return {
        "snapshot_count": len(history),
        "alert_balance": calculate_alert_balance(history),
        "alert_imbalance": calculate_alert_imbalance(history),
        "alert_balance_range": calculate_alert_balance_range(history),
    }
