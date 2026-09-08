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


def calculate_alert_balance_ratio(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the ratio of the minimum alert count to the
    average alert count.
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

    return min(values) / mean


def calculate_alert_balance_gap(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the gap between perfect balance and the
    observed balance ratio.
    """
    return 1.0 - calculate_alert_balance_ratio(history)


def build_alert_balance_ratio_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-balance-ratio summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return {
        "snapshot_count": len(history),
        "alert_balance_ratio": (
            calculate_alert_balance_ratio(history)
        ),
        "alert_balance_gap": (
            calculate_alert_balance_gap(history)
        ),
    }
