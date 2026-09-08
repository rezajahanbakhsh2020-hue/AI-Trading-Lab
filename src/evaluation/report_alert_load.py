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


def calculate_alert_load(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Calculate the total alert load across all snapshots.
    """
    _validate_history(history)

    return sum(
        item["alert_count"]
        for item in history
    )


def calculate_average_alert_load(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the average alert load per snapshot.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return calculate_alert_load(history) / len(history)


def calculate_peak_alert_load(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Calculate the peak alert load in any snapshot.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return max(
        item["alert_count"]
        for item in history
    )


def build_alert_load_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-load summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    total_load = calculate_alert_load(history)

    return {
        "snapshot_count": len(history),
        "total_alert_load": total_load,
        "average_alert_load": (
            total_load / len(history)
        ),
        "peak_alert_load": calculate_peak_alert_load(
            history
        ),
    }
