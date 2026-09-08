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


def calculate_alert_burden(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Calculate the total alert burden across all snapshots.
    """
    _validate_history(history)

    return sum(
        item["alert_count"]
        for item in history
    )


def calculate_average_alert_burden(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the average alert burden per snapshot.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return calculate_alert_burden(history) / len(history)


def calculate_burdened_snapshots(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Count snapshots containing at least one alert.
    """
    _validate_history(history)

    return sum(
        1
        for item in history
        if item["alert_count"] > 0
    )


def calculate_burden_ratio(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the fraction of snapshots carrying alert burden.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return calculate_burdened_snapshots(history) / len(history)


def build_alert_burden_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-burden summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    total_burden = calculate_alert_burden(history)
    burdened_snapshots = calculate_burdened_snapshots(history)

    return {
        "snapshot_count": len(history),
        "total_alert_burden": total_burden,
        "average_alert_burden": (
            total_burden / len(history)
        ),
        "burdened_snapshots": burdened_snapshots,
        "burden_ratio": (
            burdened_snapshots / len(history)
        ),
    }
