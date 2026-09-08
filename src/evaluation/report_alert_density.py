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


def calculate_alert_density(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the average number of alerts per snapshot.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    total_alerts = sum(
        item["alert_count"]
        for item in history
    )

    return total_alerts / len(history)


def calculate_alert_free_density(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the fraction of snapshots with zero alerts.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    alert_free_snapshots = sum(
        1
        for item in history
        if item["alert_count"] == 0
    )

    return alert_free_snapshots / len(history)


def calculate_alert_active_density(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the fraction of snapshots containing alerts.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    active_snapshots = sum(
        1
        for item in history
        if item["alert_count"] > 0
    )

    return active_snapshots / len(history)


def build_alert_density_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-density summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return {
        "snapshot_count": len(history),
        "alert_density": calculate_alert_density(history),
        "alert_free_density": calculate_alert_free_density(
            history
        ),
        "alert_active_density": calculate_alert_active_density(
            history
        ),
    }
