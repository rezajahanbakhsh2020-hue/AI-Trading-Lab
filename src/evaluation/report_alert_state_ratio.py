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


def calculate_active_state_ratio(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the ratio of snapshots with an active alert state.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    active_snapshots = sum(
        item["alert_count"] > 0
        for item in history
    )

    return active_snapshots / len(history)


def calculate_inactive_state_ratio(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the ratio of snapshots with an inactive alert state.
    """
    return 1.0 - calculate_active_state_ratio(history)


def build_alert_state_ratio_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-state-ratio summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return {
        "snapshot_count": len(history),
        "active_state_ratio": (
            calculate_active_state_ratio(history)
        ),
        "inactive_state_ratio": (
            calculate_inactive_state_ratio(history)
        ),
    }
