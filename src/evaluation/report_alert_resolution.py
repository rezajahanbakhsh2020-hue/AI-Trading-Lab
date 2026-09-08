from __future__ import annotations

from typing import Any, Mapping


def calculate_resolved_alerts(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> int:
    """
    Calculate how many alerts were resolved between two snapshots.

    A reduction in alert_count represents resolved alerts.
    """
    if not isinstance(previous, Mapping):
        raise TypeError("previous must be a mapping.")

    if not isinstance(current, Mapping):
        raise TypeError("current must be a mapping.")

    previous_count = previous.get("alert_count")
    current_count = current.get("alert_count")

    for name, value in (
        ("previous", previous_count),
        ("current", current_count),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
        ):
            raise ValueError(
                f"{name} must contain an integer alert_count."
            )

        if value < 0:
            raise ValueError(
                f"{name} alert_count must not be negative."
            )

    return max(
        previous_count - current_count,
        0,
    )


def calculate_new_alerts(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> int:
    """
    Calculate how many new alerts appeared between two snapshots.
    """
    if not isinstance(previous, Mapping):
        raise TypeError("previous must be a mapping.")

    if not isinstance(current, Mapping):
        raise TypeError("current must be a mapping.")

    previous_count = previous.get("alert_count")
    current_count = current.get("alert_count")

    for name, value in (
        ("previous", previous_count),
        ("current", current_count),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
        ):
            raise ValueError(
                f"{name} must contain an integer alert_count."
            )

        if value < 0:
            raise ValueError(
                f"{name} alert_count must not be negative."
            )

    return max(
        current_count - previous_count,
        0,
    )


def calculate_alert_change(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> int:
    """
    Calculate the signed change in alert count.
    """
    if not isinstance(previous, Mapping):
        raise TypeError("previous must be a mapping.")

    if not isinstance(current, Mapping):
        raise TypeError("current must be a mapping.")

    previous_count = previous.get("alert_count")
    current_count = current.get("alert_count")

    for name, value in (
        ("previous", previous_count),
        ("current", current_count),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
        ):
            raise ValueError(
                f"{name} must contain an integer alert_count."
            )

        if value < 0:
            raise ValueError(
                f"{name} alert_count must not be negative."
            )

    return current_count - previous_count


def build_alert_resolution_summary(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build a summary of alert resolution between two snapshots.
    """
    change = calculate_alert_change(
        previous,
        current,
    )

    resolved = calculate_resolved_alerts(
        previous,
        current,
    )

    new_alerts = calculate_new_alerts(
        previous,
        current,
    )

    return {
        "previous_alert_count": previous["alert_count"],
        "current_alert_count": current["alert_count"],
        "change": change,
        "resolved_alerts": resolved,
        "new_alerts": new_alerts,
    }
