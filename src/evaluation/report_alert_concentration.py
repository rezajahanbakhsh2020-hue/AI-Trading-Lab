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


def calculate_alert_concentration(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the share of all alerts occurring in the
    most alert-heavy snapshot.
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

    if total_alerts == 0:
        return 0.0

    peak_alerts = max(
        item["alert_count"]
        for item in history
    )

    return peak_alerts / total_alerts


def calculate_peak_alert_share(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the share of total alerts represented by
    the peak snapshot.
    """
    return calculate_alert_concentration(history)


def calculate_alert_concentration_gap(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the difference between the peak alert share
    and an evenly distributed alert share.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    concentration = calculate_alert_concentration(history)
    uniform_share = 1.0 / len(history)

    return concentration - uniform_share


def build_alert_concentration_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-concentration summary.
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

    peak_alerts = max(
        item["alert_count"]
        for item in history
    )

    concentration = calculate_alert_concentration(history)

    return {
        "snapshot_count": len(history),
        "total_alerts": total_alerts,
        "peak_alerts": peak_alerts,
        "alert_concentration": concentration,
        "peak_alert_share": concentration,
        "concentration_gap": (
            concentration - (1.0 / len(history))
        ),
    }
