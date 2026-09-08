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


def calculate_peak_alert_count(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Calculate the maximum alert count observed.
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


def count_peak_occurrences(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Count how many snapshots reached the peak alert count.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    peak = calculate_peak_alert_count(history)

    return sum(
        item["alert_count"] == peak
        for item in history
    )


def calculate_peak_frequency(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the proportion of snapshots that reached the peak.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return count_peak_occurrences(history) / len(history)


def build_alert_peak_frequency_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete peak-frequency summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    peak = calculate_peak_alert_count(history)
    occurrences = count_peak_occurrences(history)

    return {
        "snapshot_count": len(history),
        "peak_alert_count": peak,
        "peak_occurrences": occurrences,
        "peak_frequency": occurrences / len(history),
    }
