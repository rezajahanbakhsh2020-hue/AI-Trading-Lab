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


def count_peak_observations(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Count observations whose alert count equals the peak.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    peak = calculate_peak_alert_count(history)

    return sum(
        1
        for item in history
        if item["alert_count"] == peak
    )


def calculate_peak_occurrence_ratio(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the proportion of observations occurring at peak alert count.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return count_peak_observations(history) / len(history)


def calculate_peak_occurrence_percentage(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the percentage of observations occurring at peak alert count.
    """
    return calculate_peak_occurrence_ratio(history) * 100.0


def calculate_non_peak_ratio(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the proportion of observations below the peak alert count.
    """
    return 1.0 - calculate_peak_occurrence_ratio(history)


def build_alert_peak_ratio_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete peak-occurrence ratio summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    peak = calculate_peak_alert_count(history)
    peak_count = count_peak_observations(history)
    ratio = calculate_peak_occurrence_ratio(history)

    return {
        "snapshot_count": len(history),
        "peak_alert_count": peak,
        "peak_observation_count": peak_count,
        "peak_occurrence_ratio": ratio,
        "peak_occurrence_percentage": ratio * 100.0,
        "non_peak_ratio": 1.0 - ratio,
    }
