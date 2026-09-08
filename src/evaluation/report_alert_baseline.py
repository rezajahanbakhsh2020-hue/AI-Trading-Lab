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


def calculate_baseline_alert_count(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the mean alert count as the baseline level.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return sum(
        item["alert_count"]
        for item in history
    ) / len(history)


def calculate_alert_deviations(
    history: list[Mapping[str, Any]],
) -> list[float]:
    """
    Calculate each alert count's deviation from the baseline.
    """
    baseline = calculate_baseline_alert_count(history)

    return [
        item["alert_count"] - baseline
        for item in history
    ]


def calculate_absolute_deviations(
    history: list[Mapping[str, Any]],
) -> list[float]:
    """
    Calculate absolute deviations from the baseline.
    """
    return [
        abs(value)
        for value in calculate_alert_deviations(history)
    ]


def calculate_mean_absolute_deviation(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the mean absolute deviation from the baseline.
    """
    deviations = calculate_absolute_deviations(history)

    return sum(deviations) / len(deviations)


def calculate_baseline_ratio(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the peak alert count relative to the baseline.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    baseline = calculate_baseline_alert_count(history)

    if baseline == 0.0:
        return 0.0

    peak = max(
        item["alert_count"]
        for item in history
    )

    return peak / baseline


def build_alert_baseline_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert baseline summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    baseline = calculate_baseline_alert_count(history)
    deviations = calculate_alert_deviations(history)

    return {
        "snapshot_count": len(history),
        "baseline_alert_count": baseline,
        "peak_alert_count": max(
            item["alert_count"]
            for item in history
        ),
        "minimum_alert_count": min(
            item["alert_count"]
            for item in history
        ),
        "alert_deviations": deviations,
        "mean_absolute_deviation": (
            calculate_mean_absolute_deviation(history)
        ),
        "peak_to_baseline_ratio": (
            calculate_baseline_ratio(history)
        ),
    }
