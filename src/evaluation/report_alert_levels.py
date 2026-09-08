from __future__ import annotations

from typing import Any, Mapping


def classify_alert_level(
    alert_count: int,
) -> str:
    """
    Classify an alert count into a severity level.
    """
    if not isinstance(alert_count, int) or isinstance(
        alert_count,
        bool,
    ):
        raise TypeError(
            "alert_count must be an integer."
        )

    if alert_count < 0:
        raise ValueError(
            "alert_count must not be negative."
        )

    if alert_count == 0:
        return "none"

    if alert_count <= 2:
        return "low"

    if alert_count <= 5:
        return "medium"

    return "high"


def calculate_alert_level(
    report: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> str:
    """
    Calculate the alert severity level for a report.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    if not isinstance(thresholds, Mapping):
        raise TypeError("thresholds must be a mapping.")

    alert_count = 0

    for metric, threshold in thresholds.items():
        if not isinstance(metric, str):
            raise TypeError(
                "threshold metric names must be strings."
            )

        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
        ):
            raise TypeError(
                f"Threshold must be numeric: {metric}"
            )

        value = report.get(metric)

        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < threshold
        ):
            alert_count += 1

    return classify_alert_level(alert_count)


def build_alert_level_summary(
    report: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """
    Build a compact alert-level summary.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    if not isinstance(thresholds, Mapping):
        raise TypeError("thresholds must be a mapping.")

    alert_count = 0

    for metric, threshold in thresholds.items():
        if not isinstance(metric, str):
            raise TypeError(
                "threshold metric names must be strings."
            )

        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
        ):
            raise TypeError(
                f"Threshold must be numeric: {metric}"
            )

        value = report.get(metric)

        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < threshold
        ):
            alert_count += 1

    level = classify_alert_level(
        alert_count,
    )

    return {
        "alert_count": alert_count,
        "level": level,
        "has_alerts": alert_count > 0,
    }
