from __future__ import annotations

from typing import Any, Mapping


def find_report_alerts(
    report: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> list[str]:
    """
    Return alert messages for metrics that fail thresholds.

    A metric triggers an alert when it is missing, non-numeric,
    or below its configured minimum threshold.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    if not isinstance(thresholds, Mapping):
        raise TypeError("thresholds must be a mapping.")

    alerts: list[str] = []

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
        ):
            alerts.append(
                f"{metric}: invalid or missing value"
            )
            continue

        if value < threshold:
            alerts.append(
                f"{metric}: below threshold"
            )

    return alerts


def has_report_alerts(
    report: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> bool:
    """
    Return True when one or more report alerts exist.
    """
    return bool(
        find_report_alerts(
            report,
            thresholds,
        )
    )


def count_report_alerts(
    report: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> int:
    """
    Count the number of triggered report alerts.
    """
    return len(
        find_report_alerts(
            report,
            thresholds,
        )
    )


def build_report_alert_summary(
    report: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """
    Build a compact alert summary for an evaluation report.
    """
    alerts = find_report_alerts(
        report,
        thresholds,
    )

    return {
        "alert_count": len(alerts),
        "has_alerts": bool(alerts),
        "alerts": alerts,
    }
