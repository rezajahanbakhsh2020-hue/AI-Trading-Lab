from __future__ import annotations

from typing import Any, Mapping

from src.evaluation.report_alerts import find_report_alerts


def summarize_report_alerts(
    report: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """
    Build a structured summary of report alerts.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    if not isinstance(thresholds, Mapping):
        raise TypeError("thresholds must be a mapping.")

    alerts = find_report_alerts(
        report,
        thresholds,
    )

    return {
        "alert_count": len(alerts),
        "has_alerts": bool(alerts),
        "alerts": alerts,
    }


def count_alerts_by_type(
    alerts: list[str],
) -> dict[str, int]:
    """
    Count alerts by their message type.
    """
    if not isinstance(alerts, list):
        raise TypeError("alerts must be a list.")

    counts = {
        "invalid_or_missing": 0,
        "below_threshold": 0,
    }

    for alert in alerts:
        if not isinstance(alert, str):
            raise TypeError(
                "Each alert must be a string."
            )

        if ": invalid or missing value" in alert:
            counts["invalid_or_missing"] += 1
        elif ": below threshold" in alert:
            counts["below_threshold"] += 1

    return counts


def build_report_alert_overview(
    report: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """
    Build a complete alert overview for an evaluation report.
    """
    summary = summarize_report_alerts(
        report,
        thresholds,
    )

    alert_types = count_alerts_by_type(
        summary["alerts"],
    )

    return {
        "alert_count": summary["alert_count"],
        "has_alerts": summary["has_alerts"],
        "alerts": summary["alerts"],
        "alert_types": alert_types,
    }
