from __future__ import annotations

from typing import Any, Mapping


def evaluate_alert_policy(
    report: Mapping[str, Any],
    thresholds: Mapping[str, float],
    maximum_alerts: int = 0,
) -> dict[str, Any]:
    """
    Evaluate whether a report satisfies an alert policy.

    A metric triggers an alert when it is missing, non-numeric,
    or below its configured threshold.

    The policy passes when the number of alerts is less than or
    equal to maximum_alerts.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    if not isinstance(thresholds, Mapping):
        raise TypeError("thresholds must be a mapping.")

    if not isinstance(maximum_alerts, int) or isinstance(
        maximum_alerts,
        bool,
    ):
        raise TypeError(
            "maximum_alerts must be an integer."
        )

    if maximum_alerts < 0:
        raise ValueError(
            "maximum_alerts must not be negative."
        )

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

    return {
        "alert_count": len(alerts),
        "alerts": alerts,
        "maximum_alerts": maximum_alerts,
        "passed": len(alerts) <= maximum_alerts,
    }


def is_alert_policy_passed(
    report: Mapping[str, Any],
    thresholds: Mapping[str, float],
    maximum_alerts: int = 0,
) -> bool:
    """
    Return True when the report satisfies the alert policy.
    """
    result = evaluate_alert_policy(
        report,
        thresholds,
        maximum_alerts=maximum_alerts,
    )

    return bool(result["passed"])


def build_alert_policy_summary(
    report: Mapping[str, Any],
    thresholds: Mapping[str, float],
    maximum_alerts: int = 0,
) -> dict[str, Any]:
    """
    Build a compact summary of the alert policy evaluation.
    """
    result = evaluate_alert_policy(
        report,
        thresholds,
        maximum_alerts=maximum_alerts,
    )

    return {
        "passed": result["passed"],
        "alert_count": result["alert_count"],
        "maximum_alerts": result["maximum_alerts"],
        "alerts": result["alerts"],
    }
