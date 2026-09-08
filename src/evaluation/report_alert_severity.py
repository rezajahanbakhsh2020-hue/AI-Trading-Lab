from __future__ import annotations

from typing import Any, Mapping


def classify_alert_severity(
    alert_count: int,
    warning_limit: int = 1,
    critical_limit: int = 3,
) -> str:
    """
    Classify alert severity from the number of alerts.
    """
    if not isinstance(alert_count, int) or isinstance(
        alert_count,
        bool,
    ):
        raise TypeError(
            "alert_count must be an integer."
        )

    if not isinstance(warning_limit, int) or isinstance(
        warning_limit,
        bool,
    ):
        raise TypeError(
            "warning_limit must be an integer."
        )

    if not isinstance(critical_limit, int) or isinstance(
        critical_limit,
        bool,
    ):
        raise TypeError(
            "critical_limit must be an integer."
        )

    if alert_count < 0:
        raise ValueError(
            "alert_count must not be negative."
        )

    if warning_limit < 0:
        raise ValueError(
            "warning_limit must not be negative."
        )

    if critical_limit < 0:
        raise ValueError(
            "critical_limit must not be negative."
        )

    if warning_limit > critical_limit:
        raise ValueError(
            "warning_limit must not exceed critical_limit."
        )

    if alert_count >= critical_limit:
        return "critical"

    if alert_count >= warning_limit:
        return "warning"

    return "normal"


def classify_alert_history_severity(
    history: list[Mapping[str, Any]],
    warning_limit: int = 1,
    critical_limit: int = 3,
) -> list[str]:
    """
    Classify the severity of every alert-history snapshot.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    result: list[str] = []

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

        result.append(
            classify_alert_severity(
                value,
                warning_limit=warning_limit,
                critical_limit=critical_limit,
            )
        )

    return result


def count_alert_severities(
    severities: list[str],
) -> dict[str, int]:
    """
    Count normal, warning, and critical severity levels.
    """
    if not isinstance(severities, list):
        raise TypeError("severities must be a list.")

    counts = {
        "normal": 0,
        "warning": 0,
        "critical": 0,
    }

    for severity in severities:
        if not isinstance(severity, str):
            raise TypeError(
                "Each severity must be a string."
            )

        if severity not in counts:
            raise ValueError(
                f"Invalid alert severity: {severity}"
            )

        counts[severity] += 1

    return counts


def build_alert_severity_summary(
    history: list[Mapping[str, Any]],
    warning_limit: int = 1,
    critical_limit: int = 3,
) -> dict[str, Any]:
    """
    Build a complete severity summary for alert history.
    """
    severities = classify_alert_history_severity(
        history,
        warning_limit=warning_limit,
        critical_limit=critical_limit,
    )

    counts = count_alert_severities(
        severities,
    )

    return {
        "snapshot_count": len(history),
        "severities": severities,
        "severity_counts": counts,
    }
