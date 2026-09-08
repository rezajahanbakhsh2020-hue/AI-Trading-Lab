from __future__ import annotations

from typing import Any, Mapping


def calculate_report_health_score(
    report: Mapping[str, Any],
    required_metrics: list[str],
) -> float:
    """
    Calculate a report health score from metric validity.

    The score is the percentage of required metrics that contain
    finite numeric values.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    if not isinstance(required_metrics, list):
        raise TypeError("required_metrics must be a list.")

    if not required_metrics:
        raise ValueError(
            "required_metrics must not be empty."
        )

    for metric in required_metrics:
        if not isinstance(metric, str):
            raise TypeError(
                "metric names must be strings."
            )

    valid_count = 0

    for metric in required_metrics:
        value = report.get(metric)

        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            valid_count += 1

    return round(
        valid_count / len(required_metrics),
        10,
    )


def find_invalid_report_metrics(
    report: Mapping[str, Any],
    required_metrics: list[str],
) -> list[str]:
    """
    Return required metrics that are missing or non-numeric.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    if not isinstance(required_metrics, list):
        raise TypeError("required_metrics must be a list.")

    for metric in required_metrics:
        if not isinstance(metric, str):
            raise TypeError(
                "metric names must be strings."
            )

    invalid: list[str] = []

    for metric in required_metrics:
        value = report.get(metric)

        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            invalid.append(metric)

    return invalid


def is_report_healthy(
    report: Mapping[str, Any],
    required_metrics: list[str],
) -> bool:
    """
    Return True when all required metrics are numeric.
    """
    return not find_invalid_report_metrics(
        report,
        required_metrics,
    )


def build_report_health_summary(
    report: Mapping[str, Any],
    required_metrics: list[str],
) -> dict[str, Any]:
    """
    Build a compact health summary for an evaluation report.
    """
    score = calculate_report_health_score(
        report,
        required_metrics,
    )

    invalid = find_invalid_report_metrics(
        report,
        required_metrics,
    )

    return {
        "score": score,
        "healthy": not invalid,
        "invalid_metrics": invalid,
    }
