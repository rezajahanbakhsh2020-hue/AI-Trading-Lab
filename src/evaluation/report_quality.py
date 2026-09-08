from __future__ import annotations

from typing import Any, Mapping


def calculate_report_quality_score(
    report: Mapping[str, Any],
    required_metrics: list[str] | None = None,
) -> float:
    """
    Calculate a simple quality score based on metric completeness.

    The score is the percentage of required metrics that are present
    with valid numeric values.

    If required_metrics is None, the main evaluation metrics are used.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    if required_metrics is None:
        required_metrics = [
            "total_return",
            "max_drawdown",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "win_rate",
            "profit_factor",
        ]
    elif not isinstance(required_metrics, list):
        raise TypeError("required_metrics must be a list.")

    if not required_metrics:
        raise ValueError(
            "required_metrics must not be empty."
        )

    for metric in required_metrics:
        if not isinstance(metric, str):
            raise TypeError(
                "required metric names must be strings."
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


def find_missing_report_metrics(
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
                "required metric names must be strings."
            )

    missing: list[str] = []

    for metric in required_metrics:
        value = report.get(metric)

        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            missing.append(metric)

    return missing


def is_report_complete(
    report: Mapping[str, Any],
    required_metrics: list[str],
) -> bool:
    """
    Return True when all required metrics are present and numeric.
    """
    return not find_missing_report_metrics(
        report,
        required_metrics,
    )


def build_report_quality_summary(
    report: Mapping[str, Any],
    required_metrics: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build a compact quality summary for an evaluation report.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    if required_metrics is None:
        required_metrics = [
            "total_return",
            "max_drawdown",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "win_rate",
            "profit_factor",
        ]

    score = calculate_report_quality_score(
        report,
        required_metrics=required_metrics,
    )

    missing = find_missing_report_metrics(
        report,
        required_metrics,
    )

    return {
        "score": score,
        "complete": not missing,
        "missing_metrics": missing,
    }
