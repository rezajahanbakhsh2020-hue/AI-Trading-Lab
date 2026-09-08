from __future__ import annotations

from typing import Any, Mapping


def summarize_report_metrics(
    report: Mapping[str, Any],
) -> dict[str, float]:
    """
    Return the main numeric metrics from an evaluation report.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    metric_names = (
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "win_rate",
        "profit_factor",
    )

    summary: dict[str, float] = {}

    for metric in metric_names:
        if metric not in report:
            continue

        value = report[metric]

        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            summary[metric] = float(value)

    return summary


def calculate_metric_count(
    report: Mapping[str, Any],
) -> int:
    """
    Count the available numeric evaluation metrics.
    """
    return len(
        summarize_report_metrics(report)
    )


def calculate_metric_average(
    report: Mapping[str, Any],
) -> float:
    """
    Calculate the arithmetic mean of available numeric metrics.
    """
    summary = summarize_report_metrics(report)

    if not summary:
        raise ValueError(
            "No numeric evaluation metrics are available."
        )

    return sum(summary.values()) / len(summary)


def build_metrics_summary(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build a compact summary containing metrics, count, and average.
    """
    metrics = summarize_report_metrics(report)

    if metrics:
        average = sum(metrics.values()) / len(metrics)
    else:
        average = 0.0

    return {
        "metrics": metrics,
        "count": len(metrics),
        "average": average,
    }
