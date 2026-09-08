from __future__ import annotations

from typing import Any, Mapping


def evaluate_report_thresholds(
    report: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, bool]:
    """
    Evaluate numeric report metrics against minimum thresholds.

    A metric passes when its report value is greater than or equal
    to the configured threshold.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    if not isinstance(thresholds, Mapping):
        raise TypeError("thresholds must be a mapping.")

    results: dict[str, bool] = {}

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
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            results[metric] = value >= threshold
        else:
            results[metric] = False

    return results


def find_failed_thresholds(
    report: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> list[str]:
    """
    Return metrics that fail their configured thresholds.
    """
    results = evaluate_report_thresholds(
        report,
        thresholds,
    )

    return [
        metric
        for metric, passed in results.items()
        if not passed
    ]


def is_report_within_thresholds(
    report: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> bool:
    """
    Return True when every configured threshold passes.
    """
    results = evaluate_report_thresholds(
        report,
        thresholds,
    )

    return all(results.values())


def build_threshold_summary(
    report: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """
    Build a compact threshold evaluation summary.
    """
    results = evaluate_report_thresholds(
        report,
        thresholds,
    )

    failed = [
        metric
        for metric, passed in results.items()
        if not passed
    ]

    return {
        "thresholds": dict(thresholds),
        "results": results,
        "failed_metrics": failed,
        "passed": not failed,
    }
