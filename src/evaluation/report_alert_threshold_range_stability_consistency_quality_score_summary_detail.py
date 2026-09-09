from __future__ import annotations

from typing import Any, Mapping

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_summary import (
    build_quality_score_summary,
)


def _validate_history(
    history: list[Mapping[str, Any]],
) -> None:
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    for item in history:
        if not isinstance(item, Mapping):
            raise TypeError("Each history item must be a mapping.")

        value = item.get("alert_count")

        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                "Each history item must contain an integer alert_count."
            )

        if value < 0:
            raise ValueError("alert_count must not be negative.")


def _validate_bounds(
    lower_bound: int,
    upper_bound: int,
) -> None:
    for name, value in (
        ("lower_bound", lower_bound),
        ("upper_bound", upper_bound),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer.")

        if value < 0:
            raise ValueError(f"{name} must not be negative.")

    if lower_bound > upper_bound:
        raise ValueError(
            "lower_bound must not be greater than upper_bound."
        )


def build_quality_score_summary_detail(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a detailed quality-score summary."""

    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    summary = build_quality_score_summary(
        history,
        lower_bound,
        upper_bound,
    )

    details = []

    for index, item in enumerate(history):
        alert_count = item["alert_count"]
        is_stable = lower_bound <= alert_count <= upper_bound

        details.append(
            {
                "snapshot_index": index,
                "alert_count": alert_count,
                "is_stable": is_stable,
            }
        )

    stable_count = sum(
        1
        for detail in details
        if detail["is_stable"]
    )

    unstable_count = len(details) - stable_count

    return {
        "snapshot_count": summary["snapshot_count"],
        "lower_bound": summary["lower_bound"],
        "upper_bound": summary["upper_bound"],
        "stability_ratio": summary["stability_ratio"],
        "consistency_ratio": summary["consistency_ratio"],
        "quality_score": summary["quality_score"],
        "quality_percentage": summary["quality_percentage"],
        "stable_count": stable_count,
        "unstable_count": unstable_count,
        "details": details,
    }
