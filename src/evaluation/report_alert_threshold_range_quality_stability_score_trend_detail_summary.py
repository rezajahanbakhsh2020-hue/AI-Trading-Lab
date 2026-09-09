from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_quality_stability_score_trend_detail import (
    build_quality_stability_score_trend_detail,
)


def _validate_history(history: list[Mapping[str, Any]]) -> None:
    if not isinstance(history, list):
        raise TypeError("history must be a list")

    for item in history:
        if not isinstance(item, Mapping):
            raise TypeError("each history item must be a mapping")

        alert_count = item.get("alert_count")

        if isinstance(alert_count, bool) or not isinstance(alert_count, int):
            raise TypeError("alert_count must be an integer")

        if alert_count < 0:
            raise ValueError("alert_count must be non-negative")


def _validate_bounds(lower_bound: int, upper_bound: int) -> None:
    if isinstance(lower_bound, bool) or not isinstance(lower_bound, int):
        raise TypeError("lower_bound must be an integer")

    if isinstance(upper_bound, bool) or not isinstance(upper_bound, int):
        raise TypeError("upper_bound must be an integer")

    if lower_bound < 0 or upper_bound < 0:
        raise ValueError("bounds must be non-negative")

    if lower_bound > upper_bound:
        raise ValueError(
            "lower_bound must be less than or equal to upper_bound"
        )


def build_quality_stability_score_trend_detail_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a report-ready detailed trend summary."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    detail = build_quality_stability_score_trend_detail(
        history,
        lower_bound,
        upper_bound,
    )

    return {
        "snapshot_count": detail["snapshot_count"],
        "lower_bound": detail["lower_bound"],
        "upper_bound": detail["upper_bound"],
        "quality_series": detail["quality_series"],
        "stability_series": detail["stability_series"],
        "score_series": detail["score_series"],
        "details": detail["details"],
        "quality_trend": detail["quality_trend"],
        "stability_trend": detail["stability_trend"],
        "score_trend": detail["score_trend"],
        "quality_change": detail["quality_change"],
        "stability_change": detail["stability_change"],
        "score_change": detail["score_change"],
        "latest_detail": (
            detail["details"][-1] if detail["details"] else None
        ),
    }
