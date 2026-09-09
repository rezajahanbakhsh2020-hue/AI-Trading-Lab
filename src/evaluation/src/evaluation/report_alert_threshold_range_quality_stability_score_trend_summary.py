from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_quality_stability_score_trend import (
    build_quality_stability_score_trend,
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


def build_quality_stability_score_trend_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a report-ready summary from the quality/stability trend."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    trend = build_quality_stability_score_trend(
        history,
        lower_bound,
        upper_bound,
    )

    quality_series = trend["quality_series"]
    stability_series = trend["stability_series"]
    score_series = trend["score_series"]

    return {
        "snapshot_count": trend["snapshot_count"],
        "lower_bound": trend["lower_bound"],
        "upper_bound": trend["upper_bound"],
        "quality_series": quality_series,
        "stability_series": stability_series,
        "score_series": score_series,
        "latest_quality": (
            quality_series[-1] if quality_series else 0.0
        ),
        "latest_stability": (
            stability_series[-1] if stability_series else 0.0
        ),
        "latest_score": (
            score_series[-1] if score_series else 0.0
        ),
        "quality_trend": trend["quality_trend"],
        "stability_trend": trend["stability_trend"],
        "score_trend": trend["score_trend"],
        "quality_change": trend["quality_change"],
        "stability_change": trend["stability_change"],
        "score_change": trend["score_change"],
    }
