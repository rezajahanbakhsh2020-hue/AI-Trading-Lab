from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend import (
    build_quality_score_trend,
)


def build_quality_score_trend_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a compact summary of quality-score trends."""

    trend = build_quality_score_trend(
        history,
        lower_bound,
        upper_bound,
    )

    quality_series = trend["quality_series"]
    stability_series = trend["stability_series"]
    consistency_series = trend["consistency_series"]

    return {
        "snapshot_count": trend["snapshot_count"],
        "lower_bound": trend["lower_bound"],
        "upper_bound": trend["upper_bound"],
        "quality_score": (
            quality_series[-1]
            if quality_series
            else 0.0
        ),
        "stability_score": (
            stability_series[-1]
            if stability_series
            else 0.0
        ),
        "consistency_score": (
            consistency_series[-1]
            if consistency_series
            else 0.0
        ),
        "quality_trend": trend["quality_trend"],
        "stability_trend": trend["stability_trend"],
        "consistency_trend": trend["consistency_trend"],
        "quality_change": trend["quality_change"],
        "stability_change": trend["stability_change"],
        "consistency_change": trend["consistency_change"],
    }
