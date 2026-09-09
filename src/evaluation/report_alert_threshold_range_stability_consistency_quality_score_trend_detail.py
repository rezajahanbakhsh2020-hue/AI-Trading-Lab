from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend import (
    build_quality_score_trend,
)


def build_quality_score_trend_detail(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a detailed per-snapshot quality-score trend report."""

    trend = build_quality_score_trend(
        history,
        lower_bound,
        upper_bound,
    )

    details: list[dict[str, Any]] = []

    for index, item in enumerate(history):
        quality_score = trend["quality_series"][index]
        stability_score = trend["stability_series"][index]
        consistency_score = trend["consistency_series"][index]

        if index == 0:
            quality_change = 0.0
            stability_change = 0.0
            consistency_change = 0.0
        else:
            quality_change = (
                quality_score
                - trend["quality_series"][index - 1]
            )
            stability_change = (
                stability_score
                - trend["stability_series"][index - 1]
            )
            consistency_change = (
                consistency_score
                - trend["consistency_series"][index - 1]
            )

        details.append(
            {
                "snapshot_index": index,
                "alert_count": item["alert_count"],
                "quality_score": quality_score,
                "stability_score": stability_score,
                "consistency_score": consistency_score,
                "quality_change": quality_change,
                "stability_change": stability_change,
                "consistency_change": consistency_change,
            }
        )

    return {
        "snapshot_count": trend["snapshot_count"],
        "lower_bound": trend["lower_bound"],
        "upper_bound": trend["upper_bound"],
        "quality_trend": trend["quality_trend"],
        "stability_trend": trend["stability_trend"],
        "consistency_trend": trend["consistency_trend"],
        "quality_change": trend["quality_change"],
        "stability_change": trend["stability_change"],
        "consistency_change": trend["consistency_change"],
        "details": details,
    }
