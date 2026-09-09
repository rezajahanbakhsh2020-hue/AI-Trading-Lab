from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_summary_detail import (
    build_quality_score_summary_detail,
)


def _trend_direction(
    values: list[float],
    tolerance: float = 1e-12,
) -> str:
    if len(values) < 2:
        return "flat"

    change = values[-1] - values[0]

    if change > tolerance:
        return "up"

    if change < -tolerance:
        return "down"

    return "flat"


def build_quality_score_trend(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a trend report for quality-score snapshots."""

    summary = build_quality_score_summary_detail(
        history,
        lower_bound,
        upper_bound,
    )

    quality_series: list[float] = []
    stability_series: list[float] = []
    consistency_series: list[float] = []

    stable_count = 0

    for index, detail in enumerate(summary["details"]):
        if detail["is_stable"]:
            stable_count += 1

        snapshot_count = index + 1

        stability_ratio = stable_count / snapshot_count

        if index == 0:
            consistency_ratio = 1.0
        else:
            previous = summary["details"][index - 1]
            consistency_ratio = (
                1.0
                if detail["is_stable"] == previous["is_stable"]
                else 0.0
            )

        quality_score = (
            stability_ratio * consistency_ratio
        )

        quality_series.append(quality_score)
        stability_series.append(stability_ratio)
        consistency_series.append(consistency_ratio)

    return {
        "snapshot_count": summary["snapshot_count"],
        "lower_bound": summary["lower_bound"],
        "upper_bound": summary["upper_bound"],
        "quality_series": quality_series,
        "stability_series": stability_series,
        "consistency_series": consistency_series,
        "quality_trend": _trend_direction(quality_series),
        "stability_trend": _trend_direction(stability_series),
        "consistency_trend": _trend_direction(consistency_series),
        "quality_change": (
            quality_series[-1] - quality_series[0]
            if quality_series
            else 0.0
        ),
        "stability_change": (
            stability_series[-1] - stability_series[0]
            if stability_series
            else 0.0
        ),
        "consistency_change": (
            consistency_series[-1] - consistency_series[0]
            if consistency_series
            else 0.0
        ),
    }
