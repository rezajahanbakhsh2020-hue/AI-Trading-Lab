from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_quality_score import (
    calculate_quality_score,
)
from src.evaluation.report_alert_threshold_range_stability import (
    calculate_range_stability,
)


def calculate_stability_quality_score(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Calculate the combined range-stability and quality score."""

    stability_score = calculate_range_stability(
        history,
        lower_bound,
        upper_bound,
    )
    quality_score = calculate_quality_score(
        history,
        lower_bound,
        upper_bound,
    )

    return (stability_score + quality_score) / 2.0


def calculate_stability_quality_percentage(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Return the combined stability-quality score as a percentage."""

    return (
        calculate_stability_quality_score(
            history,
            lower_bound,
            upper_bound,
        )
        * 100.0
    )


def build_stability_quality_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a summary combining range stability and quality."""

    stability_score = calculate_range_stability(
        history,
        lower_bound,
        upper_bound,
    )
    quality_score = calculate_quality_score(
        history,
        lower_bound,
        upper_bound,
    )
    combined_score = (
        stability_score + quality_score
    ) / 2.0

    return {
        "snapshot_count": len(history),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "stability_score": stability_score,
        "quality_score": quality_score,
        "stability_quality_score": combined_score,
        "stability_percentage": stability_score * 100.0,
        "quality_percentage": quality_score * 100.0,
        "stability_quality_percentage": combined_score * 100.0,
        "stability_quality_gap": abs(
            stability_score - quality_score
        ),
    }
