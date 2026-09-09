from __future__ import annotations

from collections.abc import Mapping
from typing import Any


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
        raise ValueError("lower_bound must be less than or equal to upper_bound")


def _in_range(
    value: int,
    lower_bound: int,
    upper_bound: int,
) -> bool:
    return lower_bound <= value <= upper_bound


def calculate_quality_ratio(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Return the cumulative fraction of observations inside the target range."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    if not history:
        return 0.0

    in_range_count = sum(
        _in_range(
            item["alert_count"],
            lower_bound,
            upper_bound,
        )
        for item in history
    )

    return in_range_count / len(history)


def calculate_stability_ratio(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Return the fraction of adjacent observations preserving range state."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    if not history:
        return 0.0

    if len(history) == 1:
        return 1.0

    states = [
        _in_range(
            item["alert_count"],
            lower_bound,
            upper_bound,
        )
        for item in history
    ]

    stable_transitions = sum(
        states[index] == states[index - 1]
        for index in range(1, len(states))
    )

    return stable_transitions / (len(states) - 1)


def calculate_quality_stability_score(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Return the average of cumulative quality and stability ratios."""
    quality_ratio = calculate_quality_ratio(
        history,
        lower_bound,
        upper_bound,
    )
    stability_ratio = calculate_stability_ratio(
        history,
        lower_bound,
        upper_bound,
    )

    return (quality_ratio + stability_ratio) / 2


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


def build_quality_stability_score_trend(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """
    Build cumulative quality/stability/score series and classify their trends.

    Each point in the returned series represents the metrics calculated from
    the history available up to that point.
    """
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    quality_series: list[float] = []
    stability_series: list[float] = []
    score_series: list[float] = []

    for index in range(1, len(history) + 1):
        window = history[:index]

        quality_ratio = calculate_quality_ratio(
            window,
            lower_bound,
            upper_bound,
        )
        stability_ratio = calculate_stability_ratio(
            window,
            lower_bound,
            upper_bound,
        )
        score = (quality_ratio + stability_ratio) / 2

        quality_series.append(quality_ratio)
        stability_series.append(stability_ratio)
        score_series.append(score)

    return {
        "snapshot_count": len(history),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "quality_series": quality_series,
        "stability_series": stability_series,
        "score_series": score_series,
        "quality_trend": _trend_direction(quality_series),
        "stability_trend": _trend_direction(stability_series),
        "score_trend": _trend_direction(score_series),
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
        "score_change": (
            score_series[-1] - score_series[0]
            if score_series
            else 0.0
        ),
    }
