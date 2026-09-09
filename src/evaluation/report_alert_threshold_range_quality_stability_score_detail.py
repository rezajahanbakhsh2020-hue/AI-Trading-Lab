"""Detailed quality/stability score analysis for threshold-range alert histories."""

from __future__ import annotations

from typing import Mapping, Sequence


def _validate_history(history: Sequence[Mapping[str, object]]) -> None:
    if not isinstance(history, list):
        raise TypeError("history must be a list")

    for item in history:
        if not isinstance(item, Mapping):
            raise TypeError("each history item must be a mapping")

        value = item.get("alert_count")

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("alert_count must be an integer")

        if value < 0:
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


def _is_in_range(
    value: int,
    lower_bound: int,
    upper_bound: int,
) -> bool:
    return lower_bound <= value <= upper_bound


def calculate_quality_ratio(
    history: list[Mapping[str, object]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Return the fraction of observations inside the configured range."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    if not history:
        return 0.0

    in_range_count = sum(
        _is_in_range(
            item["alert_count"],
            lower_bound,
            upper_bound,
        )
        for item in history
    )

    return in_range_count / len(history)


def calculate_stability_ratio(
    history: list[Mapping[str, object]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Return the fraction of adjacent observations preserving range state."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    if len(history) <= 1:
        return 1.0 if history else 0.0

    stable_transitions = 0

    for previous, current in zip(history, history[1:]):
        previous_state = _is_in_range(
            previous["alert_count"],
            lower_bound,
            upper_bound,
        )
        current_state = _is_in_range(
            current["alert_count"],
            lower_bound,
            upper_bound,
        )

        if previous_state == current_state:
            stable_transitions += 1

    return stable_transitions / (len(history) - 1)


def calculate_quality_stability_score(
    history: list[Mapping[str, object]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Return the average of quality and stability ratios."""
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

    return (quality_ratio + stability_ratio) / 2.0


def calculate_quality_stability_percentage(
    history: list[Mapping[str, object]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Return the quality/stability score as a percentage."""
    return (
        calculate_quality_stability_score(
            history,
            lower_bound,
            upper_bound,
        )
        * 100.0
    )


def build_quality_stability_score_detail(
    history: list[Mapping[str, object]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, object]:
    """Build a detailed quality/stability score result."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

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
    score = (quality_ratio + stability_ratio) / 2.0

    return {
        "snapshot_count": len(history),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "quality_ratio": quality_ratio,
        "stability_ratio": stability_ratio,
        "quality_stability_score": score,
        "quality_stability_percentage": score * 100.0,
    }
