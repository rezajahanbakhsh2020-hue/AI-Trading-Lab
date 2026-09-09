from __future__ import annotations

from typing import Any, Mapping


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
    if not isinstance(lower_bound, int) or isinstance(lower_bound, bool):
        raise TypeError("lower_bound must be an integer.")

    if not isinstance(upper_bound, int) or isinstance(upper_bound, bool):
        raise TypeError("upper_bound must be an integer.")

    if lower_bound < 0:
        raise ValueError("lower_bound must not be negative.")

    if upper_bound < 0:
        raise ValueError("upper_bound must not be negative.")

    if lower_bound > upper_bound:
        raise ValueError(
            "lower_bound must not be greater than upper_bound."
        )


def calculate_in_range_score(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Return the fraction of observations inside the inclusive range."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    if not history:
        return 0.0

    count = sum(
        lower_bound <= item["alert_count"] <= upper_bound
        for item in history
    )

    return count / len(history)


def calculate_boundary_penalty(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Return the fraction of observations outside the target range."""
    return 1.0 - calculate_in_range_score(
        history,
        lower_bound,
        upper_bound,
    )


def calculate_quality_score(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """
    Calculate a normalized quality score in the [0, 1] interval.

    Observations inside the inclusive target range receive full quality.
    Observations outside the range reduce the score proportionally.
    """
    return calculate_in_range_score(
        history,
        lower_bound,
        upper_bound,
    )


def calculate_quality_percentage(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Return the normalized quality score as a percentage."""
    return calculate_quality_score(
        history,
        lower_bound,
        upper_bound,
    ) * 100.0


def build_quality_score_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a normalized quality-score summary."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    quality_score = calculate_quality_score(
        history,
        lower_bound,
        upper_bound,
    )

    return {
        "snapshot_count": len(history),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "quality_score": quality_score,
        "quality_percentage": quality_score * 100.0,
        "boundary_penalty": 1.0 - quality_score,
    }
