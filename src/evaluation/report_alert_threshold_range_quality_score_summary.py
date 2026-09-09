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


def calculate_quality_score(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Calculate the fraction of observations inside the inclusive range."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    if not history:
        return 0.0

    in_range = sum(
        lower_bound <= item["alert_count"] <= upper_bound
        for item in history
    )

    return in_range / len(history)


def calculate_quality_percentage(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Return the quality score as a percentage."""
    return calculate_quality_score(
        history,
        lower_bound,
        upper_bound,
    ) * 100.0


def calculate_out_of_range_ratio(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Return the fraction of observations outside the target range."""
    return 1.0 - calculate_quality_score(
        history,
        lower_bound,
        upper_bound,
    )


def calculate_in_range_count(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> int:
    """Return the number of observations inside the target range."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    return sum(
        lower_bound <= item["alert_count"] <= upper_bound
        for item in history
    )


def calculate_out_of_range_count(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> int:
    """Return the number of observations outside the target range."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    return len(history) - calculate_in_range_count(
        history,
        lower_bound,
        upper_bound,
    )


def build_quality_score_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a detailed quality-score summary."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    snapshot_count = len(history)
    in_range_count = calculate_in_range_count(
        history,
        lower_bound,
        upper_bound,
    )
    out_of_range_count = snapshot_count - in_range_count

    quality_score = (
        in_range_count / snapshot_count
        if snapshot_count
        else 0.0
    )

    return {
        "snapshot_count": snapshot_count,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "in_range_count": in_range_count,
        "out_of_range_count": out_of_range_count,
        "quality_score": quality_score,
        "quality_percentage": quality_score * 100.0,
        "out_of_range_ratio": 1.0 - quality_score,
    }
