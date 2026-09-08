from __future__ import annotations

from typing import Any, Mapping


def _validate_history(
    history: list[Mapping[str, Any]],
) -> None:
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    for item in history:
        if not isinstance(item, Mapping):
            raise TypeError(
                "Each history item must be a mapping."
            )

        value = item.get("alert_count")

        if (
            not isinstance(value, int)
            or isinstance(value, bool)
        ):
            raise ValueError(
                "Each history item must contain an integer alert_count."
            )

        if value < 0:
            raise ValueError(
                "alert_count must not be negative."
            )


def _validate_bounds(
    lower_bound: int,
    upper_bound: int,
) -> None:
    for name, value in (
        ("lower_bound", lower_bound),
        ("upper_bound", upper_bound),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
        ):
            raise TypeError(
                f"{name} must be an integer."
            )

        if value < 0:
            raise ValueError(
                f"{name} must not be negative."
            )

    if lower_bound > upper_bound:
        raise ValueError(
            "lower_bound must not be greater than upper_bound."
        )


def calculate_consistency_score(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Calculate the fraction of observations inside the target range."""
    _validate_history(history)
    _validate_bounds(
        lower_bound,
        upper_bound,
    )

    if not history:
        return 0.0

    consistent_count = sum(
        lower_bound <= item["alert_count"] <= upper_bound
        for item in history
    )

    return consistent_count / len(history)


def calculate_deviation_score(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Calculate the fraction of observations outside the target range."""
    return 1.0 - calculate_consistency_score(
        history,
        lower_bound,
        upper_bound,
    )


def calculate_consistency_percentage(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Return the consistency score as a percentage."""
    return (
        calculate_consistency_score(
            history,
            lower_bound,
            upper_bound,
        )
        * 100.0
    )


def calculate_deviation_percentage(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Return the deviation score as a percentage."""
    return (
        calculate_deviation_score(
            history,
            lower_bound,
            upper_bound,
        )
        * 100.0
    )


def build_consistency_score_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a complete consistency-score summary."""
    _validate_history(history)
    _validate_bounds(
        lower_bound,
        upper_bound,
    )

    score = calculate_consistency_score(
        history,
        lower_bound,
        upper_bound,
    )
    deviation = 1.0 - score

    return {
        "snapshot_count": len(history),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "consistency_score": score,
        "deviation_score": deviation,
        "consistency_percentage": score * 100.0,
        "deviation_percentage": deviation * 100.0,
    }
