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
    for name, value in (
        ("lower_bound", lower_bound),
        ("upper_bound", upper_bound),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer.")

        if value < 0:
            raise ValueError(f"{name} must not be negative.")

    if lower_bound > upper_bound:
        raise ValueError(
            "lower_bound must not be greater than upper_bound."
        )


def calculate_stability_ratio(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Calculate the fraction of observations inside the target range."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    if not history:
        return 0.0

    stable_count = sum(
        lower_bound <= item["alert_count"] <= upper_bound
        for item in history
    )

    return stable_count / len(history)


def calculate_consistency_ratio(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Calculate range membership as the consistency ratio."""
    return calculate_stability_ratio(
        history,
        lower_bound,
        upper_bound,
    )


def calculate_quality_score(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Calculate the combined stability-consistency quality score."""
    stability_ratio = calculate_stability_ratio(
        history,
        lower_bound,
        upper_bound,
    )

    consistency_ratio = calculate_consistency_ratio(
        history,
        lower_bound,
        upper_bound,
    )

    return (stability_ratio + consistency_ratio) / 2.0


def calculate_quality_percentage(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Return the quality score as a percentage."""
    return (
        calculate_quality_score(
            history,
            lower_bound,
            upper_bound,
        )
        * 100.0
    )


def build_quality_score_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a complete quality-score summary."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    stability_ratio = calculate_stability_ratio(
        history,
        lower_bound,
        upper_bound,
    )

    consistency_ratio = calculate_consistency_ratio(
        history,
        lower_bound,
        upper_bound,
    )

    quality_score = calculate_quality_score(
        history,
        lower_bound,
        upper_bound,
    )

    quality_percentage = calculate_quality_percentage(
        history,
        lower_bound,
        upper_bound,
    )

    return {
        "snapshot_count": len(history),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "stability_ratio": stability_ratio,
        "consistency_ratio": consistency_ratio,
        "quality_score": quality_score,
        "quality_percentage": quality_percentage,
    }
