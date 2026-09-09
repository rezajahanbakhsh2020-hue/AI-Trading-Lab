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
    """Return the fraction of observations inside the target range."""

    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    if not history:
        return 0.0

    in_range = sum(
        _in_range(
            item["alert_count"],
            lower_bound,
            upper_bound,
        )
        for item in history
    )

    return in_range / len(history)


def calculate_stability_ratio(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """
    Return the fraction of adjacent observations that preserve
    the same in-range/out-of-range state.

    A single observation has no transition to evaluate and is
    therefore considered fully stable.
    """

    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    if len(history) <= 1:
        return 1.0 if history else 0.0

    states = [
        _in_range(
            item["alert_count"],
            lower_bound,
            upper_bound,
        )
        for item in history
    ]

    stable_transitions = sum(
        previous == current
        for previous, current in zip(states, states[1:])
    )

    return stable_transitions / (len(states) - 1)


def calculate_quality_stability_score(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Combine range quality and temporal stability into one score."""

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
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Return the quality-stability score as a percentage."""

    return (
        calculate_quality_stability_score(
            history,
            lower_bound,
            upper_bound,
        )
        * 100.0
    )


def build_quality_stability_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a complete threshold-range quality-stability summary."""

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

    quality_stability_score = calculate_quality_stability_score(
        history,
        lower_bound,
        upper_bound,
    )

    return {
        "snapshot_count": len(history),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "quality_ratio": quality_ratio,
        "stability_ratio": stability_ratio,
        "quality_stability_score": quality_stability_score,
        "quality_stability_percentage": quality_stability_score * 100.0,
    }
