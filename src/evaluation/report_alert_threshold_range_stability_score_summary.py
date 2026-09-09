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


def calculate_stability_score(
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


def calculate_instability_score(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Calculate the fraction of observations outside the target range."""
    return 1.0 - calculate_stability_score(
        history,
        lower_bound,
        upper_bound,
    )


def calculate_stability_percentage(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Return the stability score as a percentage."""
    return (
        calculate_stability_score(
            history,
            lower_bound,
            upper_bound,
        )
        * 100.0
    )


def calculate_instability_percentage(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Return the instability score as a percentage."""
    return (
        calculate_instability_score(
            history,
            lower_bound,
            upper_bound,
        )
        * 100.0
    )


def build_stability_score_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a complete stability-score summary."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    stability_score = calculate_stability_score(
        history,
        lower_bound,
        upper_bound,
    )

    instability_score = 1.0 - stability_score

    return {
        "snapshot_count": len(history),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "stability_score": stability_score,
        "instability_score": instability_score,
        "stability_percentage": stability_score * 100.0,
        "instability_percentage": instability_score * 100.0,
    }
