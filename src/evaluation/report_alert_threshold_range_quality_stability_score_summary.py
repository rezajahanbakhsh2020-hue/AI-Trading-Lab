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


def _is_in_range(
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
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    if not history:
        return 0.0

    in_range = sum(
        _is_in_range(
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
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    if not history:
        return 0.0

    if len(history) == 1:
        return 1.0

    states = [
        _is_in_range(
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


def calculate_score_percentage(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    return calculate_quality_stability_score(
        history,
        lower_bound,
        upper_bound,
    ) * 100.0


def classify_score(
    score: float,
    stable_threshold: float = 0.80,
    acceptable_threshold: float = 0.60,
) -> str:
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise TypeError("score must be numeric.")

    if not isinstance(
        stable_threshold,
        (int, float),
    ) or isinstance(stable_threshold, bool):
        raise TypeError("stable_threshold must be numeric.")

    if not isinstance(
        acceptable_threshold,
        (int, float),
    ) or isinstance(acceptable_threshold, bool):
        raise TypeError("acceptable_threshold must be numeric.")

    if not 0.0 <= acceptable_threshold <= 1.0:
        raise ValueError(
            "acceptable_threshold must be between 0 and 1."
        )

    if not 0.0 <= stable_threshold <= 1.0:
        raise ValueError(
            "stable_threshold must be between 0 and 1."
        )

    if acceptable_threshold > stable_threshold:
        raise ValueError(
            "acceptable_threshold must not exceed "
            "stable_threshold."
        )

    if score >= stable_threshold:
        return "stable"

    if score >= acceptable_threshold:
        return "acceptable"

    return "unstable"


def build_quality_stability_score_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
    stable_threshold: float = 0.80,
    acceptable_threshold: float = 0.60,
) -> dict[str, Any]:
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

    score = calculate_quality_stability_score(
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
        "quality_stability_score": score,
        "quality_stability_percentage": score * 100.0,
        "classification": classify_score(
            score,
            stable_threshold,
            acceptable_threshold,
        ),
    }
