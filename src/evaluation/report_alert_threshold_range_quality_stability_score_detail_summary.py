"""Summary classification for detailed threshold-range quality/stability scores."""

from __future__ import annotations

from typing import Mapping


def _validate_history(history: list[Mapping[str, object]]) -> None:
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


def _validate_threshold(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


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

    if not history:
        return 0.0

    if len(history) == 1:
        return 1.0

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
    """Return the average quality/stability score."""
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


def classify_score(
    score: float,
    stable_threshold: float = 0.80,
    acceptable_threshold: float = 0.60,
) -> str:
    """Classify a quality/stability score."""
    _validate_threshold("score", score)
    _validate_threshold("stable_threshold", stable_threshold)
    _validate_threshold("acceptable_threshold", acceptable_threshold)

    if acceptable_threshold > stable_threshold:
        raise ValueError(
            "acceptable_threshold must be less than or equal to "
            "stable_threshold"
        )

    if score >= stable_threshold:
        return "stable"

    if score >= acceptable_threshold:
        return "acceptable"

    return "unstable"


def build_quality_stability_score_detail_summary(
    history: list[Mapping[str, object]],
    lower_bound: int,
    upper_bound: int,
    stable_threshold: float = 0.80,
    acceptable_threshold: float = 0.60,
) -> dict[str, object]:
    """Build a detailed score summary including classification."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    score = calculate_quality_stability_score(
        history,
        lower_bound,
        upper_bound,
    )

    classification = classify_score(
        score,
        stable_threshold=stable_threshold,
        acceptable_threshold=acceptable_threshold,
    )

    return {
        "snapshot_count": len(history),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "quality_ratio": calculate_quality_ratio(
            history,
            lower_bound,
            upper_bound,
        ),
        "stability_ratio": calculate_stability_ratio(
            history,
            lower_bound,
            upper_bound,
        ),
        "quality_stability_score": score,
        "quality_stability_percentage": score * 100.0,
        "stable_threshold": stable_threshold,
        "acceptable_threshold": acceptable_threshold,
        "classification": classification,
    }
