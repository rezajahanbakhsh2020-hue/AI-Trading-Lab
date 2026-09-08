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


def _validate_threshold(threshold: int) -> None:
    if (
        not isinstance(threshold, int)
        or isinstance(threshold, bool)
    ):
        raise TypeError(
            "threshold must be an integer."
        )

    if threshold < 0:
        raise ValueError(
            "threshold must not be negative."
        )


def _validate_width(width: int) -> None:
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
    ):
        raise TypeError(
            "width must be an integer."
        )

    if width < 0:
        raise ValueError(
            "width must not be negative."
        )


def calculate_lower_zone(
    threshold: int,
    width: int,
) -> int:
    """Calculate the lower boundary of the threshold zone."""
    _validate_threshold(threshold)
    _validate_width(width)

    return max(
        0,
        threshold - width,
    )


def calculate_upper_zone(
    threshold: int,
    width: int,
) -> int:
    """Calculate the upper boundary of the threshold zone."""
    _validate_threshold(threshold)
    _validate_width(width)

    return threshold + width


def classify_alert_zone(
    alert_count: int,
    threshold: int,
    width: int,
) -> str:
    """Classify an alert count as below, inside, or above the zone."""
    if (
        not isinstance(alert_count, int)
        or isinstance(alert_count, bool)
    ):
        raise TypeError(
            "alert_count must be an integer."
        )

    if alert_count < 0:
        raise ValueError(
            "alert_count must not be negative."
        )

    lower_zone = calculate_lower_zone(
        threshold,
        width,
    )
    upper_zone = calculate_upper_zone(
        threshold,
        width,
    )

    if alert_count < lower_zone:
        return "below"

    if alert_count > upper_zone:
        return "above"

    return "inside"


def count_alert_zones(
    history: list[Mapping[str, Any]],
    threshold: int,
    width: int,
) -> dict[str, int]:
    """Count snapshots in each alert zone."""
    _validate_history(history)
    _validate_threshold(threshold)
    _validate_width(width)

    counts = {
        "below": 0,
        "inside": 0,
        "above": 0,
    }

    for item in history:
        zone = classify_alert_zone(
            item["alert_count"],
            threshold,
            width,
        )
        counts[zone] += 1

    return counts


def find_zone_positions(
    history: list[Mapping[str, Any]],
    threshold: int,
    width: int,
) -> dict[str, list[int]]:
    """Return positions grouped by alert zone."""
    _validate_history(history)
    _validate_threshold(threshold)
    _validate_width(width)

    positions = {
        "below": [],
        "inside": [],
        "above": [],
    }

    for index, item in enumerate(history):
        zone = classify_alert_zone(
            item["alert_count"],
            threshold,
            width,
        )
        positions[zone].append(index)

    return positions


def build_threshold_zone_summary(
    history: list[Mapping[str, Any]],
    threshold: int,
    width: int,
) -> dict[str, Any]:
    """Build a complete threshold-zone summary."""
    _validate_history(history)
    _validate_threshold(threshold)
    _validate_width(width)

    return {
        "snapshot_count": len(history),
        "threshold": threshold,
        "width": width,
        "lower_zone": calculate_lower_zone(
            threshold,
            width,
        ),
        "upper_zone": calculate_upper_zone(
            threshold,
            width,
        ),
        "zone_counts": count_alert_zones(
            history,
            threshold,
            width,
        ),
        "zone_positions": find_zone_positions(
            history,
            threshold,
            width,
        ),
    }
