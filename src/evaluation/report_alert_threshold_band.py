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


def _validate_band(band: int) -> None:
    if (
        not isinstance(band, int)
        or isinstance(band, bool)
    ):
        raise TypeError(
            "band must be an integer."
        )

    if band < 0:
        raise ValueError(
            "band must not be negative."
        )


def calculate_lower_bound(
    threshold: int,
    band: int,
) -> int:
    """Calculate the lower boundary of the threshold band."""
    _validate_threshold(threshold)
    _validate_band(band)

    return max(
        0,
        threshold - band,
    )


def calculate_upper_bound(
    threshold: int,
    band: int,
) -> int:
    """Calculate the upper boundary of the threshold band."""
    _validate_threshold(threshold)
    _validate_band(band)

    return threshold + band


def is_within_band(
    alert_count: int,
    threshold: int,
    band: int,
) -> bool:
    """Return whether alert count lies inside the threshold band."""
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

    lower_bound = calculate_lower_bound(
        threshold,
        band,
    )
    upper_bound = calculate_upper_bound(
        threshold,
        band,
    )

    return lower_bound <= alert_count <= upper_bound


def count_within_band(
    history: list[Mapping[str, Any]],
    threshold: int,
    band: int,
) -> int:
    """Count snapshots whose alert count lies inside the band."""
    _validate_history(history)
    _validate_threshold(threshold)
    _validate_band(band)

    return sum(
        is_within_band(
            item["alert_count"],
            threshold,
            band,
        )
        for item in history
    )


def calculate_band_ratio(
    history: list[Mapping[str, Any]],
    threshold: int,
    band: int,
) -> float:
    """Calculate the proportion of snapshots inside the band."""
    _validate_history(history)
    _validate_threshold(threshold)
    _validate_band(band)

    if not history:
        return 0.0

    return count_within_band(
        history,
        threshold,
        band,
    ) / len(history)


def find_within_band_positions(
    history: list[Mapping[str, Any]],
    threshold: int,
    band: int,
) -> list[int]:
    """Return positions whose alert counts lie inside the band."""
    _validate_history(history)
    _validate_threshold(threshold)
    _validate_band(band)

    return [
        index
        for index, item in enumerate(history)
        if is_within_band(
            item["alert_count"],
            threshold,
            band,
        )
    ]


def find_outside_band_positions(
    history: list[Mapping[str, Any]],
    threshold: int,
    band: int,
) -> list[int]:
    """Return positions whose alert counts lie outside the band."""
    _validate_history(history)
    _validate_threshold(threshold)
    _validate_band(band)

    return [
        index
        for index, item in enumerate(history)
        if not is_within_band(
            item["alert_count"],
            threshold,
            band,
        )
    ]


def build_threshold_band_summary(
    history: list[Mapping[str, Any]],
    threshold: int,
    band: int,
) -> dict[str, Any]:
    """Build a complete threshold-band summary."""
    _validate_history(history)
    _validate_threshold(threshold)
    _validate_band(band)

    return {
        "snapshot_count": len(history),
        "threshold": threshold,
        "band": band,
        "lower_bound": calculate_lower_bound(
            threshold,
            band,
        ),
        "upper_bound": calculate_upper_bound(
            threshold,
            band,
        ),
        "within_band_count": count_within_band(
            history,
            threshold,
            band,
        ),
        "band_ratio": calculate_band_ratio(
            history,
            threshold,
            band,
        ),
        "within_band_positions": find_within_band_positions(
            history,
            threshold,
            band,
        ),
        "outside_band_positions": find_outside_band_positions(
            history,
            threshold,
            band,
        ),
    }
