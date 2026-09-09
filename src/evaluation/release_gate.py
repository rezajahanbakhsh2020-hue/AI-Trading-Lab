"""Final release-readiness gate for AI-Trading-Lab."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


REQUIRED_WALK_FORWARD_KEYS = (
    "strategy",
    "stability_score",
    "total_return",
    "max_drawdown",
    "sharpe_ratio",
)

REQUIRED_PRODUCTION_KEYS = (
    "strategy",
    "stability_score",
    "observations",
    "total_return",
    "max_drawdown",
    "sharpe_ratio",
)


def _validate_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _check_required_keys(
    value: Mapping[str, Any],
    required_keys: tuple[str, ...],
    name: str,
) -> dict[str, bool]:
    return {
        key: key in value
        for key in required_keys
    }


def _all_finite_numeric(
    value: Mapping[str, Any],
    keys: tuple[str, ...],
) -> bool:
    for key in keys:
        if key not in value:
            return False

        try:
            number = float(value[key])
        except (TypeError, ValueError):
            return False

        if number != number:
            return False

        if number in (float("inf"), float("-inf")):
            return False

    return True


def build_release_gate(
    *,
    walk_forward: Mapping[str, Any],
    production: Mapping[str, Any],
    robustness: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the minimum evidence required for a release decision."""

    walk_forward = _validate_mapping(walk_forward, "walk_forward")
    production = _validate_mapping(production, "production")
    robustness = _validate_mapping(robustness, "robustness")

    walk_forward_keys = _check_required_keys(
        walk_forward,
        REQUIRED_WALK_FORWARD_KEYS,
        "walk_forward",
    )

    production_keys = _check_required_keys(
        production,
        REQUIRED_PRODUCTION_KEYS,
        "production",
    )

    walk_forward_complete = all(walk_forward_keys.values())
    production_complete = all(production_keys.values())

    walk_forward_numeric = _all_finite_numeric(
        walk_forward,
        (
            "stability_score",
            "total_return",
            "max_drawdown",
            "sharpe_ratio",
        ),
    )

    production_numeric = _all_finite_numeric(
        production,
        (
            "stability_score",
            "observations",
            "total_return",
            "max_drawdown",
            "sharpe_ratio",
        ),
    )

    robustness_passed = bool(
        robustness.get("robust", False)
    )

    strategy_consistent = (
        bool(walk_forward.get("strategy"))
        and bool(production.get("strategy"))
        and walk_forward.get("strategy") == production.get("strategy")
    )

    observations_valid = (
        "observations" in production
        and isinstance(
            production["observations"],
            (int, float),
        )
        and not isinstance(production["observations"], bool)
        and float(production["observations"]) > 0
    )

    checks = {
        "walk_forward_complete": walk_forward_complete,
        "walk_forward_numeric": walk_forward_numeric,
        "production_complete": production_complete,
        "production_numeric": production_numeric,
        "robustness_passed": robustness_passed,
        "strategy_consistent": strategy_consistent,
        "production_observations_valid": observations_valid,
    }

    ready = all(checks.values())

    return {
        "ready": ready,
        "status": "RELEASE_READY" if ready else "RELEASE_BLOCKED",
        "strategy": production.get("strategy"),
        "checks": checks,
        "walk_forward_required_keys": walk_forward_keys,
        "production_required_keys": production_keys,
    }


def assert_release_ready(
    gate: Mapping[str, Any],
) -> None:
    """Raise an error when the final release gate is not passed."""
    gate = _validate_mapping(gate, "gate")

    if not bool(gate.get("ready", False)):
        raise RuntimeError(
            f"release gate failed: {gate.get('status', 'UNKNOWN')}"
        )
