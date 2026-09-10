from __future__ import annotations

from typing import Any, Mapping


REQUIRED_FIELDS = (
    "strategy",
    "stability_score",
    "total_return",
    "max_drawdown",
    "sharpe_ratio",
)


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_production_result(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")

    missing = [
        field
        for field in REQUIRED_FIELDS
        if field not in result
    ]

    if missing:
        raise ValueError(
            f"missing production fields: {', '.join(missing)}"
        )

    strategy = result.get("strategy")

    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError(
            "strategy must be a non-empty string"
        )

    stability_score = _to_float(
        result.get("stability_score")
    )
    total_return = _to_float(
        result.get("total_return")
    )
    max_drawdown = _to_float(
        result.get("max_drawdown")
    )
    sharpe_ratio = _to_float(
        result.get("sharpe_ratio")
    )

    numeric_fields = {
        "stability_score": stability_score,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
    }

    invalid = [
        name
        for name, value in numeric_fields.items()
        if value is None
    ]

    if invalid:
        raise ValueError(
            "invalid production metrics: "
            + ", ".join(invalid)
        )

    return {
        "valid": True,
        "strategy": strategy,
        "stability_score": stability_score,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
    }


def build_production_guard(
    result: Mapping[str, Any],
    *,
    minimum_stability_score: float = 0.0,
) -> dict[str, Any]:
    validated = validate_production_result(result)

    threshold = float(
        minimum_stability_score
    )

    stability_score = float(
        validated["stability_score"]
    )

    passed = (
        stability_score >= threshold
    )

    return {
        **validated,
        "minimum_stability_score": threshold,
        "stability_gate": passed,
        "status": (
            "READY"
            if passed
            else "BLOCKED"
        ),
    }


def is_production_ready(
    result: Mapping[str, Any],
    *,
    minimum_stability_score: float = 0.0,
) -> bool:
    guard = build_production_guard(
        result,
        minimum_stability_score=minimum_stability_score,
    )

    return bool(
        guard["stability_gate"]
    )


def production_guard_message(
    result: Mapping[str, Any],
    *,
    minimum_stability_score: float = 0.0,
) -> str:
    guard = build_production_guard(
        result,
        minimum_stability_score=minimum_stability_score,
    )

    if guard["stability_gate"]:
        return (
            "PRODUCTION READY: "
            f"{guard['strategy']} passed the "
            "stability gate."
        )

    return (
        "PRODUCTION BLOCKED: "
        f"{guard['strategy']} did not pass the "
        "stability gate."
    )
