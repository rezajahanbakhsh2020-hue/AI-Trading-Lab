from __future__ import annotations

from typing import Any, Mapping


def _validate_score(name: str, value: float) -> float:
    score = float(value)

    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")

    return score


def build_production_runtime_gate(
    readiness: Mapping[str, Any],
    *,
    minimum_readiness_score: float = 0.70,
) -> dict[str, Any]:
    """
    Build the final gate that controls whether the live runtime
    may be entered.

    The gate requires:
    1. production readiness to be READY
    2. a selected strategy
    3. a valid strategy stability score
    4. a valid portfolio stability score
    5. the combined readiness score to meet the threshold
    """
    if not isinstance(readiness, Mapping):
        raise TypeError("readiness must be a mapping.")

    minimum_readiness_score = _validate_score(
        "minimum_readiness_score",
        minimum_readiness_score,
    )

    strategy = readiness.get("strategy")
    strategy_score = readiness.get(
        "strategy_stability_score"
    )
    portfolio_score = readiness.get(
        "portfolio_stability_score"
    )
    readiness_score = readiness.get(
        "readiness_score"
    )

    failed_gates = list(
        readiness.get("failed_gates", [])
    )

    runtime_blockers: list[str] = []

    if not isinstance(strategy, str) or not strategy.strip():
        runtime_blockers.append("strategy_selection")

    try:
        strategy_score = _validate_score(
            "strategy_stability_score",
            strategy_score,
        )
    except (TypeError, ValueError):
        runtime_blockers.append("strategy_stability")

    try:
        portfolio_score = _validate_score(
            "portfolio_stability_score",
            portfolio_score,
        )
    except (TypeError, ValueError):
        runtime_blockers.append("portfolio_stability")

    try:
        readiness_score = _validate_score(
            "readiness_score",
            readiness_score,
        )
    except (TypeError, ValueError):
        runtime_blockers.append("combined_readiness")

    if (
        readiness_score is not None
        and readiness_score < minimum_readiness_score
    ):
        if "combined_readiness" not in runtime_blockers:
            runtime_blockers.append(
                "combined_readiness"
            )

    if readiness.get("status") != "READY":
        if not failed_gates:
            failed_gates.append("production_readiness")

    all_blockers = []

    for blocker in failed_gates + runtime_blockers:
        if blocker not in all_blockers:
            all_blockers.append(blocker)

    runtime_ready = len(all_blockers) == 0

    return {
        "runtime_ready": runtime_ready,
        "status": (
            "READY"
            if runtime_ready
            else "BLOCKED"
        ),
        "strategy": strategy,
        "strategy_stability_score": strategy_score,
        "portfolio_stability_score": portfolio_score,
        "readiness_score": readiness_score,
        "minimum_readiness_score": (
            minimum_readiness_score
        ),
        "failed_gates": all_blockers,
    }


def is_production_runtime_ready(
    readiness: Mapping[str, Any],
    *,
    minimum_readiness_score: float = 0.70,
) -> bool:
    """
    Return only whether the live runtime is allowed to start.
    """
    result = build_production_runtime_gate(
        readiness,
        minimum_readiness_score=minimum_readiness_score,
    )

    return bool(result["runtime_ready"])


def production_runtime_gate_message(
    readiness: Mapping[str, Any],
    *,
    minimum_readiness_score: float = 0.70,
) -> str:
    """
    Build a human-readable runtime gate message.
    """
    result = build_production_runtime_gate(
        readiness,
        minimum_readiness_score=minimum_readiness_score,
    )

    strategy = result.get("strategy")

    if result["runtime_ready"]:
        return (
            "RUNTIME READY: "
            f"{strategy} passed the production runtime gate."
        )

    blockers = result.get("failed_gates", [])

    return (
        "RUNTIME BLOCKED: "
        f"{strategy or 'no strategy selected'}; "
        f"failed gates: {', '.join(blockers)}."
    )
