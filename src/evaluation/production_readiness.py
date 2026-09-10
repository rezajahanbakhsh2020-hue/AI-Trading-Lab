from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.production_selection import (
    select_production_strategy,
)
from src.evaluation.stable_strategy_readiness import (
    evaluate_strategy_readiness,
)


DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "walk_forward"
)


def _validate_score(name: str, value: float) -> float:
    value = float(value)

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")

    return value


def _selection_strategy(selection: dict[str, Any]) -> str | None:
    strategy = selection.get("strategy")

    if strategy is not None:
        return str(strategy)

    candidates = selection.get("candidates")

    if isinstance(candidates, list) and candidates:
        first = candidates[0]

        if isinstance(first, dict):
            candidate_strategy = (
                first.get("strategy")
                or first.get("name")
            )

            if candidate_strategy is not None:
                return str(candidate_strategy)

        if isinstance(first, str):
            return first

    return None


def _selection_stability_score(
    selection: dict[str, Any],
) -> float | None:
    score = selection.get("stability_score")

    if score is not None:
        return float(score)

    candidates = selection.get("candidates")

    if isinstance(candidates, list) and candidates:
        first = candidates[0]

        if isinstance(first, dict):
            score = (
                first.get("stability_score")
                or first.get("stability")
            )

            if score is not None:
                return float(score)

    return None


def _fallback_selection(
    results_dir: str | Path,
) -> dict[str, Any]:
    """
    Provide a minimal selection fallback for a directly supplied
    stability report.

    This keeps the readiness layer compatible with both the
    production-selection contract and a simple stability report.
    """
    path = Path(results_dir) / "stability_report.csv"

    if not path.exists():
        return {
            "strategy": None,
            "stability_score": None,
            "candidates": [],
        }

    try:
        import pandas as pd

        frame = pd.read_csv(path)
    except Exception:
        return {
            "strategy": None,
            "stability_score": None,
            "candidates": [],
        }

    if frame.empty or "strategy" not in frame.columns:
        return {
            "strategy": None,
            "stability_score": None,
            "candidates": [],
        }

    score_column = None

    for column in (
        "stability_score",
        "stability",
        "score",
    ):
        if column in frame.columns:
            score_column = column
            break

    if score_column is None:
        return {
            "strategy": None,
            "stability_score": None,
            "candidates": [],
        }

    candidates = []

    for _, row in frame.iterrows():
        try:
            score = float(row[score_column])
        except (TypeError, ValueError):
            continue

        candidates.append(
            {
                "strategy": str(row["strategy"]),
                "stability_score": score,
            }
        )

    candidates.sort(
        key=lambda item: item["stability_score"],
        reverse=True,
    )

    if not candidates:
        return {
            "strategy": None,
            "stability_score": None,
            "candidates": [],
        }

    best = candidates[0]

    return {
        "strategy": best["strategy"],
        "stability_score": best["stability_score"],
        "candidates": candidates,
    }


def _select_strategy(
    results_dir: str | Path,
) -> dict[str, Any]:
    """
    Use the existing production selector first.

    If the selector cannot resolve a strategy from a directly
    supplied stability report, fall back to that report without
    changing the existing selector.
    """
    selection = select_production_strategy(
        results_dir=results_dir,
    )

    if not isinstance(selection, dict):
        selection = {}

    if (
        _selection_strategy(selection) is not None
        and _selection_stability_score(selection) is not None
    ):
        return selection

    fallback = _fallback_selection(results_dir)

    if fallback["strategy"] is not None:
        return fallback

    return selection


def build_production_readiness(
    portfolio_stability_score: float,
    *,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    minimum_strategy_stability: float = 0.70,
    minimum_portfolio_stability: float = 0.70,
    minimum_readiness_score: float = 0.70,
    strategy_weight: float = 0.70,
    portfolio_weight: float = 0.30,
) -> dict[str, Any]:
    """
    Combine the selected production strategy with portfolio
    stability and produce a final readiness decision.
    """
    portfolio_stability_score = _validate_score(
        "portfolio_stability_score",
        portfolio_stability_score,
    )

    selection = _select_strategy(results_dir)

    strategy = _selection_strategy(selection)
    strategy_stability_score = _selection_stability_score(
        selection
    )

    if strategy is None:
        return {
            "strategy": None,
            "strategy_stability_score": None,
            "portfolio_stability_score": portfolio_stability_score,
            "readiness_score": 0.0,
            "strategy_gate": False,
            "portfolio_gate": (
                portfolio_stability_score
                >= minimum_portfolio_stability
            ),
            "readiness_gate": False,
            "ready": False,
            "status": "BLOCKED",
            "failed_gates": ["strategy_selection"],
            "selection": selection,
        }

    if strategy_stability_score is None:
        return {
            "strategy": strategy,
            "strategy_stability_score": None,
            "portfolio_stability_score": portfolio_stability_score,
            "readiness_score": 0.0,
            "strategy_gate": False,
            "portfolio_gate": (
                portfolio_stability_score
                >= minimum_portfolio_stability
            ),
            "readiness_gate": False,
            "ready": False,
            "status": "BLOCKED",
            "failed_gates": ["strategy_stability"],
            "selection": selection,
        }

    readiness = evaluate_strategy_readiness(
        strategy=strategy,
        strategy_stability_score=strategy_stability_score,
        portfolio_stability_score=portfolio_stability_score,
        minimum_strategy_stability=(
            minimum_strategy_stability
        ),
        minimum_portfolio_stability=(
            minimum_portfolio_stability
        ),
        minimum_readiness_score=minimum_readiness_score,
        strategy_weight=strategy_weight,
        portfolio_weight=portfolio_weight,
    )

    return {
        **readiness,
        "selection": selection,
    }


def is_production_ready(
    portfolio_stability_score: float,
    *,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    minimum_strategy_stability: float = 0.70,
    minimum_portfolio_stability: float = 0.70,
    minimum_readiness_score: float = 0.70,
    strategy_weight: float = 0.70,
    portfolio_weight: float = 0.30,
) -> bool:
    """
    Return only the final production readiness decision.
    """
    result = build_production_readiness(
        portfolio_stability_score=portfolio_stability_score,
        results_dir=results_dir,
        minimum_strategy_stability=minimum_strategy_stability,
        minimum_portfolio_stability=minimum_portfolio_stability,
        minimum_readiness_score=minimum_readiness_score,
        strategy_weight=strategy_weight,
        portfolio_weight=portfolio_weight,
    )

    return bool(result["ready"])


def production_readiness_message(
    portfolio_stability_score: float,
    *,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    minimum_strategy_stability: float = 0.70,
    minimum_portfolio_stability: float = 0.70,
    minimum_readiness_score: float = 0.70,
    strategy_weight: float = 0.70,
    portfolio_weight: float = 0.30,
) -> str:
    """
    Build a human-readable production readiness message.
    """
    result = build_production_readiness(
        portfolio_stability_score=portfolio_stability_score,
        results_dir=results_dir,
        minimum_strategy_stability=minimum_strategy_stability,
        minimum_portfolio_stability=minimum_portfolio_stability,
        minimum_readiness_score=minimum_readiness_score,
        strategy_weight=strategy_weight,
        portfolio_weight=portfolio_weight,
    )

    strategy = result.get("strategy")

    if result["ready"]:
        return (
            "PRODUCTION READY: "
            f"{strategy} passed strategy stability, "
            "portfolio stability, and combined readiness gates."
        )

    failed_gates = result.get("failed_gates", [])

    if not failed_gates:
        return "PRODUCTION BLOCKED."

    return (
        "PRODUCTION BLOCKED: "
        f"{strategy or 'no strategy selected'}; "
        f"failed gates: {', '.join(failed_gates)}."
    )
