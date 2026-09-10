from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class CompositeValidation:
    composite_name: str
    composite_score: float
    best_component_score: float
    mean_component_score: float
    improvement_vs_best: float
    improvement_vs_mean: float
    accepted: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "composite_name": self.composite_name,
            "composite_score": self.composite_score,
            "best_component_score": self.best_component_score,
            "mean_component_score": self.mean_component_score,
            "improvement_vs_best": self.improvement_vs_best,
            "improvement_vs_mean": self.improvement_vs_mean,
            "accepted": self.accepted,
            "reason": self.reason,
        }


def validate_composite(
    composite_score: float,
    component_scores: Iterable[float],
    *,
    composite_name: str = "ensemble",
    minimum_improvement: float = 0.0,
) -> CompositeValidation:
    """
    Decide whether a composite beats its individual components.

    Acceptance requires the composite to outperform the strongest
    component by at least ``minimum_improvement``.

    This is a validation gate, not a claim of future profitability.
    Walk-Forward and Stability remain additional gates.
    """

    if not isinstance(composite_name, str):
        raise TypeError("composite_name must be a string.")

    composite_name = composite_name.strip()

    if not composite_name:
        raise ValueError("composite_name must not be empty.")

    if minimum_improvement < 0:
        raise ValueError(
            "minimum_improvement must be non-negative."
        )

    scores = [float(score) for score in component_scores]

    if not scores:
        raise ValueError("component_scores must not be empty.")

    if any(pd.isna(score) for score in scores):
        raise ValueError("component scores must be finite.")

    composite = float(composite_score)

    if pd.isna(composite):
        raise ValueError("composite_score must be finite.")

    best = max(scores)
    mean = sum(scores) / len(scores)

    improvement_best = composite - best
    improvement_mean = composite - mean

    accepted = improvement_best >= float(minimum_improvement)

    if accepted:
        reason = (
            "Composite outperforms the strongest component "
            "by the required margin."
        )
    else:
        reason = (
            "Composite does not outperform the strongest component "
            "by the required margin."
        )

    return CompositeValidation(
        composite_name=composite_name,
        composite_score=composite,
        best_component_score=best,
        mean_component_score=mean,
        improvement_vs_best=improvement_best,
        improvement_vs_mean=improvement_mean,
        accepted=accepted,
        reason=reason,
    )


def validate_composite_from_evaluations(
    composite_score: float,
    evaluations: pd.DataFrame,
    *,
    top_n: int = 3,
    composite_name: str = "ensemble",
    score_column: str = "ranking_score",
    minimum_improvement: float = 0.0,
) -> CompositeValidation:
    """
    Validate a composite against the Top-N component evaluations.
    """

    if not isinstance(evaluations, pd.DataFrame):
        raise TypeError(
            "evaluations must be a pandas DataFrame."
        )

    if not isinstance(top_n, int):
        raise TypeError("top_n must be an integer.")

    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    if score_column not in evaluations.columns:
        raise ValueError(
            f"evaluations must contain '{score_column}'."
        )

    selected = evaluations.head(top_n)

    if selected.empty:
        raise ValueError(
            "evaluations must contain at least one component."
        )

    return validate_composite(
        composite_score,
        selected[score_column].tolist(),
        composite_name=composite_name,
        minimum_improvement=minimum_improvement,
    )


def require_accepted_composite(
    validation: CompositeValidation,
) -> CompositeValidation:
    """
    Raise if a composite fails the validation gate.
    """

    if not isinstance(validation, CompositeValidation):
        raise TypeError(
            "validation must be a CompositeValidation."
        )

    if not validation.accepted:
        raise ValueError(
            f"Composite '{validation.composite_name}' rejected: "
            f"{validation.reason}"
        )

    return validation
