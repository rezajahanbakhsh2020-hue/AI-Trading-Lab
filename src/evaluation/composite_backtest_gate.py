from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.evaluation.composite_backtest import CompositeBacktestResult
from src.evaluation.composite_validation import (
    CompositeValidation,
    validate_composite,
)


@dataclass(frozen=True)
class CompositeBacktestGate:
    """
    Final gate for an already backtested composite.

    The gate compares the composite's actual backtest ranking score
    against the actual ranking scores of its selected components.

    This is still an in-sample gate. Walk-Forward and Stability remain
    mandatory before treating the composite as production-ready.
    """

    validation: CompositeValidation
    component_count: int
    composite_total_return: float
    best_component_total_return: float
    composite_sharpe: float
    best_component_sharpe: float
    composite_max_drawdown: float
    best_component_max_drawdown: float

    @property
    def accepted(self) -> bool:
        return self.validation.accepted

    @property
    def composite_name(self) -> str:
        return self.validation.composite_name

    @property
    def reason(self) -> str:
        return self.validation.reason

    def as_dict(self) -> dict:
        return {
            **self.validation.as_dict(),
            "component_count": self.component_count,
            "composite_total_return": self.composite_total_return,
            "best_component_total_return": self.best_component_total_return,
            "composite_sharpe": self.composite_sharpe,
            "best_component_sharpe": self.best_component_sharpe,
            "composite_max_drawdown": self.composite_max_drawdown,
            "best_component_max_drawdown": self.best_component_max_drawdown,
        }


def _validate_comparison_frame(
    comparison: pd.DataFrame,
) -> None:
    if not isinstance(comparison, pd.DataFrame):
        raise TypeError("comparison must be a pandas DataFrame.")

    required = {
        "name",
        "kind",
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
        "ranking_score",
    }

    missing = required.difference(comparison.columns)

    if missing:
        raise ValueError(
            "comparison is missing columns: "
            + ", ".join(sorted(missing))
        )

    if comparison.empty:
        raise ValueError("comparison must not be empty.")


def validate_composite_backtest(
    composite: CompositeBacktestResult,
    comparison: pd.DataFrame,
    *,
    minimum_improvement: float = 0.0,
) -> CompositeBacktestGate:
    """
    Validate a composite using its actual backtest result.

    Unlike the original ranking-based validation helper, this function
    receives the actual CompositeBacktestResult and therefore validates
    the score produced by the composite's own backtest.

    The selected component rows must come from the same comparison
    produced by compare_composite_to_components().
    """

    if not isinstance(
        composite,
        CompositeBacktestResult,
    ):
        raise TypeError(
            "composite must be a CompositeBacktestResult."
        )

    _validate_comparison_frame(comparison)

    if minimum_improvement < 0:
        raise ValueError(
            "minimum_improvement must be non-negative."
        )

    component_rows = comparison[
        comparison["kind"] == "component"
    ].copy()

    composite_rows = comparison[
        comparison["kind"] == "composite"
    ].copy()

    if component_rows.empty:
        raise ValueError(
            "comparison must contain at least one component."
        )

    if composite_rows.empty:
        raise ValueError(
            "comparison must contain a composite row."
        )

    matching_composite = composite_rows[
        composite_rows["name"].astype(str).str.strip().str.lower()
        == composite.name
    ]

    if matching_composite.empty:
        raise ValueError(
            "comparison does not contain the supplied composite."
        )

    actual_composite_score = float(
        composite.evaluation.ranking_score
    )

    component_scores = component_rows[
        "ranking_score"
    ].astype(float).tolist()

    validation = validate_composite(
        actual_composite_score,
        component_scores,
        composite_name=composite.name,
        minimum_improvement=minimum_improvement,
    )

    best_component = component_rows.sort_values(
        "ranking_score",
        ascending=False,
    ).iloc[0]

    return CompositeBacktestGate(
        validation=validation,
        component_count=len(component_rows),
        composite_total_return=float(
            composite.total_return
        ),
        best_component_total_return=float(
            best_component["total_return"]
        ),
        composite_sharpe=float(
            composite.sharpe_ratio
        ),
        best_component_sharpe=float(
            best_component["sharpe_ratio"]
        ),
        composite_max_drawdown=float(
            composite.max_drawdown
        ),
        best_component_max_drawdown=float(
            best_component["max_drawdown"]
        ),
    )


def require_valid_composite_backtest(
    gate: CompositeBacktestGate,
) -> CompositeBacktestGate:
    """
    Stop the pipeline when the actual composite backtest fails.
    """

    if not isinstance(
        gate,
        CompositeBacktestGate,
    ):
        raise TypeError(
            "gate must be a CompositeBacktestGate."
        )

    if not gate.accepted:
        raise ValueError(
            f"Composite '{gate.composite_name}' failed "
            f"the actual backtest gate: {gate.reason}"
        )

    return gate
