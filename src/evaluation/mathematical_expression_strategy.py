"""Research-Only Mathematical Expression Strategy Adapter & Dedicated Registry.

Provides signal adapter for MathematicalExpression execution on pandas DataFrames and
a dedicated, isolated StrategyRegistry factory for research-only mathematical experiments.
"""

from __future__ import annotations

import pandas as pd

from src.evaluation.mathematical_expression import (
    MathematicalExpression,
    MathematicalExpressionError,
)
from src.strategies.registry import StrategyRegistry, StrategySpec


def generate_mathematical_expression_signal(
    df: pd.DataFrame,
    expression_dict: dict | None = None,
    signal_policy_dict: dict | None = None,
    expression_obj: MathematicalExpression | None = None,
    signal_policy_obj: MathematicalSignalInterpretationPolicy | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Generate trading signal DataFrame from a MathematicalExpression.

    Reconstructs MathematicalExpression and MathematicalSignalInterpretationPolicy,
    evaluates the expression AST over input DataFrame, applies sign interpretation policy,
    and enforces NO TRADE (0) during the max_lookback warmup period.

    Fails closed on MathematicalEvaluationError / MathematicalDomainError / NaN / Inf in valid post-warmup rows.
    Preserves input row ordering and index. Never inspects future rows.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df)}")

    # Resolve expression instance
    if expression_obj is not None:
        expr = expression_obj
    elif expression_dict is not None:
        expr = MathematicalExpression.from_canonical_dict(expression_dict)
    else:
        raise ValueError("Either expression_obj or expression_dict must be supplied.")

    from src.evaluation.mathematical_expression_candidate import (
        MathematicalSignalInterpretationPolicy,
    )

    # Resolve signal policy instance
    if signal_policy_obj is not None:
        policy = signal_policy_obj
    elif signal_policy_dict is not None:
        policy = MathematicalSignalInterpretationPolicy.from_canonical_dict(signal_policy_dict)
    else:
        policy = MathematicalSignalInterpretationPolicy()

    result = df.copy()

    # Evaluate expression Series
    expr_values = expr.evaluate(result, t=None)

    # Convert evaluated outputs to signals
    # Warmup period (< expr.max_lookback) MUST default to NO TRADE (0)
    signals = pd.Series(0, index=result.index, dtype=int)

    warmup_len = expr.max_lookback
    if warmup_len < len(result):
        valid_values = expr_values.iloc[warmup_len:]
        valid_signals = policy.evaluate_series(valid_values)
        signals.iloc[warmup_len:] = valid_signals

    result["signal"] = signals
    result["expression_value"] = expr_values

    return result


def create_mathematical_research_registry() -> StrategyRegistry:
    """Create a dedicated, research-only StrategyRegistry containing mathematical_expression strategy.

    ISOLATION INVARIANT:
    This registry is created on demand for research experiments and is NOT registered
    in DEFAULT_REGISTRY, preventing mathematical expression candidates from becoming
    production strategies merely by being executable in research.
    """
    spec = StrategySpec(
        name="mathematical_expression",
        category="mathematical_discovery",
        function=generate_mathematical_expression_signal,
        description="Research-only Strategy Adapter for canonical MathematicalExpression AST evaluation.",
        supports_long=True,
        supports_short=True,
    )

    return StrategyRegistry((spec,))
