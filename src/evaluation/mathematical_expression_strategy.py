"""Research-Only Mathematical Expression Strategy Adapter & Dedicated Registry.

Provides signal adapter for MathematicalExpression execution on pandas DataFrames and
a dedicated, isolated StrategyRegistry factory for research-only mathematical experiments.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluation.mathematical_expression import (
    MathematicalExpression,
    MathematicalExpressionError,
)
from src.strategies.registry import StrategyRegistry, StrategySpec


def validate_mathematical_hypothesis_parameters(
    parameters: dict,
    strategy_name: str = "mathematical_expression",
) -> tuple[MathematicalExpression, Any]:
    """Validate that mathematical hypothesis parameters are internally self-consistent and untampered.

    Fails closed before any DataFrame or AST evaluation if:
    - strategy_name != "mathematical_expression"
    - required mathematical identity fields are missing or incomplete
    - reconstructed expression.fingerprint != parameters["expression_fingerprint"]
    - reconstructed signal_policy.fingerprint != parameters["signal_policy_fingerprint"]
    - re-derived candidate fingerprint != parameters["candidate_fingerprint"]
    - candidate_id != math_cand_<first 16 chars of candidate fingerprint>
    """
    if strategy_name != "mathematical_expression":
        raise MathematicalExpressionError(
            f"Strategy '{strategy_name}' is not the canonical research-only mathematical strategy ('mathematical_expression')."
        )

    if not isinstance(parameters, dict):
        raise MathematicalExpressionError("parameters must be a dictionary.")

    required_fields = (
        "expression_dict",
        "expression_fingerprint",
        "search_space_fingerprint",
        "signal_policy_dict",
        "signal_policy_fingerprint",
        "candidate_id",
        "candidate_fingerprint",
        "candidate_version",
        "generator_id",
        "generator_version",
        "random_seed",
    )
    for field_name in required_fields:
        if field_name not in parameters or parameters[field_name] is None:
            raise MathematicalExpressionError(
                f"Tampered or incomplete mathematical hypothesis: missing required parameter '{field_name}'."
            )

    # Reconstruct expression & verify fingerprint match
    try:
        expr = MathematicalExpression.from_canonical_dict(parameters["expression_dict"])
    except Exception as exc:
        raise MathematicalExpressionError(f"Failed to reconstruct MathematicalExpression from expression_dict: {exc}") from exc

    if expr.fingerprint != parameters["expression_fingerprint"]:
        raise MathematicalExpressionError(
            f"Mathematical identity mismatch: reconstructed expression fingerprint '{expr.fingerprint}' "
            f"does not match parameter fingerprint '{parameters['expression_fingerprint']}'."
        )

    # Validate expression dataset scope, execution assumptions, code provenance presence
    if expr.dataset_scope is None or expr.execution_assumptions is None or expr.code_provenance is None:
        raise MathematicalExpressionError(
            "Mathematical expression in hypothesis parameters lacks complete research lineage (dataset_scope, execution_assumptions, code_provenance)."
        )

    from src.evaluation.mathematical_expression_candidate import (
        MathematicalExpressionCandidate,
        MathematicalSignalInterpretationPolicy,
    )

    # Reconstruct signal policy & verify fingerprint match
    try:
        policy = MathematicalSignalInterpretationPolicy.from_canonical_dict(parameters["signal_policy_dict"])
    except Exception as exc:
        raise MathematicalExpressionError(f"Failed to reconstruct MathematicalSignalInterpretationPolicy from signal_policy_dict: {exc}") from exc

    if policy.fingerprint != parameters["signal_policy_fingerprint"]:
        raise MathematicalExpressionError(
            f"Mathematical signal policy identity mismatch: reconstructed policy fingerprint '{policy.fingerprint}' "
            f"does not match parameter fingerprint '{parameters['signal_policy_fingerprint']}'."
        )

    # Re-derive expected candidate fingerprint using single canonical helper in MathematicalExpressionCandidate
    expected_cand_fp = MathematicalExpressionCandidate.compute_fingerprint_from_components(
        version=str(parameters["candidate_version"]),
        expression_dict=expr.to_canonical_dict(),
        expression_fingerprint=expr.fingerprint,
        search_space_fingerprint=str(parameters["search_space_fingerprint"]),
        signal_policy_dict=policy.to_canonical_dict(),
        signal_policy_fingerprint=policy.fingerprint,
        generator_id=str(parameters["generator_id"]),
        generator_version=str(parameters["generator_version"]),
        random_seed=int(parameters["random_seed"]),
        dataset_scope=expr.dataset_scope,
        execution_assumptions=expr.execution_assumptions,
        code_provenance=expr.code_provenance,
    )

    if expected_cand_fp != parameters["candidate_fingerprint"]:
        raise MathematicalExpressionError(
            f"Mathematical candidate identity mismatch: re-derived candidate fingerprint '{expected_cand_fp}' "
            f"does not match parameter candidate_fingerprint '{parameters['candidate_fingerprint']}'."
        )

    expected_cand_id = f"math_cand_{expected_cand_fp[:16]}"
    if parameters["candidate_id"] != expected_cand_id:
        raise MathematicalExpressionError(
            f"Mathematical candidate ID mismatch: candidate_id '{parameters['candidate_id']}' "
            f"does not match expected canonical ID '{expected_cand_id}'."
        )

    return expr, policy


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

    # If invoked via run_research_experiment, validate hypothesis parameters strictly for tampering
    if expression_obj is None and expression_dict is not None:
        param_context = {
            "expression_dict": expression_dict,
            "signal_policy_dict": signal_policy_dict,
            **kwargs,
        }
        expr, policy = validate_mathematical_hypothesis_parameters(param_context)
    else:
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
