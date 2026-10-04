"""Canonical Bounded Mathematical Search Strategy & Generation Architecture.

Provides pluggable search strategy protocol, deterministic symbolic enumeration,
canonical result metadata, and fail-closed constitutional governance for
Mathematical Relationship Discovery in Project 1.

ANTI-OVERFITTING / ANTI-RECURRENCE BOUNDARY:
---------------------------------------------
The mathematical search generator is strictly a structural discovery layer.
- It NEVER evaluates market data or DataFrames during candidate generation.
- It NEVER ranks candidates using performance outcomes, returns, Sharpe, or PnL.
- It NEVER inspects ResearchEvidence, OOS results, or live production signals.
- It MUST NOT bypass memory or discovery governance.
Performance and economic evaluation belong exclusively to the existing downstream
research execution and evidence pipeline.
"""

from __future__ import annotations

import enum
import hashlib
import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Iterator, List, Protocol, Sequence, Set, Tuple, runtime_checkable

from src.evaluation.mathematical_expression import (
    MathematicalExpression,
    MathematicalExpressionError,
    MathematicalOperator,
    MathematicalSearchSpace,
    SearchSpaceValidationError,
)
from src.evaluation.mathematical_expression_candidate import (
    MathematicalCandidateValidationError,
    MathematicalExpressionCandidate,
    MathematicalSignalInterpretationPolicy,
)
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
)


class MathematicalSearchError(MathematicalExpressionError):
    """Raised when mathematical search generation fails or encounters invalid configuration/lineage."""


class SearchTerminationReason(str, enum.Enum):
    """Reason why candidate generation concluded."""

    EXHAUSTED_SEARCH_SPACE = "EXHAUSTED_SEARCH_SPACE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


@dataclass(frozen=True)
class MathematicalSearchResult:
    """Immutable result and metadata object produced by a mathematical search strategy.

    Contains no performance metrics or market observations. Performance evaluation
    belongs exclusively to downstream research execution.
    """

    search_space_fingerprint: str
    generator_id: str
    generator_version: str
    random_seed: int
    generated_count: int
    budget: int
    termination_reason: SearchTerminationReason
    candidates: Tuple[MathematicalExpressionCandidate, ...]

    def __post_init__(self) -> None:
        if not self.search_space_fingerprint or not isinstance(self.search_space_fingerprint, str):
            raise MathematicalSearchError("search_space_fingerprint must be a non-empty string.")
        if not self.generator_id or not isinstance(self.generator_id, str):
            raise MathematicalSearchError("generator_id must be a non-empty string.")
        if not self.generator_version or not isinstance(self.generator_version, str):
            raise MathematicalSearchError("generator_version must be a non-empty string.")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise MathematicalSearchError("random_seed must be an integer.")
        if not isinstance(self.generated_count, int) or self.generated_count < 0:
            raise MathematicalSearchError("generated_count must be a non-negative integer.")
        if not isinstance(self.budget, int) or self.budget < 1:
            raise MathematicalSearchError("budget must be a positive integer.")
        if not isinstance(self.termination_reason, SearchTerminationReason):
            raise MathematicalSearchError(f"Invalid termination_reason: {self.termination_reason}")
        if not isinstance(self.candidates, tuple):
            raise MathematicalSearchError("candidates must be a tuple of MathematicalExpressionCandidate instances.")
        if len(self.candidates) != self.generated_count:
            raise MathematicalSearchError(
                f"generated_count ({self.generated_count}) does not match candidates tuple length ({len(self.candidates)})."
            )


@runtime_checkable
class MathematicalSearchStrategy(Protocol):
    """Pluggable search strategy protocol for generating MathematicalExpressionCandidates."""

    def search(
        self,
        search_space: MathematicalSearchSpace,
        dataset_scope: DatasetScope,
        execution_assumptions: ExecutionAssumptions,
        code_provenance: CodeProvenance,
        signal_policy: MathematicalSignalInterpretationPolicy | None = None,
        constant_values: Sequence[float] = (),
        limit: int | None = None,
        generator_id: str = "symbolic_search",
        generator_version: str = "1.0",
        random_seed: int = 0,
    ) -> MathematicalSearchResult:
        """Generate bounded MathematicalExpressionCandidate objects from a search space constitution."""
        ...


class SymbolicSearch:
    """Concrete deterministic structural enumeration strategy for Mathematical Expression search space.

    Generates terminal and composite AST expressions in deterministic order (by node count and operator priority),
    enforcing canonical commutative ordering for ADD and MUL, and deduplicating by AST SHA-256 fingerprint.

    ANTI-OVERFITTING GUARANTEE:
    - Pure structural search with zero market data inspection, zero returns calculation, zero backtesting.
    """

    def search(
        self,
        search_space: MathematicalSearchSpace,
        dataset_scope: DatasetScope,
        execution_assumptions: ExecutionAssumptions,
        code_provenance: CodeProvenance,
        signal_policy: MathematicalSignalInterpretationPolicy | None = None,
        constant_values: Sequence[float] = (),
        limit: int | None = None,
        generator_id: str = "symbolic_search",
        generator_version: str = "1.0",
        random_seed: int = 0,
    ) -> MathematicalSearchResult:
        """Perform bounded, deterministic symbolic search over the search space constitution."""
        # 1. Validate search_space input
        if not isinstance(search_space, MathematicalSearchSpace):
            raise MathematicalSearchError(f"search_space must be MathematicalSearchSpace, got {type(search_space)}")

        # 2. Validate lineage parameters and fail closed if missing or mismatched
        if dataset_scope is None or not isinstance(dataset_scope, DatasetScope):
            raise MathematicalSearchError("dataset_scope must be a valid DatasetScope instance.")
        if execution_assumptions is None or not isinstance(execution_assumptions, ExecutionAssumptions):
            raise MathematicalSearchError("execution_assumptions must be a valid ExecutionAssumptions instance.")
        if code_provenance is None or not isinstance(code_provenance, CodeProvenance):
            raise MathematicalSearchError("code_provenance must be a valid CodeProvenance instance.")

        if search_space.dataset_scope is not None and search_space.dataset_scope != dataset_scope:
            raise MathematicalSearchError(
                f"Supplied DatasetScope ({dataset_scope}) does not match SearchSpace DatasetScope ({search_space.dataset_scope})."
            )
        if search_space.execution_assumptions is not None and search_space.execution_assumptions != execution_assumptions:
            raise MathematicalSearchError(
                f"Supplied ExecutionAssumptions ({execution_assumptions}) does not match SearchSpace ExecutionAssumptions ({search_space.execution_assumptions})."
            )
        if search_space.code_provenance is not None and search_space.code_provenance != code_provenance:
            raise MathematicalSearchError(
                f"Supplied CodeProvenance ({code_provenance}) does not match SearchSpace CodeProvenance ({search_space.code_provenance})."
            )

        # 3. Validate limit and compute effective budget
        if limit is not None:
            if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
                raise MathematicalSearchError(f"Requested search limit must be a positive integer, got {limit}")
            effective_budget = min(limit, search_space.max_search_budget)
        else:
            effective_budget = search_space.max_search_budget

        # 4. Validate signal policy
        if signal_policy is None:
            effective_policy = MathematicalSignalInterpretationPolicy()
        else:
            if not isinstance(signal_policy, MathematicalSignalInterpretationPolicy):
                raise MathematicalSearchError(
                    f"signal_policy must be MathematicalSignalInterpretationPolicy, got {type(signal_policy)}"
                )
            effective_policy = signal_policy

        # 5. Validate generator metadata
        if not generator_id or not isinstance(generator_id, str) or not generator_id.strip():
            raise MathematicalSearchError("generator_id must be a non-empty string.")
        if not generator_version or not isinstance(generator_version, str) or not generator_version.strip():
            raise MathematicalSearchError("generator_version must be a non-empty string.")
        if not isinstance(random_seed, int) or isinstance(random_seed, bool):
            raise MathematicalSearchError("random_seed must be an integer.")

        # Fail-closed generator metadata consistency boundary between search space constitution and execution arguments
        if search_space.generator_id != generator_id:
            raise MathematicalSearchError(
                f"Generator metadata mismatch for 'generator_id': search space constitution has '{search_space.generator_id}' "
                f"but execution argument supplied '{generator_id}'."
            )
        if search_space.generator_version != generator_version:
            raise MathematicalSearchError(
                f"Generator metadata mismatch for 'generator_version': search space constitution has '{search_space.generator_version}' "
                f"but execution argument supplied '{generator_version}'."
            )
        if search_space.random_seed != random_seed:
            raise MathematicalSearchError(
                f"Generator metadata mismatch for 'random_seed': search space constitution has '{search_space.random_seed}' "
                f"but execution argument supplied '{random_seed}'."
            )

        # 6. Validate constant values against search space bounds and precision
        validated_constants = self._validate_constants(constant_values, search_space)

        # 7. Perform deterministic structural AST enumeration
        candidates, termination_reason = self._enumerate_candidates(
            search_space=search_space,
            dataset_scope=dataset_scope,
            execution_assumptions=execution_assumptions,
            code_provenance=code_provenance,
            signal_policy=effective_policy,
            constants=validated_constants,
            budget=effective_budget,
            generator_id=generator_id,
            generator_version=generator_version,
            random_seed=random_seed,
        )

        return MathematicalSearchResult(
            search_space_fingerprint=search_space.fingerprint,
            generator_id=generator_id,
            generator_version=generator_version,
            random_seed=random_seed,
            generated_count=len(candidates),
            budget=effective_budget,
            termination_reason=termination_reason,
            candidates=tuple(candidates),
        )

    def _validate_constants(
        self, constant_values: Sequence[float], search_space: MathematicalSearchSpace
    ) -> Tuple[float, ...]:
        """Validate explicit constant values against search space bounds and precision."""
        if not isinstance(constant_values, (tuple, list, Sequence)):
            raise MathematicalSearchError("constant_values must be a sequence.")

        validated: List[float] = []
        prec_dec = Decimal(str(search_space.constant_precision))
        c_min, c_max = search_space.constant_bounds

        for c in constant_values:
            if not isinstance(c, (int, float)) or isinstance(c, bool):
                raise MathematicalSearchError(f"Constant value must be float or int, got {type(c)}")

            f_val = float(c)
            if math.isnan(f_val) or math.isinf(f_val):
                raise MathematicalSearchError(f"Non-finite constant value rejected: {c}")

            if f_val < c_min or f_val > c_max:
                raise MathematicalSearchError(
                    f"Constant value {f_val} violates search space constant_bounds ({c_min}, {c_max})."
                )

            c_dec = Decimal(str(f_val))
            if c_dec % prec_dec != Decimal("0"):
                raise MathematicalSearchError(
                    f"Constant value {f_val} violates search space constant_precision ({search_space.constant_precision})."
                )

            if f_val not in validated:
                validated.append(f_val)

        return tuple(sorted(validated))

    def _enumerate_candidates(
        self,
        *,
        search_space: MathematicalSearchSpace,
        dataset_scope: DatasetScope,
        execution_assumptions: ExecutionAssumptions,
        code_provenance: CodeProvenance,
        signal_policy: MathematicalSignalInterpretationPolicy,
        constants: Tuple[float, ...],
        budget: int,
        generator_id: str,
        generator_version: str,
        random_seed: int,
    ) -> Tuple[List[MathematicalExpressionCandidate], SearchTerminationReason]:
        """Iteratively enumerate expressions by node count in deterministic order."""
        visited_fingerprints: Set[str] = set()
        candidates: List[MathematicalExpressionCandidate] = []
        exprs_by_size: Dict[int, List[MathematicalExpression]] = {}

        # Categorize allowed operators deterministically
        allowed_ops = tuple(sorted(search_space.allowed_operators, key=lambda op: op.value))
        unary_ops = tuple(op for op in allowed_ops if op.arity == 1)
        binary_ops = tuple(op for op in allowed_ops if op.arity == 2)

        sorted_features = tuple(sorted(search_space.allowed_features))
        lag_range = tuple(range(search_space.min_lag, search_space.max_lag + 1))

        # Size 1: Terminal nodes (FEATURE, CONSTANT)
        size_1_exprs: List[MathematicalExpression] = []

        # Generate FEATURE terminals
        if MathematicalOperator.FEATURE in allowed_ops:
            for feat in sorted_features:
                for lag in lag_range:
                    try:
                        expr = MathematicalExpression(
                            operator=MathematicalOperator.FEATURE,
                            feature_name=feat,
                            lag=lag,
                            generator_id=generator_id,
                            generator_version=generator_version,
                            random_seed=random_seed,
                            dataset_scope=dataset_scope,
                            execution_assumptions=execution_assumptions,
                            code_provenance=code_provenance,
                        )
                        search_space.validate_expression(expr)
                    except (SearchSpaceValidationError, MathematicalExpressionError):
                        continue

                    if expr.fingerprint not in visited_fingerprints:
                        visited_fingerprints.add(expr.fingerprint)
                        size_1_exprs.append(expr)

                        cand = MathematicalExpressionCandidate(
                            expression=expr,
                            search_space=search_space,
                            signal_policy=signal_policy,
                            dataset_scope=dataset_scope,
                            execution_assumptions=execution_assumptions,
                            code_provenance=code_provenance,
                            generator_id=generator_id,
                            generator_version=generator_version,
                            random_seed=random_seed,
                        )
                        cand.validate()
                        candidates.append(cand)

                        if len(candidates) == budget:
                            return candidates, SearchTerminationReason.BUDGET_EXHAUSTED

        # Generate CONSTANT terminals
        if MathematicalOperator.CONSTANT in allowed_ops:
            for c in constants:
                # Constants have lag=0
                if 0 in lag_range or (search_space.min_lag <= 0 <= search_space.max_lag):
                    try:
                        expr = MathematicalExpression(
                            operator=MathematicalOperator.CONSTANT,
                            constant_value=c,
                            lag=0,
                            generator_id=generator_id,
                            generator_version=generator_version,
                            random_seed=random_seed,
                            dataset_scope=dataset_scope,
                            execution_assumptions=execution_assumptions,
                            code_provenance=code_provenance,
                        )
                        search_space.validate_expression(expr)
                    except (SearchSpaceValidationError, MathematicalExpressionError):
                        continue

                    if expr.fingerprint not in visited_fingerprints:
                        visited_fingerprints.add(expr.fingerprint)
                        size_1_exprs.append(expr)

                        cand = MathematicalExpressionCandidate(
                            expression=expr,
                            search_space=search_space,
                            signal_policy=signal_policy,
                            dataset_scope=dataset_scope,
                            execution_assumptions=execution_assumptions,
                            code_provenance=code_provenance,
                            generator_id=generator_id,
                            generator_version=generator_version,
                            random_seed=random_seed,
                        )
                        cand.validate()
                        candidates.append(cand)

                        if len(candidates) == budget:
                            return candidates, SearchTerminationReason.BUDGET_EXHAUSTED

        exprs_by_size[1] = size_1_exprs

        # Size 2 up to max_node_count: Non-terminal composite expressions
        for size in range(2, search_space.max_node_count + 1):
            curr_size_exprs: List[MathematicalExpression] = []

            # 1. Unary operators (size = 1 + child_size => child_size = size - 1)
            child_size = size - 1
            if child_size in exprs_by_size:
                for op in unary_ops:
                    for child in exprs_by_size[child_size]:
                        for lag in lag_range:
                            try:
                                expr = MathematicalExpression(
                                    operator=op,
                                    children=(child,),
                                    lag=lag,
                                    generator_id=generator_id,
                                    generator_version=generator_version,
                                    random_seed=random_seed,
                                    dataset_scope=dataset_scope,
                                    execution_assumptions=execution_assumptions,
                                    code_provenance=code_provenance,
                                )
                                search_space.validate_expression(expr)
                            except (SearchSpaceValidationError, MathematicalExpressionError):
                                continue

                            if expr.fingerprint not in visited_fingerprints:
                                visited_fingerprints.add(expr.fingerprint)
                                curr_size_exprs.append(expr)

                                cand = MathematicalExpressionCandidate(
                                    expression=expr,
                                    search_space=search_space,
                                    signal_policy=signal_policy,
                                    dataset_scope=dataset_scope,
                                    execution_assumptions=execution_assumptions,
                                    code_provenance=code_provenance,
                                    generator_id=generator_id,
                                    generator_version=generator_version,
                                    random_seed=random_seed,
                                )
                                cand.validate()
                                candidates.append(cand)

                                if len(candidates) == budget:
                                    return candidates, SearchTerminationReason.BUDGET_EXHAUSTED

            # 2. Binary operators (size = 1 + left_size + right_size)
            for left_size in range(1, size - 1):
                right_size = size - 1 - left_size
                if left_size in exprs_by_size and right_size in exprs_by_size:
                    for op in binary_ops:
                        is_commutative = op in (MathematicalOperator.ADD, MathematicalOperator.MUL)
                        for left_expr in exprs_by_size[left_size]:
                            for right_expr in exprs_by_size[right_size]:
                                # Canonical child ordering for commutative ADD and MUL
                                if is_commutative and left_expr.fingerprint > right_expr.fingerprint:
                                    continue

                                for lag in lag_range:
                                    try:
                                        expr = MathematicalExpression(
                                            operator=op,
                                            children=(left_expr, right_expr),
                                            lag=lag,
                                            generator_id=generator_id,
                                            generator_version=generator_version,
                                            random_seed=random_seed,
                                            dataset_scope=dataset_scope,
                                            execution_assumptions=execution_assumptions,
                                            code_provenance=code_provenance,
                                        )
                                        search_space.validate_expression(expr)
                                    except (SearchSpaceValidationError, MathematicalExpressionError):
                                        continue

                                    if expr.fingerprint not in visited_fingerprints:
                                        visited_fingerprints.add(expr.fingerprint)
                                        curr_size_exprs.append(expr)

                                        cand = MathematicalExpressionCandidate(
                                            expression=expr,
                                            search_space=search_space,
                                            signal_policy=signal_policy,
                                            dataset_scope=dataset_scope,
                                            execution_assumptions=execution_assumptions,
                                            code_provenance=code_provenance,
                                            generator_id=generator_id,
                                            generator_version=generator_version,
                                            random_seed=random_seed,
                                        )
                                        cand.validate()
                                        candidates.append(cand)

                                        if len(candidates) == budget:
                                            return candidates, SearchTerminationReason.BUDGET_EXHAUSTED

            exprs_by_size[size] = curr_size_exprs

        return candidates, SearchTerminationReason.EXHAUSTED_SEARCH_SPACE
