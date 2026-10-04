"""Canonical Mathematical Expression + Search-Space Constitution Layer.

Provides immutable typed AST representations, temporal safety contracts, safe operator
evaluations, deterministic SHA-256 fingerprinting, and search-space governance for
Mathematical Relationship Discovery in Project 1.
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
)


class MathematicalExpressionError(ValueError):
    """Base exception for mathematical expression errors."""


class MathematicalDomainError(MathematicalExpressionError):
    """Raised when mathematical domain bounds are violated during evaluation."""


class MathematicalEvaluationError(MathematicalExpressionError):
    """Raised when expression evaluation fails unexpectedly or outputs invalid non-finite numbers."""


class SearchSpaceValidationError(MathematicalExpressionError):
    """Raised when an expression violates search space constraints."""


class MathematicalOperator(str, enum.Enum):
    """Allowlist of canonical mathematically meaningful operator primitives."""

    CONSTANT = "CONSTANT"
    FEATURE = "FEATURE"
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    PROTECTED_DIV = "PROTECTED_DIV"
    NEG = "NEG"
    PROTECTED_LOG = "PROTECTED_LOG"
    PROTECTED_SQRT = "PROTECTED_SQRT"
    ABS = "ABS"

    @property
    def arity(self) -> int:
        """Expected number of child expressions for this operator."""
        if self in (MathematicalOperator.CONSTANT, MathematicalOperator.FEATURE):
            return 0
        if self in (
            MathematicalOperator.NEG,
            MathematicalOperator.PROTECTED_LOG,
            MathematicalOperator.PROTECTED_SQRT,
            MathematicalOperator.ABS,
        ):
            return 1
        if self in (
            MathematicalOperator.ADD,
            MathematicalOperator.SUB,
            MathematicalOperator.MUL,
            MathematicalOperator.PROTECTED_DIV,
        ):
            return 2
        raise ValueError(f"Unknown operator arity for {self}")


@dataclass(frozen=True)
class MathematicalExpression:
    """Canonical immutable typed Mathematical Expression AST node.

    Represents a node in a mathematical expression tree with complete provenance,
    temporal metadata, complexity tracking, and deterministic fingerprinting.
    """

    operator: MathematicalOperator
    children: Tuple[MathematicalExpression, ...] = ()
    feature_name: str | None = None
    constant_value: float | None = None
    lag: int = 0
    generator_id: str = "manual"
    generator_version: str = "1.0"
    random_seed: int = 0
    dataset_scope: DatasetScope | None = None
    execution_assumptions: ExecutionAssumptions | None = None
    code_provenance: CodeProvenance | None = None
    version: str = "1.0"

    def __post_init__(self) -> None:
        """Enforce strict fail-closed constitutional invariants upon construction."""
        # Validate operator
        if not isinstance(self.operator, MathematicalOperator):
            if isinstance(self.operator, str):
                try:
                    object.__setattr__(self, "operator", MathematicalOperator(self.operator))
                except ValueError:
                    raise MathematicalExpressionError(f"Unknown operator: {self.operator}")
            else:
                raise MathematicalExpressionError(f"Invalid operator type: {type(self.operator)}")

        # Validate children container
        if not isinstance(self.children, tuple):
            if isinstance(self.children, (list, Sequence)):
                object.__setattr__(self, "children", tuple(self.children))
            else:
                raise MathematicalExpressionError("children must be a tuple or sequence")

        # Validate operator arity
        expected_arity = self.operator.arity
        if len(self.children) != expected_arity:
            raise MathematicalExpressionError(
                f"Operator {self.operator.value} requires arity {expected_arity}, got {len(self.children)} children."
            )

        # Validate children types and cycle prevention
        for child in self.children:
            if not isinstance(child, MathematicalExpression):
                raise MathematicalExpressionError(
                    f"Child node must be instance of MathematicalExpression, got {type(child)}"
                )

        self._check_cycles()

        # Validate lag parameter (must be non-negative integer, no negative/future lags allowed)
        if not isinstance(self.lag, int) or isinstance(self.lag, bool):
            raise MathematicalExpressionError(f"lag must be an integer, got {type(self.lag)}")
        if self.lag < 0:
            raise MathematicalExpressionError(
                f"Temporal safety violation: lag cannot be negative ({self.lag}). Future lookahead is strictly forbidden."
            )

        # Validate CONSTANT node specifics
        if self.operator == MathematicalOperator.CONSTANT:
            if self.constant_value is None:
                raise MathematicalExpressionError("CONSTANT node requires constant_value to be specified.")
            if not isinstance(self.constant_value, (int, float)) or isinstance(self.constant_value, bool):
                raise MathematicalExpressionError(
                    f"constant_value must be float or int, got {type(self.constant_value)}"
                )

            val = float(self.constant_value)
            if math.isnan(val) or math.isinf(val):
                raise MathematicalExpressionError(f"CONSTANT node rejects NaN/Inf value: {self.constant_value}")

            object.__setattr__(self, "constant_value", val)

            if self.feature_name is not None:
                raise MathematicalExpressionError("CONSTANT node cannot specify feature_name.")

        else:
            if self.constant_value is not None:
                raise MathematicalExpressionError(f"Non-CONSTANT operator {self.operator.value} cannot specify constant_value.")

        # Validate FEATURE node specifics
        if self.operator == MathematicalOperator.FEATURE:
            if not self.feature_name or not isinstance(self.feature_name, str) or not self.feature_name.strip():
                raise MathematicalExpressionError("FEATURE node requires non-empty feature_name string.")
            object.__setattr__(self, "feature_name", self.feature_name.strip())
        else:
            if self.feature_name is not None:
                raise MathematicalExpressionError(f"Non-FEATURE operator {self.operator.value} cannot specify feature_name.")

        # Validate generator metadata
        if not self.generator_id or not isinstance(self.generator_id, str) or not self.generator_id.strip():
            raise MathematicalExpressionError("generator_id must be a non-empty string.")
        if not self.generator_version or not isinstance(self.generator_version, str) or not self.generator_version.strip():
            raise MathematicalExpressionError("generator_version must be a non-empty string.")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise MathematicalExpressionError("random_seed must be an integer.")
        if not self.version or not isinstance(self.version, str) or not self.version.strip():
            raise MathematicalExpressionError("version must be a non-empty string.")

        # Validate research identity / lineage types if supplied
        if self.dataset_scope is not None and not isinstance(self.dataset_scope, DatasetScope):
            raise MathematicalExpressionError(f"dataset_scope must be instance of DatasetScope, got {type(self.dataset_scope)}")
        if self.execution_assumptions is not None and not isinstance(self.execution_assumptions, ExecutionAssumptions):
            raise MathematicalExpressionError(
                f"execution_assumptions must be instance of ExecutionAssumptions, got {type(self.execution_assumptions)}"
            )
        if self.code_provenance is not None and not isinstance(self.code_provenance, CodeProvenance):
            raise MathematicalExpressionError(f"code_provenance must be instance of CodeProvenance, got {type(self.code_provenance)}")

        # Propagate / validate lineage consistency across child nodes
        for child in self.children:
            if self.dataset_scope is not None and child.dataset_scope is not None:
                if self.dataset_scope != child.dataset_scope:
                    raise MathematicalExpressionError(
                        "DatasetScope mismatch between parent and child expression node."
                    )
            if self.execution_assumptions is not None and child.execution_assumptions is not None:
                if self.execution_assumptions != child.execution_assumptions:
                    raise MathematicalExpressionError(
                        "ExecutionAssumptions mismatch between parent and child expression node."
                    )
            if self.code_provenance is not None and child.code_provenance is not None:
                if self.code_provenance != child.code_provenance:
                    raise MathematicalExpressionError(
                        "CodeProvenance mismatch between parent and child expression node."
                    )

    def _check_cycles(self, seen: Set[int] | None = None) -> None:
        """Detect self-reference / cycles in expression DAG structure."""
        if seen is None:
            seen = set()

        node_id = id(self)
        if node_id in seen:
            raise MathematicalExpressionError("Cycle detected in MathematicalExpression AST structure.")

        seen.add(node_id)
        for child in self.children:
            child._check_cycles(seen.copy())

    # --- Computed Structural Properties ---

    @property
    def node_count(self) -> int:
        """Total number of nodes in this expression AST."""
        return 1 + sum(child.node_count for child in self.children)

    @property
    def depth(self) -> int:
        """Maximum depth of this expression AST."""
        if not self.children:
            return 1
        return 1 + max(child.depth for child in self.children)

    @property
    def feature_references(self) -> Tuple[str, ...]:
        """Ordered tuple of unique feature names referenced in this expression."""
        refs: List[str] = []
        if self.operator == MathematicalOperator.FEATURE and self.feature_name:
            refs.append(self.feature_name)
        for child in self.children:
            for ref in child.feature_references:
                if ref not in refs:
                    refs.append(ref)
        return tuple(sorted(refs))

    @property
    def feature_count(self) -> int:
        """Number of unique feature references in this expression."""
        return len(self.feature_references)

    @property
    def interaction_count(self) -> int:
        """Number of feature interaction boundaries in this expression."""
        child_interactions = sum(child.interaction_count for child in self.children)
        if self.operator in (
            MathematicalOperator.ADD,
            MathematicalOperator.SUB,
            MathematicalOperator.MUL,
            MathematicalOperator.PROTECTED_DIV,
        ):
            left, right = self.children[0], self.children[1]
            if left.feature_count > 0 and right.feature_count > 0:
                return child_interactions + 1
        return child_interactions

    @property
    def max_lookback(self) -> int:
        """Declared maximum lookback (lag/temporal dependency) required by this expression."""
        child_max = max((child.max_lookback for child in self.children), default=0)
        return self.lag + child_max

    @property
    def warmup_requirement(self) -> int:
        """Declared warmup period required before expression evaluation is valid."""
        return self.max_lookback

    # --- Serialization & Deterministic Identity ---

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Convert expression node to deterministic, ordered canonical dictionary."""
        d: Dict[str, Any] = {
            "version": self.version,
            "operator": self.operator.value,
            "children": [child.to_canonical_dict() for child in self.children],
            "feature_name": self.feature_name,
            "constant_value": f"{self.constant_value:.12g}" if self.constant_value is not None else None,
            "lag": self.lag,
            "max_lookback": self.max_lookback,
            "warmup_requirement": self.warmup_requirement,
            "node_count": self.node_count,
            "depth": self.depth,
            "generator_id": self.generator_id,
            "generator_version": self.generator_version,
            "random_seed": self.random_seed,
            "dataset_scope": (
                {
                    "dataset_id": self.dataset_scope.dataset_id,
                    "symbol": self.dataset_scope.symbol,
                    "timeframe": self.dataset_scope.timeframe,
                    "start_date": self.dataset_scope.start_date,
                    "end_date": self.dataset_scope.end_date,
                }
                if self.dataset_scope is not None
                else None
            ),
            "execution_assumptions": (
                {
                    "transaction_cost": f"{self.execution_assumptions.transaction_cost:.12g}",
                    "slippage": f"{self.execution_assumptions.slippage:.12g}",
                    "latency_ms": f"{self.execution_assumptions.latency_ms:.12g}",
                }
                if self.execution_assumptions is not None
                else None
            ),
            "code_provenance": (
                {
                    "commit_sha": self.code_provenance.commit_sha,
                    "repository_status": self.code_provenance.repository_status,
                    "author": self.code_provenance.author,
                }
                if self.code_provenance is not None
                else None
            ),
        }
        return d

    def to_canonical_json(self) -> str:
        """Serialize expression to deterministic, canonical JSON string."""
        return json.dumps(self.to_canonical_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        """Deterministic SHA-256 fingerprint computed across canonical serialization."""
        return hashlib.sha256(self.to_canonical_json().encode("utf-8")).hexdigest()

    @property
    def expression_id(self) -> str:
        """Alias for fingerprint as canonical expression identity."""
        return self.fingerprint

    # --- Roundtrip Reconstruction ---

    @classmethod
    def from_canonical_dict(cls, data: Dict[str, Any]) -> MathematicalExpression:
        """Reconstruct a MathematicalExpression from a canonical dictionary."""
        op = MathematicalOperator(data["operator"])
        children = tuple(cls.from_canonical_dict(c) for c in data.get("children", []))
        feature_name = data.get("feature_name")
        constant_str = data.get("constant_value")
        constant_value = float(constant_str) if constant_str is not None else None
        lag = int(data.get("lag", 0))
        gen_id = data.get("generator_id", "manual")
        gen_ver = data.get("generator_version", "1.0")
        seed = int(data.get("random_seed", 0))
        ver = data.get("version", "1.0")

        ds_dict = data.get("dataset_scope")
        dataset_scope = DatasetScope(**ds_dict) if ds_dict is not None else None

        ea_dict = data.get("execution_assumptions")
        execution_assumptions = (
            ExecutionAssumptions(
                transaction_cost=float(ea_dict["transaction_cost"]),
                slippage=float(ea_dict["slippage"]),
                latency_ms=float(ea_dict["latency_ms"]),
            )
            if ea_dict is not None
            else None
        )

        cp_dict = data.get("code_provenance")
        code_provenance = CodeProvenance(**cp_dict) if cp_dict is not None else None

        return cls(
            operator=op,
            children=children,
            feature_name=feature_name,
            constant_value=constant_value,
            lag=lag,
            generator_id=gen_id,
            generator_version=gen_ver,
            random_seed=seed,
            dataset_scope=dataset_scope,
            execution_assumptions=execution_assumptions,
            code_provenance=code_provenance,
            version=ver,
        )

    # --- Safe Evaluation Engine ---

    def evaluate(self, data: pd.DataFrame, t: int | None = None) -> Union[float, pd.Series]:
        """Evaluate mathematical expression safely on input DataFrame.

        If index/time `t` is specified (as int row index), evaluates single point strictly
        using data at or before `t - lag`.
        If `t` is None, evaluates vector Series, applying lag offset.

        Fail closed on:
        - Missing feature columns
        - Negative or invalid lags
        - Protected division by zero or near-zero domain violation
        - Protected log non-positive domain violation
        - Protected sqrt negative domain violation
        - NaN or Infinite output propagation
        """
        if not isinstance(data, pd.DataFrame):
            raise MathematicalEvaluationError(f"data must be a pandas DataFrame, got {type(data)}")

        if t is not None:
            if not isinstance(t, int) or isinstance(t, bool):
                raise MathematicalEvaluationError(f"t index must be an integer, got {type(t)}")
            if t < 0 or t >= len(data):
                raise MathematicalEvaluationError(
                    f"Point evaluation index t={t} out of bounds for DataFrame length {len(data)}"
                )

            effective_t = t - self.lag
            if effective_t < 0:
                raise MathematicalEvaluationError(
                    f"Insufficient lookback history at index t={t} with expression lag={self.lag}. Warmup required: {self.max_lookback}"
                )

            return self._evaluate_point(data, effective_t)
        else:
            series_val = self._evaluate_series(data)
            if self.lag > 0:
                series_val = series_val.shift(self.lag)

            # Check that post-warmup values contain no NaN/Inf
            warmup_len = self.max_lookback
            valid_slice = series_val.iloc[warmup_len:]
            if valid_slice.isna().any() or np.isinf(valid_slice).any():
                raise MathematicalEvaluationError(
                    f"Non-finite output values produced by expression {self.operator.value} after warmup period."
                )

            return series_val

    def _evaluate_point(self, data: pd.DataFrame, t: int) -> float:
        """Evaluate single point at exact row index t."""
        if self.operator == MathematicalOperator.CONSTANT:
            return float(self.constant_value)

        if self.operator == MathematicalOperator.FEATURE:
            if self.feature_name not in data.columns:
                raise MathematicalEvaluationError(f"Feature column '{self.feature_name}' missing from input data.")
            val = data[self.feature_name].iloc[t]
            val = float(val)
            if math.isnan(val) or math.isinf(val):
                raise MathematicalEvaluationError(
                    f"Non-finite value ({val}) encountered for feature '{self.feature_name}' at index {t}."
                )
            return val

        child_vals = [c.evaluate(data, t=t) for c in self.children]

        if self.operator == MathematicalOperator.ADD:
            res = child_vals[0] + child_vals[1]
        elif self.operator == MathematicalOperator.SUB:
            res = child_vals[0] - child_vals[1]
        elif self.operator == MathematicalOperator.MUL:
            res = child_vals[0] * child_vals[1]
        elif self.operator == MathematicalOperator.PROTECTED_DIV:
            denom = child_vals[1]
            if abs(denom) < 1e-12:
                raise MathematicalDomainError(
                    f"Protected division by zero or near-zero denominator ({denom}) at index {t}."
                )
            res = child_vals[0] / denom
        elif self.operator == MathematicalOperator.NEG:
            res = -child_vals[0]
        elif self.operator == MathematicalOperator.PROTECTED_LOG:
            val = child_vals[0]
            if val <= 0.0:
                raise MathematicalDomainError(
                    f"Protected log domain violation: input value ({val}) <= 0 at index {t}."
                )
            res = math.log(val)
        elif self.operator == MathematicalOperator.PROTECTED_SQRT:
            val = child_vals[0]
            if val < 0.0:
                raise MathematicalDomainError(
                    f"Protected sqrt domain violation: input value ({val}) < 0 at index {t}."
                )
            res = math.sqrt(val)
        elif self.operator == MathematicalOperator.ABS:
            res = abs(child_vals[0])
        else:
            raise MathematicalEvaluationError(f"Unhandled operator: {self.operator}")

        if math.isnan(res) or math.isinf(res):
            raise MathematicalEvaluationError(
                f"Non-finite evaluation result ({res}) for operator {self.operator.value} at index {t}."
            )

        return float(res)

    def _evaluate_series(self, data: pd.DataFrame) -> pd.Series:
        """Evaluate full Series over DataFrame."""
        if self.operator == MathematicalOperator.CONSTANT:
            return pd.Series(self.constant_value, index=data.index, dtype=float)

        if self.operator == MathematicalOperator.FEATURE:
            if self.feature_name not in data.columns:
                raise MathematicalEvaluationError(f"Feature column '{self.feature_name}' missing from input data.")
            s = data[self.feature_name].astype(float)
            if s.isna().any() or np.isinf(s).any():
                raise MathematicalEvaluationError(
                    f"Non-finite values encountered in input feature column '{self.feature_name}'."
                )
            return s

        child_series = [c.evaluate(data, t=None) for c in self.children]

        if self.operator == MathematicalOperator.ADD:
            res = child_series[0] + child_series[1]
        elif self.operator == MathematicalOperator.SUB:
            res = child_series[0] - child_series[1]
        elif self.operator == MathematicalOperator.MUL:
            res = child_series[0] * child_series[1]
        elif self.operator == MathematicalOperator.PROTECTED_DIV:
            denom = child_series[1]
            # Check domain violation on valid post-warmup slice for denominator
            valid_denom = denom.iloc[self.max_lookback:]
            if (valid_denom.abs() < 1e-12).any():
                raise MathematicalDomainError("Protected division by zero or near-zero denominator encountered in Series.")
            res = child_series[0] / denom
        elif self.operator == MathematicalOperator.NEG:
            res = -child_series[0]
        elif self.operator == MathematicalOperator.PROTECTED_LOG:
            val = child_series[0]
            valid_val = val.iloc[self.max_lookback:]
            if (valid_val <= 0.0).any():
                raise MathematicalDomainError("Protected log domain violation: non-positive value in Series.")
            res = np.log(val)
        elif self.operator == MathematicalOperator.PROTECTED_SQRT:
            val = child_series[0]
            valid_val = val.iloc[self.max_lookback:]
            if (valid_val < 0.0).any():
                raise MathematicalDomainError("Protected sqrt domain violation: negative value in Series.")
            res = np.sqrt(val)
        elif self.operator == MathematicalOperator.ABS:
            res = child_series[0].abs()
        else:
            raise MathematicalEvaluationError(f"Unhandled operator: {self.operator}")

        return res


@dataclass(frozen=True)
class MathematicalSearchSpace:
    """Canonical immutable Search-Space Constitution for Mathematical Discovery.

    Enforces structural, temporal, complexity, constant, and lineage search budget constraints.
    """

    search_id: str
    version: str = "1.0"
    allowed_operators: Tuple[MathematicalOperator, ...] = field(
        default_factory=lambda: tuple(MathematicalOperator)
    )
    allowed_features: Tuple[str, ...] = ()
    max_depth: int = 5
    max_node_count: int = 15
    max_feature_count: int = 4
    max_interaction_count: int = 2
    min_lag: int = 0
    max_lag: int = 10
    max_window_size: int = 50
    constant_bounds: Tuple[float, float] = (-100.0, 100.0)
    constant_precision: float = 1e-4
    max_complexity: int = 20
    generator_id: str = "manual_search_space"
    generator_version: str = "1.0"
    random_seed: int = 0
    dataset_scope: DatasetScope | None = None
    execution_assumptions: ExecutionAssumptions | None = None
    code_provenance: CodeProvenance | None = None

    def __post_init__(self) -> None:
        """Enforce strict fail-closed search space constitutional invariants."""
        if not self.search_id or not isinstance(self.search_id, str) or not self.search_id.strip():
            raise SearchSpaceValidationError("search_id must be a non-empty string.")
        if not self.version or not isinstance(self.version, str) or not self.version.strip():
            raise SearchSpaceValidationError("version must be a non-empty string.")

        # Normalize allowed_operators
        if not isinstance(self.allowed_operators, tuple):
            if isinstance(self.allowed_operators, (list, Sequence)):
                ops = tuple(
                    MathematicalOperator(op) if isinstance(op, str) else op
                    for op in self.allowed_operators
                )
                object.__setattr__(self, "allowed_operators", ops)
            else:
                raise SearchSpaceValidationError("allowed_operators must be a tuple or sequence")

        if not self.allowed_operators:
            raise SearchSpaceValidationError("allowed_operators cannot be empty.")

        for op in self.allowed_operators:
            if not isinstance(op, MathematicalOperator):
                raise SearchSpaceValidationError(f"Invalid operator in allowed_operators: {op}")

        # Normalize allowed_features
        if not isinstance(self.allowed_features, tuple):
            if isinstance(self.allowed_features, (list, Sequence)):
                object.__setattr__(
                    self, "allowed_features", tuple(sorted({f.strip() for f in self.allowed_features if f and f.strip()}))
                )
            else:
                raise SearchSpaceValidationError("allowed_features must be a tuple or sequence")

        # Validate numeric bounds and limits
        if not isinstance(self.max_depth, int) or self.max_depth < 1:
            raise SearchSpaceValidationError(f"max_depth must be an integer >= 1, got {self.max_depth}")
        if not isinstance(self.max_node_count, int) or self.max_node_count < 1:
            raise SearchSpaceValidationError(f"max_node_count must be an integer >= 1, got {self.max_node_count}")
        if not isinstance(self.max_feature_count, int) or self.max_feature_count < 1:
            raise SearchSpaceValidationError(
                f"max_feature_count must be an integer >= 1, got {self.max_feature_count}"
            )
        if not isinstance(self.max_interaction_count, int) or self.max_interaction_count < 0:
            raise SearchSpaceValidationError(
                f"max_interaction_count must be an integer >= 0, got {self.max_interaction_count}"
            )

        if not isinstance(self.min_lag, int) or self.min_lag < 0:
            raise SearchSpaceValidationError(f"min_lag must be an integer >= 0, got {self.min_lag}")
        if not isinstance(self.max_lag, int) or self.max_lag < self.min_lag:
            raise SearchSpaceValidationError(
                f"max_lag ({self.max_lag}) must be an integer >= min_lag ({self.min_lag})"
            )
        if not isinstance(self.max_window_size, int) or self.max_window_size < 1:
            raise SearchSpaceValidationError(
                f"max_window_size must be an integer >= 1, got {self.max_window_size}"
            )

        # Constant bounds
        if not isinstance(self.constant_bounds, tuple) or len(self.constant_bounds) != 2:
            raise SearchSpaceValidationError("constant_bounds must be a tuple of (min_val, max_val)")
        c_min, c_max = float(self.constant_bounds[0]), float(self.constant_bounds[1])
        if math.isnan(c_min) or math.isinf(c_min) or math.isnan(c_max) or math.isinf(c_max):
            raise SearchSpaceValidationError("constant_bounds cannot contain NaN or Inf values")
        if c_min > c_max:
            raise SearchSpaceValidationError(
                f"constant_bounds min_val ({c_min}) cannot be greater than max_val ({c_max})"
            )
        object.__setattr__(self, "constant_bounds", (c_min, c_max))

        if not isinstance(self.constant_precision, (int, float)) or self.constant_precision <= 0:
            raise SearchSpaceValidationError(
                f"constant_precision must be a positive float, got {self.constant_precision}"
            )

        if not isinstance(self.max_complexity, int) or self.max_complexity < 1:
            raise SearchSpaceValidationError(
                f"max_complexity must be an integer >= 1, got {self.max_complexity}"
            )

        # Validate generator metadata
        if not self.generator_id or not isinstance(self.generator_id, str) or not self.generator_id.strip():
            raise SearchSpaceValidationError("generator_id must be a non-empty string.")
        if not self.generator_version or not isinstance(self.generator_version, str) or not self.generator_version.strip():
            raise SearchSpaceValidationError("generator_version must be a non-empty string.")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise SearchSpaceValidationError("random_seed must be an integer.")

        # Validate research identity
        if self.dataset_scope is not None and not isinstance(self.dataset_scope, DatasetScope):
            raise SearchSpaceValidationError(
                f"dataset_scope must be instance of DatasetScope, got {type(self.dataset_scope)}"
            )
        if self.execution_assumptions is not None and not isinstance(self.execution_assumptions, ExecutionAssumptions):
            raise SearchSpaceValidationError(
                f"execution_assumptions must be instance of ExecutionAssumptions, got {type(self.execution_assumptions)}"
            )
        if self.code_provenance is not None and not isinstance(self.code_provenance, CodeProvenance):
            raise SearchSpaceValidationError(
                f"code_provenance must be instance of CodeProvenance, got {type(self.code_provenance)}"
            )

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Convert search space to deterministic, ordered canonical dictionary."""
        return {
            "search_id": self.search_id,
            "version": self.version,
            "allowed_operators": sorted([op.value for op in self.allowed_operators]),
            "allowed_features": list(self.allowed_features),
            "max_depth": self.max_depth,
            "max_node_count": self.max_node_count,
            "max_feature_count": self.max_feature_count,
            "max_interaction_count": self.max_interaction_count,
            "min_lag": self.min_lag,
            "max_lag": self.max_lag,
            "max_window_size": self.max_window_size,
            "constant_bounds": [f"{self.constant_bounds[0]:.12g}", f"{self.constant_bounds[1]:.12g}"],
            "constant_precision": f"{self.constant_precision:.12g}",
            "max_complexity": self.max_complexity,
            "generator_id": self.generator_id,
            "generator_version": self.generator_version,
            "random_seed": self.random_seed,
            "dataset_scope": (
                {
                    "dataset_id": self.dataset_scope.dataset_id,
                    "symbol": self.dataset_scope.symbol,
                    "timeframe": self.dataset_scope.timeframe,
                    "start_date": self.dataset_scope.start_date,
                    "end_date": self.dataset_scope.end_date,
                }
                if self.dataset_scope is not None
                else None
            ),
            "execution_assumptions": (
                {
                    "transaction_cost": f"{self.execution_assumptions.transaction_cost:.12g}",
                    "slippage": f"{self.execution_assumptions.slippage:.12g}",
                    "latency_ms": f"{self.execution_assumptions.latency_ms:.12g}",
                }
                if self.execution_assumptions is not None
                else None
            ),
            "code_provenance": (
                {
                    "commit_sha": self.code_provenance.commit_sha,
                    "repository_status": self.code_provenance.repository_status,
                    "author": self.code_provenance.author,
                }
                if self.code_provenance is not None
                else None
            ),
        }

    def to_canonical_json(self) -> str:
        """Serialize search space to deterministic canonical JSON string."""
        return json.dumps(self.to_canonical_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        """Deterministic SHA-256 fingerprint computed across canonical serialization."""
        return hashlib.sha256(self.to_canonical_json().encode("utf-8")).hexdigest()

    @property
    def search_fingerprint(self) -> str:
        """Alias for fingerprint as canonical search space identity."""
        return self.fingerprint

    def validate_expression(self, expr: MathematicalExpression) -> None:
        """Fail closed if an expression violates any constraint of this Search Space."""
        if not isinstance(expr, MathematicalExpression):
            raise SearchSpaceValidationError(f"Expected MathematicalExpression, got {type(expr)}")

        # Check operators recursively
        def _check_node_ops(node: MathematicalExpression) -> None:
            if node.operator not in self.allowed_operators:
                raise SearchSpaceValidationError(
                    f"Expression node uses operator '{node.operator.value}' which is not in search space allowed operators."
                )
            if node.operator == MathematicalOperator.FEATURE and self.allowed_features:
                if node.feature_name not in self.allowed_features:
                    raise SearchSpaceValidationError(
                        f"Expression references feature '{node.feature_name}' which is not in search space allowed features."
                    )
            if node.operator == MathematicalOperator.CONSTANT and node.constant_value is not None:
                c = node.constant_value
                c_min, c_max = self.constant_bounds
                if c < c_min or c > c_max:
                    raise SearchSpaceValidationError(
                        f"Constant value {c} is out of search space bounds ({c_min}, {c_max})."
                    )

            if node.lag < self.min_lag or node.lag > self.max_lag:
                raise SearchSpaceValidationError(
                    f"Expression node lag {node.lag} is outside search space lag bounds ({self.min_lag}, {self.max_lag})."
                )

            for child in node.children:
                _check_node_ops(child)

        _check_node_ops(expr)

        # Check global complexity and depth constraints
        if expr.depth > self.max_depth:
            raise SearchSpaceValidationError(
                f"Expression depth {expr.depth} exceeds search space limit of {self.max_depth}."
            )

        if expr.node_count > self.max_node_count:
            raise SearchSpaceValidationError(
                f"Expression node count {expr.node_count} exceeds search space limit of {self.max_node_count}."
            )

        if expr.feature_count > self.max_feature_count:
            raise SearchSpaceValidationError(
                f"Expression feature count {expr.feature_count} exceeds search space limit of {self.max_feature_count}."
            )

        if expr.interaction_count > self.max_interaction_count:
            raise SearchSpaceValidationError(
                f"Expression interaction count {expr.interaction_count} exceeds search space limit of {self.max_interaction_count}."
            )

        if expr.node_count > self.max_complexity:
            raise SearchSpaceValidationError(
                f"Expression complexity {expr.node_count} exceeds search space max_complexity limit of {self.max_complexity}."
            )

        # Check research lineage alignment if specified on search space
        if self.dataset_scope is not None and expr.dataset_scope is not None:
            if self.dataset_scope != expr.dataset_scope:
                raise SearchSpaceValidationError(
                    "Expression DatasetScope does not match search space DatasetScope."
                )

        if self.execution_assumptions is not None and expr.execution_assumptions is not None:
            if self.execution_assumptions != expr.execution_assumptions:
                raise SearchSpaceValidationError(
                    "Expression ExecutionAssumptions do not match search space ExecutionAssumptions."
                )

        if self.code_provenance is not None and expr.code_provenance is not None:
            if self.code_provenance != expr.code_provenance:
                raise SearchSpaceValidationError(
                    "Expression CodeProvenance does not match search space CodeProvenance."
                )
