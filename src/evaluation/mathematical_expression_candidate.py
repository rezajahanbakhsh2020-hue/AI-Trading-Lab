"""Canonical Research-Only Mathematical Expression Candidate & Signal Policy.

Provides immutable representations for mathematical strategy candidates,
canonical sign interpretation policies, search-space pre-validation,
and bridge methods to create governed ResearchHypothesis instances.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import pandas as pd

from src.evaluation.mathematical_expression import (
    MathematicalExpression,
    MathematicalExpressionError,
    MathematicalSearchSpace,
    SearchSpaceValidationError,
    _float_to_lossless_str,
)
from src.evaluation.hypothesis_generator import accept_hypothesis_for_research
from src.evaluation.mathematical_expression_strategy import create_mathematical_research_registry
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
    HypothesisStatus,
    ResearchEvidence,
    ResearchHypothesis,
    WalkForwardProtocol,
)
from src.evaluation.research_runner import DiscoveryCriteria, run_research_experiment


class MathematicalCandidateValidationError(MathematicalExpressionError):
    """Raised when a mathematical candidate fails governance pre-validation or lineage checks."""


@dataclass(frozen=True)
class MathematicalSignalInterpretationPolicy:
    """Canonical immutable sign policy for converting expression outputs to trading signals.

    Canonical Rules:
        expression_value > 0  => LONG (+1)
        expression_value < 0  => SHORT (-1)
        expression_value == 0 => NO TRADE (0)

    Fails closed on NaN, Inf, or non-finite inputs.
    """

    policy_name: str = "SIGN_POLICY"
    version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.policy_name or not isinstance(self.policy_name, str) or not self.policy_name.strip():
            raise MathematicalCandidateValidationError("policy_name must be a non-empty string.")
        if not self.version or not isinstance(self.version, str) or not self.version.strip():
            raise MathematicalCandidateValidationError("version must be a non-empty string.")

    def evaluate_value(self, val: float) -> int:
        """Evaluate a single numeric value into signal {-1, 0, 1}."""
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            raise MathematicalCandidateValidationError(f"Non-finite evaluation value rejected by signal policy: {val}")

        if f_val > 0.0:
            return 1
        elif f_val < 0.0:
            return -1
        else:
            return 0

    def evaluate_series(self, series: pd.Series) -> pd.Series:
        """Evaluate a Series into integer signal series {-1, 0, 1}."""
        if not isinstance(series, pd.Series):
            raise TypeError(f"series must be a pandas Series, got {type(series)}")

        if series.isna().any() or series.map(lambda x: math.isinf(float(x)) if pd.notna(x) else False).any():
            raise MathematicalCandidateValidationError("Non-finite values encountered during Series signal evaluation.")

        signals = pd.Series(0, index=series.index, dtype=int)
        signals.loc[series > 0.0] = 1
        signals.loc[series < 0.0] = -1
        return signals

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Convert policy to deterministic, ordered canonical dictionary."""
        return {
            "policy_name": self.policy_name,
            "version": self.version,
        }

    def to_canonical_json(self) -> str:
        """Serialize policy to deterministic canonical JSON string."""
        return json.dumps(self.to_canonical_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        """Deterministic SHA-256 fingerprint computed across canonical serialization."""
        return hashlib.sha256(self.to_canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_canonical_dict(cls, data: Dict[str, Any]) -> MathematicalSignalInterpretationPolicy:
        """Reconstruct policy from canonical dictionary."""
        return cls(
            policy_name=data.get("policy_name", "SIGN_POLICY"),
            version=data.get("version", "1.0"),
        )


@dataclass(frozen=True)
class MathematicalExpressionCandidate:
    """Research-only mathematical candidate binding AST, search space, and signal policy.

    Guarantees deterministic identity across expression structure, search space constitution,
    signal policy, generator metadata, and complete research lineage.
    """

    expression: MathematicalExpression
    search_space: MathematicalSearchSpace
    signal_policy: MathematicalSignalInterpretationPolicy
    dataset_scope: DatasetScope
    execution_assumptions: ExecutionAssumptions
    code_provenance: CodeProvenance
    generator_id: str = "manual"
    generator_version: str = "1.0"
    random_seed: int = 0
    version: str = "1.0"
    candidate_id: str = field(init=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.expression, MathematicalExpression):
            raise TypeError(f"expression must be MathematicalExpression, got {type(self.expression)}")
        if not isinstance(self.search_space, MathematicalSearchSpace):
            raise TypeError(f"search_space must be MathematicalSearchSpace, got {type(self.search_space)}")
        if not isinstance(self.signal_policy, MathematicalSignalInterpretationPolicy):
            raise TypeError(f"signal_policy must be MathematicalSignalInterpretationPolicy, got {type(self.signal_policy)}")
        if not isinstance(self.dataset_scope, DatasetScope):
            raise TypeError(f"dataset_scope must be DatasetScope, got {type(self.dataset_scope)}")
        if not isinstance(self.execution_assumptions, ExecutionAssumptions):
            raise TypeError(f"execution_assumptions must be ExecutionAssumptions, got {type(self.execution_assumptions)}")
        if not isinstance(self.code_provenance, CodeProvenance):
            raise TypeError(f"code_provenance must be CodeProvenance, got {type(self.code_provenance)}")

        if not self.generator_id or not isinstance(self.generator_id, str) or not self.generator_id.strip():
            raise MathematicalCandidateValidationError("generator_id must be a non-empty string.")
        if not self.generator_version or not isinstance(self.generator_version, str) or not self.generator_version.strip():
            raise MathematicalCandidateValidationError("generator_version must be a non-empty string.")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise MathematicalCandidateValidationError("random_seed must be an integer.")
        if not self.version or not isinstance(self.version, str) or not self.version.strip():
            raise MathematicalCandidateValidationError("version must be a non-empty string.")

        # Compute canonical serialization and fingerprint
        canonical_dict = self.to_canonical_dict()
        serialized = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))
        fp = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        object.__setattr__(self, "fingerprint", fp)
        object.__setattr__(self, "candidate_id", f"math_cand_{fp[:16]}")

    def validate(self) -> None:
        """Perform authoritative pre-execution governance validation.

        Enforces:
        - Expression validation against search space constitutional constraints
        - Strict DatasetScope, ExecutionAssumptions, and CodeProvenance lineage consistency
        """
        # 1. Search space constitution check
        self.search_space.validate_expression(self.expression)

        # 2. Lineage agreement check
        if self.search_space.dataset_scope is not None and self.search_space.dataset_scope != self.dataset_scope:
            raise MathematicalCandidateValidationError(
                f"Candidate DatasetScope ({self.dataset_scope}) does not match SearchSpace DatasetScope ({self.search_space.dataset_scope})."
            )
        if self.expression.dataset_scope is not None and self.expression.dataset_scope != self.dataset_scope:
            raise MathematicalCandidateValidationError(
                f"Candidate DatasetScope ({self.dataset_scope}) does not match Expression DatasetScope ({self.expression.dataset_scope})."
            )

        if self.search_space.execution_assumptions is not None and self.search_space.execution_assumptions != self.execution_assumptions:
            raise MathematicalCandidateValidationError(
                f"Candidate ExecutionAssumptions ({self.execution_assumptions}) does not match SearchSpace ExecutionAssumptions ({self.search_space.execution_assumptions})."
            )
        if self.expression.execution_assumptions is not None and self.expression.execution_assumptions != self.execution_assumptions:
            raise MathematicalCandidateValidationError(
                f"Candidate ExecutionAssumptions ({self.execution_assumptions}) does not match Expression ExecutionAssumptions ({self.expression.execution_assumptions})."
            )

        if self.search_space.code_provenance is not None and self.search_space.code_provenance != self.code_provenance:
            raise MathematicalCandidateValidationError(
                f"Candidate CodeProvenance ({self.code_provenance}) does not match SearchSpace CodeProvenance ({self.search_space.code_provenance})."
            )
        if self.expression.code_provenance is not None and self.expression.code_provenance != self.code_provenance:
            raise MathematicalCandidateValidationError(
                f"Candidate CodeProvenance ({self.code_provenance}) does not match Expression CodeProvenance ({self.expression.code_provenance})."
            )

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Convert candidate to deterministic canonical dictionary."""
        return {
            "version": self.version,
            "expression": self.expression.to_canonical_dict(),
            "expression_fingerprint": self.expression.fingerprint,
            "search_space_fingerprint": self.search_space.fingerprint,
            "signal_policy": self.signal_policy.to_canonical_dict(),
            "signal_policy_fingerprint": self.signal_policy.fingerprint,
            "generator_id": self.generator_id,
            "generator_version": self.generator_version,
            "random_seed": self.random_seed,
            "dataset_scope": {
                "dataset_id": self.dataset_scope.dataset_id,
                "symbol": self.dataset_scope.symbol,
                "timeframe": self.dataset_scope.timeframe,
                "start_date": self.dataset_scope.start_date,
                "end_date": self.dataset_scope.end_date,
            },
            "execution_assumptions": {
                "transaction_cost": _float_to_lossless_str(self.execution_assumptions.transaction_cost),
                "slippage": _float_to_lossless_str(self.execution_assumptions.slippage),
                "latency_ms": _float_to_lossless_str(self.execution_assumptions.latency_ms),
            },
            "code_provenance": {
                "commit_sha": self.code_provenance.commit_sha,
                "repository_status": self.code_provenance.repository_status,
                "author": self.code_provenance.author,
            },
        }

    def to_canonical_json(self) -> str:
        """Serialize candidate to deterministic canonical JSON string."""
        return json.dumps(self.to_canonical_dict(), sort_keys=True, separators=(",", ":"))

    def to_hypothesis(
        self,
        *,
        walk_forward_protocol: WalkForwardProtocol,
        benchmark_reference: str = "buy_and_hold",
        methodology_version: str = "discovery_v1.0",
        strategy_version: str = "1.0.0",
    ) -> ResearchHypothesis:
        """Bridge candidate into governed ResearchHypothesis with GENERATED status.

        Validates search space constraints and lineage prior to constructing hypothesis.
        Fails closed if walk_forward_protocol is missing or not a WalkForwardProtocol instance.
        """
        self.validate()

        if walk_forward_protocol is None or not isinstance(walk_forward_protocol, WalkForwardProtocol):
            raise MathematicalCandidateValidationError(
                "A mathematical research candidate requires an explicit WalkForwardProtocol. Implicit fallback defaults are strictly forbidden."
            )

        wf_protocol = walk_forward_protocol

        statement = (
            f"Mathematical expression candidate [{self.candidate_id}] evaluating "
            f"AST op={self.expression.operator.value} with fingerprint {self.expression.fingerprint[:12]}"
        )

        parameters = {
            "expression_dict": self.expression.to_canonical_dict(),
            "expression_fingerprint": self.expression.fingerprint,
            "search_space_fingerprint": self.search_space.fingerprint,
            "signal_policy_dict": self.signal_policy.to_canonical_dict(),
            "signal_policy_fingerprint": self.signal_policy.fingerprint,
            "candidate_id": self.candidate_id,
            "candidate_fingerprint": self.fingerprint,
            "generator_id": self.generator_id,
            "generator_version": self.generator_version,
            "random_seed": self.random_seed,
        }

        return ResearchHypothesis(
            statement=statement,
            methodology_version=methodology_version,
            strategy_name="mathematical_expression",
            strategy_version=strategy_version,
            dataset_scope=self.dataset_scope,
            execution_assumptions=self.execution_assumptions,
            code_provenance=self.code_provenance,
            benchmark_reference=benchmark_reference,
            parameters=parameters,
            random_seed=self.random_seed,
            walk_forward_protocol=wf_protocol,
            status=HypothesisStatus.GENERATED,
        )


def run_mathematical_research_experiment(
    candidate: MathematicalExpressionCandidate,
    df: pd.DataFrame | None = None,
    *,
    walk_forward_protocol: WalkForwardProtocol,
    criteria: DiscoveryCriteria | None = None,
    benchmark_reference: str = "buy_and_hold",
    persist_evidence: bool = False,
) -> ResearchEvidence:
    """Execute a mathematical research candidate through the existing governed research pipeline.

    Execution Bridge Flow:
        MathematicalExpressionCandidate
            -> pre-execution search space & lineage validation (.validate())
            -> ResearchHypothesis(status=GENERATED)
            -> accept_hypothesis_for_research() => ACCEPTED_FOR_RESEARCH
            -> run_research_experiment(registry=research_registry)
            -> ResearchEvidence

    Guarantees identity invariant tracing:
        expression.fingerprint
            -> candidate.fingerprint
            -> hypothesis.fingerprint
            -> spec.fingerprint
            -> evidence.experiment_fingerprint
    """
    # 1. Pre-execution search space & lineage validation
    candidate.validate()

    # 2. Convert candidate to ResearchHypothesis (status=GENERATED)
    raw_hypothesis = candidate.to_hypothesis(
        benchmark_reference=benchmark_reference,
        walk_forward_protocol=walk_forward_protocol,
    )

    # 3. Transition hypothesis status to ACCEPTED_FOR_RESEARCH via canonical governance entry point
    accepted_hypothesis = accept_hypothesis_for_research(raw_hypothesis)

    # 4. Create dedicated research-only StrategyRegistry
    research_registry = create_mathematical_research_registry()

    # 5. Execute via canonical run_research_experiment
    evidence = run_research_experiment(
        spec=accepted_hypothesis,
        df=df,
        criteria=criteria,
        registry=research_registry,
        persist_evidence=persist_evidence,
    )

    # 6. Verify end-to-end identity invariant traceability
    if evidence.experiment_fingerprint != accepted_hypothesis.fingerprint:
        raise MathematicalCandidateValidationError(
            f"Identity invariant broken: evidence.experiment_fingerprint ({evidence.experiment_fingerprint}) "
            f"mismatches accepted hypothesis fingerprint ({accepted_hypothesis.fingerprint})."
        )

    return evidence
