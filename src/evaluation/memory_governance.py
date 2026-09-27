"""Memory-Aware Discovery Governance for Project 1.

Provides deterministic, evidence-backed governance checking for candidate research trials prior to
expensive experiment execution.

Consults canonical ResearchLearningMemory (specifically active DoNotRepeatConstraint records)
and determines whether an active, verified constraint applies to the candidate being considered.

Design Principles:
- Informational lessons (ResearchLesson) do NOT block candidates.
- Only explicit active DoNotRepeatConstraint records may block a candidate.
- Deterministic evaluation and SHA-256 governance fingerprinting.
- Fail closed on malformed or ambiguous constraint data.
- Complete auditability and provenance retention.
- No mutation of production code, strategy runtime, or live execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from src.evaluation.candidate_generator import CandidateSpec
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
    ResearchExperimentSpec,
)
from src.evaluation.research_registry import (
    DoNotRepeatConstraint,
    _compute_scope_id,
)


class MemoryGovernanceDecision(str, Enum):
    """Explicit decision made by the Memory-Aware Discovery Governance stage."""

    ALLOWED = "ALLOWED"
    BLOCKED = "BLOCKED"
    FAIL_CLOSED = "FAIL_CLOSED"


@dataclass(frozen=True)
class MemoryConstraintMatch:
    """Detailed record of a matched or evaluated DoNotRepeatConstraint."""

    constraint_id: str
    constraint_fingerprint: str
    pattern_key: str
    reason: str
    source_record_id: str
    experiment_fingerprint: str
    evidence_fingerprint: str | None = None
    learning_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "constraint_fingerprint": self.constraint_fingerprint,
            "pattern_key": self.pattern_key,
            "reason": self.reason,
            "source_record_id": self.source_record_id,
            "experiment_fingerprint": self.experiment_fingerprint,
            "evidence_fingerprint": self.evidence_fingerprint,
            "learning_id": self.learning_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryConstraintMatch:
        if not isinstance(data, dict):
            raise TypeError("MemoryConstraintMatch data must be a dictionary.")
        return cls(
            constraint_id=data.get("constraint_id", ""),
            constraint_fingerprint=data.get("constraint_fingerprint", ""),
            pattern_key=data.get("pattern_key", ""),
            reason=data.get("reason", ""),
            source_record_id=data.get("source_record_id", ""),
            experiment_fingerprint=data.get("experiment_fingerprint", ""),
            evidence_fingerprint=data.get("evidence_fingerprint"),
            learning_id=data.get("learning_id"),
        )


@dataclass(frozen=True)
class DiscoveryMemoryGovernanceResult:
    """Canonical result of memory-aware discovery governance evaluation for a candidate/trial."""

    candidate_id: str
    search_id: str | None
    search_fingerprint: str | None
    experiment_fingerprint: str
    decision: MemoryGovernanceDecision
    reason: str
    matching_constraint: MemoryConstraintMatch | None = None
    evaluated_constraints_count: int = 0
    methodology_version: str = "discovery_v1.0"
    governance_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string.")
        if not isinstance(self.decision, MemoryGovernanceDecision):
            if isinstance(self.decision, str) and self.decision in MemoryGovernanceDecision.__members__:
                object.__setattr__(self, "decision", MemoryGovernanceDecision(self.decision))
            else:
                raise TypeError(f"Invalid MemoryGovernanceDecision: '{self.decision}'.")

        semantic_payload = {
            "candidate_id": self.candidate_id,
            "search_id": self.search_id,
            "search_fingerprint": self.search_fingerprint,
            "experiment_fingerprint": self.experiment_fingerprint,
            "decision": self.decision.value,
            "reason": self.reason,
            "matching_constraint": self.matching_constraint.as_dict() if self.matching_constraint else None,
            "evaluated_constraints_count": self.evaluated_constraints_count,
            "methodology_version": self.methodology_version,
        }
        serialized = json.dumps(semantic_payload, sort_keys=True, separators=(",", ":"))
        fp = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        object.__setattr__(self, "governance_fingerprint", fp)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "search_id": self.search_id,
            "search_fingerprint": self.search_fingerprint,
            "experiment_fingerprint": self.experiment_fingerprint,
            "decision": self.decision.value,
            "reason": self.reason,
            "matching_constraint": self.matching_constraint.as_dict() if self.matching_constraint else None,
            "evaluated_constraints_count": self.evaluated_constraints_count,
            "methodology_version": self.methodology_version,
            "governance_fingerprint": self.governance_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscoveryMemoryGovernanceResult:
        if not isinstance(data, dict):
            raise TypeError("DiscoveryMemoryGovernanceResult data must be a dictionary.")

        decision_str = data.get("decision")
        if not decision_str or decision_str not in MemoryGovernanceDecision.__members__:
            raise ValueError(f"Invalid decision string in data: '{decision_str}'.")

        match_dict = data.get("matching_constraint")
        match_obj = MemoryConstraintMatch.from_dict(match_dict) if match_dict else None

        return cls(
            candidate_id=data.get("candidate_id", ""),
            search_id=data.get("search_id"),
            search_fingerprint=data.get("search_fingerprint"),
            experiment_fingerprint=data.get("experiment_fingerprint", ""),
            decision=MemoryGovernanceDecision(decision_str),
            reason=data.get("reason", ""),
            matching_constraint=match_obj,
            evaluated_constraints_count=int(data.get("evaluated_constraints_count", 0)),
            methodology_version=data.get("methodology_version", "discovery_v1.0"),
        )


def evaluate_candidate_memory_governance(
    *,
    candidate: CandidateSpec,
    dataset_scope: DatasetScope,
    execution_assumptions: ExecutionAssumptions,
    code_provenance: CodeProvenance,
    active_constraints: Sequence[DoNotRepeatConstraint],
    search_id: str | None = None,
    search_fingerprint: str | None = None,
    methodology_version: str = "discovery_v1.0",
) -> DiscoveryMemoryGovernanceResult:
    """Perform deterministic, fail-closed memory governance evaluation for a research candidate.

    Consults active DoNotRepeatConstraint records to determine whether any verified
    prohibition constraint applies to the candidate.
    """
    if not isinstance(candidate, CandidateSpec):
        raise TypeError("candidate must be a CandidateSpec instance.")
    if not isinstance(dataset_scope, DatasetScope):
        raise TypeError("dataset_scope must be a DatasetScope instance.")
    if not isinstance(execution_assumptions, ExecutionAssumptions):
        raise TypeError("execution_assumptions must be an ExecutionAssumptions instance.")
    if not isinstance(code_provenance, CodeProvenance):
        raise TypeError("code_provenance must be a CodeProvenance instance.")

    hypothesis_stmt = (
        candidate.hypothesis_template.replace("{candidate_id}", candidate.candidate_id)
        if candidate.hypothesis_template
        else f"Hypothesis for candidate {candidate.candidate_id}"
    )

    spec = ResearchExperimentSpec(
        hypothesis=hypothesis_stmt,
        methodology_version=methodology_version,
        strategy_name=candidate.strategy_name,
        strategy_version="1.0.0",
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        benchmark_reference="buy_and_hold",
        parameters=dict(candidate.parameters),
        random_seed=candidate.random_seed,
    )
    cand_exp_fp = spec.fingerprint

    # Filter to active constraints and sort deterministically
    raw_active = [c for c in active_constraints if getattr(c, "is_active", True)]
    # Sort deterministically by constraint_id or canonical_fingerprint
    sorted_constraints = sorted(
        raw_active,
        key=lambda c: (
            getattr(c, "constraint_id", ""),
            getattr(c, "canonical_fingerprint", ""),
        ),
    )

    evaluated_count = 0
    strat_name = candidate.strategy_name
    symbol = dataset_scope.symbol
    tf = dataset_scope.timeframe
    ds_id = _compute_scope_id(dataset_scope)
    cand_id = candidate.candidate_id

    candidate_patterns = {
        cand_id,
        f"candidate:{cand_id}",
        f"fail_pattern:{strat_name}:{symbol}:{tf}:{ds_id[:8]}",
        f"fail_pattern:{strat_name}:{symbol}:{tf}",
        f"{strat_name}:{symbol}:{tf}",
        f"strategy:{strat_name}",
    }
    for p_key, p_val in candidate.parameters.items():
        candidate_patterns.add(f"param:{p_key}={p_val}")
        candidate_patterns.add(f"{p_key}={p_val}")

    for c in sorted_constraints:
        # Fail closed on malformed constraint structures
        if not isinstance(c, DoNotRepeatConstraint):
            return DiscoveryMemoryGovernanceResult(
                candidate_id=cand_id,
                search_id=search_id,
                search_fingerprint=search_fingerprint,
                experiment_fingerprint=cand_exp_fp,
                decision=MemoryGovernanceDecision.FAIL_CLOSED,
                reason=f"Malformed constraint in governance input: object is not DoNotRepeatConstraint ({type(c).__name__}).",
                matching_constraint=None,
                evaluated_constraints_count=evaluated_count,
                methodology_version=methodology_version,
            )

        if not c.constraint_id or not c.constraint_id.strip():
            return DiscoveryMemoryGovernanceResult(
                candidate_id=cand_id,
                search_id=search_id,
                search_fingerprint=search_fingerprint,
                experiment_fingerprint=cand_exp_fp,
                decision=MemoryGovernanceDecision.FAIL_CLOSED,
                reason="Malformed constraint in governance input: missing or empty constraint_id.",
                matching_constraint=None,
                evaluated_constraints_count=evaluated_count,
                methodology_version=methodology_version,
            )

        if not isinstance(c.confidence_score, (int, float)) or math.isnan(c.confidence_score) or not (0.0 <= c.confidence_score <= 1.0):
            return DiscoveryMemoryGovernanceResult(
                candidate_id=cand_id,
                search_id=search_id,
                search_fingerprint=search_fingerprint,
                experiment_fingerprint=cand_exp_fp,
                decision=MemoryGovernanceDecision.FAIL_CLOSED,
                reason=f"Malformed constraint '{c.constraint_id}': invalid confidence_score '{c.confidence_score}'. Expected float between 0.0 and 1.0.",
                matching_constraint=None,
                evaluated_constraints_count=evaluated_count,
                methodology_version=methodology_version,
            )

        c_exp_fp = (c.experiment_fingerprint or "").strip()
        c_pk = (c.pattern_key or "").strip()

        if not c_exp_fp and not c_pk:
            return DiscoveryMemoryGovernanceResult(
                candidate_id=cand_id,
                search_id=search_id,
                search_fingerprint=search_fingerprint,
                experiment_fingerprint=cand_exp_fp,
                decision=MemoryGovernanceDecision.FAIL_CLOSED,
                reason=f"Ambiguous constraint '{c.constraint_id}': both experiment_fingerprint and pattern_key are empty.",
                matching_constraint=None,
                evaluated_constraints_count=evaluated_count,
                methodology_version=methodology_version,
            )

        evaluated_count += 1
        matched = False
        match_reason = ""

        # Match 1: Direct experiment fingerprint match
        if c_exp_fp and c_exp_fp == cand_exp_fp:
            matched = True
            match_reason = f"Direct experiment fingerprint match ({c_exp_fp})"

        # Match 2: Pattern key in candidate patterns
        if not matched and c_pk and c_pk in candidate_patterns:
            matched = True
            match_reason = f"Pattern key match ({c_pk})"

        # Match 3: Colon-separated token matching for fail_pattern:...
        if not matched and c_pk and c_pk.startswith("fail_pattern:"):
            tokens = c_pk.split(":")
            if len(tokens) >= 4:
                pk_strat, pk_sym, pk_tf = tokens[1], tokens[2], tokens[3]
                if pk_strat == strat_name and pk_sym == symbol and pk_tf == tf:
                    if len(tokens) >= 5:
                        pk_ds = tokens[4]
                        if ds_id.startswith(pk_ds) or pk_ds.startswith(ds_id[:8]):
                            matched = True
                            match_reason = f"Structured fail_pattern match ({c_pk}) on strategy, symbol, timeframe, and dataset scope"
                    else:
                        matched = True
                        match_reason = f"Structured fail_pattern match ({c_pk}) on strategy, symbol, and timeframe"

        if matched:
            match_info = MemoryConstraintMatch(
                constraint_id=c.constraint_id,
                constraint_fingerprint=c.canonical_fingerprint,
                pattern_key=c.pattern_key,
                reason=c.reason,
                source_record_id=c.source_record_id,
                experiment_fingerprint=c.experiment_fingerprint,
                evidence_fingerprint=c.evidence_fingerprint,
                learning_id=c.superseded_by_learning_id,
            )
            return DiscoveryMemoryGovernanceResult(
                candidate_id=cand_id,
                search_id=search_id,
                search_fingerprint=search_fingerprint,
                experiment_fingerprint=cand_exp_fp,
                decision=MemoryGovernanceDecision.BLOCKED,
                reason=f"Candidate '{cand_id}' blocked by active DoNotRepeatConstraint '{c.constraint_id}': {c.reason}. ({match_reason})",
                matching_constraint=match_info,
                evaluated_constraints_count=evaluated_count,
                methodology_version=methodology_version,
            )

    return DiscoveryMemoryGovernanceResult(
        candidate_id=cand_id,
        search_id=search_id,
        search_fingerprint=search_fingerprint,
        experiment_fingerprint=cand_exp_fp,
        decision=MemoryGovernanceDecision.ALLOWED,
        reason="No applicable active DoNotRepeatConstraint matched candidate.",
        matching_constraint=None,
        evaluated_constraints_count=evaluated_count,
        methodology_version=methodology_version,
    )
