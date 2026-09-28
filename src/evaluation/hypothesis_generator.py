"""Canonical Knowledge-to-Hypothesis Generator for Project 1.

Transforms eligible, validated research knowledge patterns (ResearchKnowledgePattern)
into explicit, machine-readable, traceable research hypotheses (ResearchHypothesis).

Design Principles:
- DISCOVERY / RESEARCH capability ONLY.
- Does NOT produce live signals, trading decisions, live strategies, production parameters,
  orders, Project 2 payloads, or execution instructions.
- Preserves full provenance and lineage from source knowledge to source evidence.
- Fails closed when required lineage, DatasetScope, ExecutionAssumptions, or CodeProvenance are missing/invalid.
- Deterministic hypothesis identity and deduplication.
- Auditable hypothesis lifecycle transitions (GENERATED, ACCEPTED_FOR_RESEARCH, REJECTED, SUPERSEDED).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping, Sequence

from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
    HypothesisStatus,
    ResearchHypothesis,
    WalkForwardProtocol,
)
from src.evaluation.research_knowledge import (
    ResearchKnowledgePattern,
    ResearchPatternCategory,
)

GENERATOR_METHODOLOGY_VERSION = "knowledge_hypothesis_v1.0"
GENERATOR_VERSION = "1.0.0"


class HypothesisGenerationError(ValueError):
    """Raised when hypothesis generation encounters missing or invalid inputs."""


@dataclass(frozen=True)
class KnowledgeHypothesisGeneratorPolicy:
    """Policy governing controlled hypothesis generation from research knowledge."""

    allow_contradictory_knowledge: bool = False
    allow_inconclusive_knowledge: bool = False
    require_evidence_lineage: bool = True
    default_benchmark_reference: str = "BUY_AND_HOLD"
    generator_version: str = GENERATOR_VERSION
    methodology_version: str = GENERATOR_METHODOLOGY_VERSION


@dataclass(frozen=True)
class HypothesisGenerationContext:
    """Explicit context providing dataset scope, execution assumptions, and code provenance."""

    dataset_scope: DatasetScope
    execution_assumptions: ExecutionAssumptions
    code_provenance: CodeProvenance
    benchmark_reference: str = "BUY_AND_HOLD"
    parameters_override: dict[str, Any] = field(default_factory=dict)
    walk_forward_protocol: WalkForwardProtocol | None = None
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_scope, DatasetScope):
            raise TypeError("dataset_scope must be a DatasetScope instance.")
        if not isinstance(self.execution_assumptions, ExecutionAssumptions):
            raise TypeError("execution_assumptions must be an ExecutionAssumptions instance.")
        if not isinstance(self.code_provenance, CodeProvenance):
            raise TypeError("code_provenance must be a CodeProvenance instance.")
        if not self.benchmark_reference or not self.benchmark_reference.strip():
            raise ValueError("benchmark_reference must be a non-empty string.")


class KnowledgeHypothesisGenerator:
    """Controlled transformation service: Validated Research Knowledge -> Governed Research Hypothesis."""

    def __init__(
        self,
        policy: KnowledgeHypothesisGeneratorPolicy | None = None,
    ) -> None:
        self.policy = policy or KnowledgeHypothesisGeneratorPolicy()

    def generate(
        self,
        knowledge: Sequence[ResearchKnowledgePattern],
        *,
        context: HypothesisGenerationContext,
    ) -> tuple[ResearchHypothesis, ...]:
        """Transform eligible validated knowledge patterns into governed research hypotheses.

        Fails closed on missing or invalid inputs. Preserves complete lineage.
        Deduplicates equivalent hypotheses deterministically.
        """
        if not isinstance(context, HypothesisGenerationContext):
            raise HypothesisGenerationError("context must be a HypothesisGenerationContext instance.")

        if not isinstance(knowledge, (list, tuple)):
            raise HypothesisGenerationError("knowledge must be a sequence of ResearchKnowledgePattern instances.")

        if not knowledge:
            raise HypothesisGenerationError("Cannot generate hypothesis from empty knowledge sequence.")

        # Fail closed on missing/incomplete context provenance
        self._validate_context_provenance(context)

        # Filter eligible knowledge patterns
        eligible_patterns: list[ResearchKnowledgePattern] = []
        for pat in knowledge:
            if not isinstance(pat, ResearchKnowledgePattern):
                raise HypothesisGenerationError(
                    f"Invalid item in knowledge sequence: expected ResearchKnowledgePattern, got {type(pat).__name__}."
                )
            if not pat.is_active or pat.superseded_by_pattern_id is not None:
                continue
            if pat.is_contradictory and not self.policy.allow_contradictory_knowledge:
                continue
            if pat.category == ResearchPatternCategory.INCONCLUSIVE_PATTERN and not self.policy.allow_inconclusive_knowledge:
                continue
            if self.policy.require_evidence_lineage and not pat.supporting_evidence_fingerprints:
                raise HypothesisGenerationError(
                    f"Knowledge pattern '{pat.pattern_id}' lacks required supporting evidence lineage."
                )
            eligible_patterns.append(pat)

        if not eligible_patterns:
            raise HypothesisGenerationError("No eligible knowledge patterns found for hypothesis generation.")

        # Sort patterns deterministically by pattern_id
        sorted_patterns = sorted(eligible_patterns, key=lambda p: p.pattern_id)

        # Group patterns by strategy name to create focused hypotheses
        grouped_by_strategy: dict[str, list[ResearchKnowledgePattern]] = {}
        for pat in sorted_patterns:
            strat = pat.normalized_conditions.strategy_name or "unknown_strategy"
            grouped_by_strategy.setdefault(strat, []).append(pat)

        generated_hypotheses: list[ResearchHypothesis] = []
        now_utc = datetime.now(timezone.utc).isoformat()

        for strat_name in sorted(grouped_by_strategy.keys()):
            pats = grouped_by_strategy[strat_name]

            # Aggregate lineage
            sup_k_ids = tuple(sorted(set(p.pattern_id for p in pats)))
            sup_ev_ids = tuple(sorted(set(ev for p in pats for ev in p.supporting_evidence_fingerprints)))

            if not sup_k_ids:
                raise HypothesisGenerationError("Missing source knowledge lineage.")
            if self.policy.require_evidence_lineage and not sup_ev_ids:
                raise HypothesisGenerationError("Missing source evidence lineage.")

            # Formulate structured statement
            cat_counts = {}
            for p in pats:
                cat_counts[p.category.value] = cat_counts.get(p.category.value, 0) + 1

            statements = [p.statement for p in pats]
            combined_stmt = (
                f"Governed Research Hypothesis for strategy '{strat_name}' "
                f"derived from knowledge patterns [{', '.join(sup_k_ids)}]: "
                f"{'; '.join(statements)}"
            )

            # Resolve parameters
            params = dict(context.parameters_override)
            constraints_dict = {
                "source_pattern_categories": [p.category.value for p in pats],
                "observation_counts": [p.observation_count for p in pats],
            }

            hyp = ResearchHypothesis(
                statement=combined_stmt,
                methodology_version=self.policy.methodology_version,
                strategy_name=strat_name,
                strategy_version=pats[0].normalized_conditions.strategy_version or "1.0.0",
                dataset_scope=context.dataset_scope,
                execution_assumptions=context.execution_assumptions,
                code_provenance=context.code_provenance,
                benchmark_reference=context.benchmark_reference,
                parameters=params,
                random_seed=context.random_seed,
                walk_forward_protocol=context.walk_forward_protocol,
                source_knowledge_ids=sup_k_ids,
                source_evidence_ids=sup_ev_ids,
                hypothesis_version="1.0",
                generation_method="KNOWLEDGE_PATTERN_TRANSFORMATION",
                generator_version=self.policy.generator_version,
                constraints=constraints_dict,
                status=HypothesisStatus.GENERATED,
                created_at_utc=now_utc,
            )
            generated_hypotheses.append(hyp)

        # Deduplicate hypotheses deterministically by canonical_hypothesis_fingerprint
        unique_hypotheses: dict[str, ResearchHypothesis] = {}
        for h in generated_hypotheses:
            if h.canonical_hypothesis_fingerprint not in unique_hypotheses:
                unique_hypotheses[h.canonical_hypothesis_fingerprint] = h

        return tuple(sorted(unique_hypotheses.values(), key=lambda h: h.hypothesis_id))

    def _validate_context_provenance(self, context: HypothesisGenerationContext) -> None:
        """Fail closed if context provenance or assumptions are missing or incomplete."""
        ds = context.dataset_scope
        ea = context.execution_assumptions
        cp = context.code_provenance

        if not ds or not ds.dataset_id or not ds.symbol or not ds.timeframe:
            raise HypothesisGenerationError("Incomplete DatasetScope in hypothesis generation context.")

        if not ea or ea.transaction_cost < 0.0 or ea.slippage < 0.0 or ea.latency_ms < 0.0:
            raise HypothesisGenerationError("Invalid ExecutionAssumptions in hypothesis generation context.")

        if not cp or not cp.commit_sha or not cp.commit_sha.strip():
            raise HypothesisGenerationError("Incomplete CodeProvenance in hypothesis generation context.")


def accept_hypothesis_for_research(hypothesis: ResearchHypothesis) -> ResearchHypothesis:
    """Transition hypothesis state from GENERATED to ACCEPTED_FOR_RESEARCH.

    Does NOT promote to production or create live signals.
    """
    if not isinstance(hypothesis, ResearchHypothesis):
        raise TypeError("hypothesis must be a ResearchHypothesis instance.")

    if hypothesis.status == HypothesisStatus.REJECTED:
        raise ValueError(f"Cannot accept REJECTED hypothesis '{hypothesis.hypothesis_id}' for research.")

    hyp_dict = hypothesis.as_dict()
    hyp_dict["status"] = HypothesisStatus.ACCEPTED_FOR_RESEARCH.value

    return ResearchHypothesis(
        statement=hypothesis.statement,
        methodology_version=hypothesis.methodology_version,
        strategy_name=hypothesis.strategy_name,
        strategy_version=hypothesis.strategy_version,
        dataset_scope=hypothesis.dataset_scope,
        execution_assumptions=hypothesis.execution_assumptions,
        code_provenance=hypothesis.code_provenance,
        benchmark_reference=hypothesis.benchmark_reference,
        parameters=dict(hypothesis.parameters),
        random_seed=hypothesis.random_seed,
        walk_forward_protocol=hypothesis.walk_forward_protocol,
        source_knowledge_ids=hypothesis.source_knowledge_ids,
        source_evidence_ids=hypothesis.source_evidence_ids,
        hypothesis_version=hypothesis.hypothesis_version,
        generation_method=hypothesis.generation_method,
        generator_version=hypothesis.generator_version,
        constraints=dict(hypothesis.constraints),
        status=HypothesisStatus.ACCEPTED_FOR_RESEARCH,
        created_at_utc=hypothesis.created_at_utc,
    )


def reject_hypothesis(hypothesis: ResearchHypothesis, reason: str = "") -> ResearchHypothesis:
    """Transition hypothesis state to REJECTED."""
    if not isinstance(hypothesis, ResearchHypothesis):
        raise TypeError("hypothesis must be a ResearchHypothesis instance.")

    constraints = dict(hypothesis.constraints)
    if reason:
        constraints["rejection_reason"] = reason

    return ResearchHypothesis(
        statement=hypothesis.statement,
        methodology_version=hypothesis.methodology_version,
        strategy_name=hypothesis.strategy_name,
        strategy_version=hypothesis.strategy_version,
        dataset_scope=hypothesis.dataset_scope,
        execution_assumptions=hypothesis.execution_assumptions,
        code_provenance=hypothesis.code_provenance,
        benchmark_reference=hypothesis.benchmark_reference,
        parameters=dict(hypothesis.parameters),
        random_seed=hypothesis.random_seed,
        walk_forward_protocol=hypothesis.walk_forward_protocol,
        source_knowledge_ids=hypothesis.source_knowledge_ids,
        source_evidence_ids=hypothesis.source_evidence_ids,
        hypothesis_version=hypothesis.hypothesis_version,
        generation_method=hypothesis.generation_method,
        generator_version=hypothesis.generator_version,
        constraints=constraints,
        status=HypothesisStatus.REJECTED,
        created_at_utc=hypothesis.created_at_utc,
    )


def supersede_hypothesis(hypothesis: ResearchHypothesis, superseding_id: str) -> ResearchHypothesis:
    """Transition hypothesis state to SUPERSEDED."""
    if not isinstance(hypothesis, ResearchHypothesis):
        raise TypeError("hypothesis must be a ResearchHypothesis instance.")

    constraints = dict(hypothesis.constraints)
    constraints["superseded_by"] = superseding_id

    return ResearchHypothesis(
        statement=hypothesis.statement,
        methodology_version=hypothesis.methodology_version,
        strategy_name=hypothesis.strategy_name,
        strategy_version=hypothesis.strategy_version,
        dataset_scope=hypothesis.dataset_scope,
        execution_assumptions=hypothesis.execution_assumptions,
        code_provenance=hypothesis.code_provenance,
        benchmark_reference=hypothesis.benchmark_reference,
        parameters=dict(hypothesis.parameters),
        random_seed=hypothesis.random_seed,
        walk_forward_protocol=hypothesis.walk_forward_protocol,
        source_knowledge_ids=hypothesis.source_knowledge_ids,
        source_evidence_ids=hypothesis.source_evidence_ids,
        hypothesis_version=hypothesis.hypothesis_version,
        generation_method=hypothesis.generation_method,
        generator_version=hypothesis.generator_version,
        constraints=constraints,
        status=HypothesisStatus.SUPERSEDED,
        created_at_utc=hypothesis.created_at_utc,
    )
