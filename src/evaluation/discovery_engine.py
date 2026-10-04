"""Research Discovery Engine for Project 1.

Orchestrates candidate generation, chronological dataset partitioning (In-Sample, Validation,
Out-Of-Sample, Walk-Forward), strategy evaluation via canonical research experiment runner,
Research Constitution evidence creation, fail-closed rejection/promotion decisioning, and
optional evidence persistence.

No market data generation, silent substitution, or OOS leakage permitted.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.candidate_generator import CandidateSpec, ResearchSearchSpace
from src.evaluation.discovery_feedback import (
    DiscoveryFeedbackType,
    ResearchDiscoveryFeedback,
    evaluate_candidate_discovery_feedback,
)
from src.evaluation.hypothesis_generator import (
    accept_hypothesis_for_research,
)
from src.evaluation.memory_governance import (
    DiscoveryMemoryGovernanceResult,
    MemoryGovernanceDecision,
    evaluate_candidate_memory_governance,
)
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    EvidencePartitionRole,
    ExecutionAssumptions,
    PromotionStatus,
    RejectionReason,
    ResearchCampaign,
    ResearchCampaignDefinition,
    ResearchCampaignStatus,
    ResearchCandidate,
    ResearchEvidence,
    ResearchExperimentSpec,
    ResearchHypothesis,
    ResearchPlannedTrial,
    ResearchTrialPlan,
    RobustnessCriteria,
    WalkForwardProtocol,
    compute_campaign_fingerprint,
    compute_criteria_fingerprint,
    compute_search_policy_fingerprint,
    resolve_walk_forward_protocol,
)
from src.evaluation.research_knowledge import ResearchKnowledgePattern
from src.evaluation.research_registry import (
    DoNotRepeatConstraint,
    RegistryValidationError,
    ResearchLearningRecord,
    ResearchRegistryRecord,
    ResearchRegistryStore,
    _compute_cp_id,
    _compute_ea_id,
    _compute_scope_id,
    construct_learning_record_from_registry_record,
    construct_registry_record_from_evidence,
)
from src.evaluation.research_robustness import (
    ResearchRobustnessAssessment,
    assess_research_robustness,
)
from src.evaluation.research_runner import (
    run_research_experiment,
    validate_and_prepare_dataset,
)
from src.evaluation.research_store import (
    DEFAULT_CAMPAIGN_DIR,
    DEFAULT_RESEARCH_DIR,
    ResearchCampaignStore,
    save_research_campaign,
    save_research_experiment,
)
from src.evaluation.mathematical_expression import (
    MathematicalSearchSpace,
)
from src.evaluation.mathematical_expression_candidate import (
    MathematicalExpressionCandidate,
    MathematicalSignalInterpretationPolicy,
    validate_accepted_hypothesis_against_candidate,
)
from src.evaluation.mathematical_expression_strategy import (
    create_mathematical_research_registry,
)
from src.evaluation.mathematical_search import (
    MathematicalSearchError,
    MathematicalSearchResult,
    MathematicalSearchStrategy,
    SearchTerminationReason,
    SymbolicSearch,
)
from src.evaluation.selection_governance import (
    ResearchSelectionAssessment,
    assess_research_selection,
)
from src.strategies.registry import DEFAULT_REGISTRY, StrategyRegistry


@dataclass(frozen=True)
class DiscoveryCriteria:
    """Configurable quantitative thresholds and criteria for Research Discovery Engine evaluation."""

    min_observations_is: int = 30
    min_observations_oos: int = 10
    min_is_sharpe: float = 0.0
    min_is_total_return: float = -1.0
    max_is_drawdown: float = 1.0  # drawdown is negative, e.g., -0.5
    min_validation_sharpe: float = -0.5
    min_oos_sharpe: float = -0.5
    max_oos_sharpe_degradation: float = 0.8  # Max OOS Sharpe drop relative to IS
    min_walk_forward_positive_ratio: float = 0.5
    benchmark_reference: str = "buy_and_hold"
    methodology_version: str = "discovery_v1.0"
    strategy_version: str = "1.0.0"
    robustness_criteria: RobustnessCriteria = field(default_factory=RobustnessCriteria)

    def __post_init__(self) -> None:
        if self.min_observations_is <= 0:
            raise ValueError("min_observations_is must be positive.")
        if self.min_observations_oos <= 0:
            raise ValueError("min_observations_oos must be positive.")
        if not isinstance(self.robustness_criteria, RobustnessCriteria):
            raise TypeError("robustness_criteria must be a RobustnessCriteria instance.")


@dataclass(frozen=True)
class ResearchSearchPolicy:
    """Policy governing research search budget and execution constraints."""

    max_trials: int
    fail_fast: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.max_trials, int) or self.max_trials <= 0:
            raise ValueError("max_trials must be a positive integer.")


@dataclass(frozen=True)
class ResearchTrialRecord:
    """Canonical, immutable record of an attempted candidate research trial."""

    search_id: str
    trial_id: str
    trial_index: int
    candidate_id: str
    candidate_fingerprint: str
    experiment_fingerprint: str
    evidence_fingerprint: str | None
    qualification_status: PromotionStatus | None
    rejection_reasons: tuple[RejectionReason, ...]
    status: str  # PENDING, RUNNING, COMPLETED, FAILED, QUALIFIED, REJECTED, BLOCKED
    error_message: str = ""
    campaign_id: str = ""

    def __post_init__(self) -> None:
        if not self.search_id or not self.search_id.strip():
            raise ValueError("search_id must be a non-empty string.")
        if not self.trial_id or not self.trial_id.strip():
            raise ValueError("trial_id must be a non-empty string.")
        if self.trial_index < 0:
            raise ValueError("trial_index must be non-negative.")
        if not self.candidate_id or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string.")
        if self.status not in (
            "PENDING",
            "RUNNING",
            "COMPLETED",
            "FAILED",
            "QUALIFIED",
            "REJECTED",
            "BLOCKED",
        ):
            raise ValueError(f"Invalid trial status '{self.status}'.")


@dataclass(frozen=True)
class DiscoveryRunResult:
    """Aggregated result of a Research Discovery Engine execution run."""

    dataset_scope: DatasetScope
    execution_assumptions: ExecutionAssumptions
    code_provenance: CodeProvenance
    candidates_evaluated: int
    promoted_evidence: tuple[ResearchEvidence, ...]
    rejected_evidence: tuple[ResearchEvidence, ...]
    search_space_fingerprint: str = ""
    search_id: str = ""
    trial_ledger: tuple[ResearchTrialRecord, ...] = field(default_factory=tuple)
    search_truncated: bool = False
    selection_assessments: tuple[ResearchSelectionAssessment, ...] = field(
        default_factory=tuple
    )
    robustness_assessments: tuple[ResearchRobustnessAssessment, ...] = field(
        default_factory=tuple
    )
    registry_records: tuple[ResearchRegistryRecord, ...] = field(
        default_factory=tuple
    )
    learning_records: tuple[ResearchLearningRecord, ...] = field(
        default_factory=tuple
    )
    research_candidates: tuple[ResearchCandidate, ...] = field(
        default_factory=tuple
    )
    memory_governance_results: tuple[DiscoveryMemoryGovernanceResult, ...] = field(
        default_factory=tuple
    )
    discovery_feedback: tuple[ResearchDiscoveryFeedback, ...] = field(
        default_factory=tuple
    )
    campaign: ResearchCampaign | None = None

    @property
    def total_candidates(self) -> int:
        return self.candidates_evaluated

    @property
    def blocked_trial_count(self) -> int:
        return sum(
            1 for t in self.trial_ledger
            if t.status == "BLOCKED" or RejectionReason.GOVERNANCE_BLOCKED in t.rejection_reasons
        )

    @property
    def executed_trial_count(self) -> int:
        return sum(
            1 for t in self.trial_ledger
            if t.status in ("COMPLETED", "QUALIFIED", "REJECTED") and RejectionReason.GOVERNANCE_BLOCKED not in t.rejection_reasons
        )

    @property
    def governance_failed_trial_count(self) -> int:
        return sum(
            1 for g in self.memory_governance_results
            if g.decision == MemoryGovernanceDecision.FAIL_CLOSED
        )

    @property
    def trial_count(self) -> int:
        return len(self.trial_ledger) if self.trial_ledger else self.candidates_evaluated

    @property
    def successful_trial_count(self) -> int:
        return sum(1 for t in self.trial_ledger if t.status in ("COMPLETED", "QUALIFIED", "REJECTED"))

    @property
    def failed_trial_count(self) -> int:
        return sum(1 for t in self.trial_ledger if t.status == "FAILED")

    @property
    def qualified_trial_count(self) -> int:
        return sum(1 for t in self.trial_ledger if t.status == "QUALIFIED")

    @property
    def rejected_trial_count(self) -> int:
        return sum(1 for t in self.trial_ledger if t.status == "REJECTED")


class DiscoveryEngine:
    """Orchestrates candidate evaluation and evidence synthesis across partitioned data."""

    def __init__(
        self,
        criteria: DiscoveryCriteria | None = None,
        registry: StrategyRegistry = DEFAULT_REGISTRY,
        memory_store: ResearchRegistryStore | None = None,
    ) -> None:
        self.criteria = criteria or DiscoveryCriteria()
        self.registry = registry
        self.memory_store = memory_store

    def run_discovery(
        self,
        df: pd.DataFrame,
        candidates: Sequence[CandidateSpec] | ResearchSearchSpace,
        dataset_scope: DatasetScope,
        execution_assumptions: ExecutionAssumptions,
        code_provenance: CodeProvenance,
        *,
        search_policy: ResearchSearchPolicy | None = None,
        val_ratio: float = 0.2,
        oos_ratio: float = 0.3,
        wf_train_size: int | None = None,
        wf_test_size: int | None = None,
        persist_evidence: bool = False,
        persist_registry_dir: str | Path | None = None,
        memory_store: ResearchRegistryStore | Sequence[DoNotRepeatConstraint] | None = None,
        enable_memory_governance: bool = True,
        knowledge_patterns: Sequence[ResearchKnowledgePattern] | None = None,
        enable_discovery_feedback: bool = True,
    ) -> DiscoveryRunResult:
        """Execute discovery workflow over CandidateSpec candidates.

        Delegates directly to the single unified _run_governed_discovery_lifecycle.
        """
        data = validate_and_prepare_dataset(df, dataset_scope)

        # Search space resolution
        if isinstance(candidates, ResearchSearchSpace):
            search_space = candidates
        elif isinstance(candidates, (list, tuple)):
            search_space = ResearchSearchSpace(candidate_definitions=tuple(candidates))
        else:
            raise TypeError("candidates must be a sequence of CandidateSpec or a ResearchSearchSpace.")

        eval_candidates = search_space.candidate_definitions
        search_truncated = False

        if search_policy is not None:
            if not isinstance(search_policy, ResearchSearchPolicy):
                raise TypeError("search_policy must be a ResearchSearchPolicy instance.")
            if len(eval_candidates) > search_policy.max_trials:
                eval_candidates = eval_candidates[: search_policy.max_trials]
                search_truncated = True

        return self._run_governed_discovery_lifecycle(
            data=data,
            search_id=search_space.search_id,
            search_fingerprint=search_space.search_fingerprint,
            eval_candidates=eval_candidates,
            dataset_scope=dataset_scope,
            execution_assumptions=execution_assumptions,
            code_provenance=code_provenance,
            search_policy=search_policy,
            search_truncated=search_truncated,
            val_ratio=val_ratio,
            oos_ratio=oos_ratio,
            wf_train_size=wf_train_size,
            wf_test_size=wf_test_size,
            persist_evidence=persist_evidence,
            persist_registry_dir=persist_registry_dir,
            memory_store=memory_store,
            enable_memory_governance=enable_memory_governance,
            knowledge_patterns=knowledge_patterns,
            enable_discovery_feedback=enable_discovery_feedback,
            execution_registry=self.registry,
        )

    def run_mathematical_discovery(
        self,
        df: pd.DataFrame,
        search_space: MathematicalSearchSpace,
        dataset_scope: DatasetScope,
        execution_assumptions: ExecutionAssumptions,
        code_provenance: CodeProvenance,
        *,
        search_policy: ResearchSearchPolicy | None = None,
        search_strategy: MathematicalSearchStrategy | None = None,
        signal_policy: MathematicalSignalInterpretationPolicy | None = None,
        constant_values: Sequence[float] = (),
        limit: int | None = None,
        val_ratio: float = 0.2,
        oos_ratio: float = 0.3,
        wf_train_size: int | None = None,
        wf_test_size: int | None = None,
        persist_evidence: bool = False,
        persist_registry_dir: str | Path | None = None,
        memory_store: ResearchRegistryStore | Sequence[DoNotRepeatConstraint] | None = None,
        enable_memory_governance: bool = True,
        knowledge_patterns: Sequence[ResearchKnowledgePattern] | None = None,
        enable_discovery_feedback: bool = True,
    ) -> DiscoveryRunResult:
        """Execute canonical governed mathematical relationship discovery workflow.

        Thin adapter that performs structural candidate generation, lineage validation,
        and delegates to the ONE unified _run_governed_discovery_lifecycle.
        """
        if not isinstance(search_space, MathematicalSearchSpace):
            raise TypeError(f"search_space must be a MathematicalSearchSpace instance, got {type(search_space)}")
        if not isinstance(dataset_scope, DatasetScope):
            raise TypeError("dataset_scope must be a DatasetScope instance.")
        if not isinstance(execution_assumptions, ExecutionAssumptions):
            raise TypeError("execution_assumptions must be an ExecutionAssumptions instance.")
        if not isinstance(code_provenance, CodeProvenance):
            raise TypeError("code_provenance must be a CodeProvenance instance.")

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

        data = validate_and_prepare_dataset(df, dataset_scope)

        # 1. Bounded structural search generation (Zero market data evaluation)
        strategy = search_strategy or SymbolicSearch()

        if limit is not None:
            if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
                raise MathematicalSearchError(f"Requested search limit must be a positive integer, got {limit}")
            eff_limit = min(limit, search_space.max_search_budget)
        else:
            eff_limit = search_space.max_search_budget

        if search_policy is not None:
            if not isinstance(search_policy, ResearchSearchPolicy):
                raise TypeError("search_policy must be a ResearchSearchPolicy instance.")
            eff_limit = min(eff_limit, search_policy.max_trials)

        search_result = strategy.search(
            search_space=search_space,
            dataset_scope=dataset_scope,
            execution_assumptions=execution_assumptions,
            code_provenance=code_provenance,
            signal_policy=signal_policy,
            constant_values=constant_values,
            limit=eff_limit,
            generator_id=search_space.generator_id,
            generator_version=search_space.generator_version,
            random_seed=search_space.random_seed,
        )

        candidates = search_result.candidates
        search_truncated = (search_result.termination_reason == SearchTerminationReason.BUDGET_EXHAUSTED)

        # Create research-only StrategyRegistry for mathematical discovery execution
        research_registry = create_mathematical_research_registry()

        return self._run_governed_discovery_lifecycle(
            data=data,
            search_id=search_space.search_id,
            search_fingerprint=search_space.fingerprint,
            eval_candidates=candidates,
            dataset_scope=dataset_scope,
            execution_assumptions=execution_assumptions,
            code_provenance=code_provenance,
            search_policy=search_policy,
            search_truncated=search_truncated,
            val_ratio=val_ratio,
            oos_ratio=oos_ratio,
            wf_train_size=wf_train_size,
            wf_test_size=wf_test_size,
            persist_evidence=persist_evidence,
            persist_registry_dir=persist_registry_dir,
            memory_store=memory_store,
            enable_memory_governance=enable_memory_governance,
            knowledge_patterns=knowledge_patterns,
            enable_discovery_feedback=enable_discovery_feedback,
            execution_registry=research_registry,
        )

    def _run_governed_discovery_lifecycle(
        self,
        *,
        data: pd.DataFrame,
        search_id: str,
        search_fingerprint: str,
        eval_candidates: Sequence[CandidateSpec | MathematicalExpressionCandidate],
        dataset_scope: DatasetScope,
        execution_assumptions: ExecutionAssumptions,
        code_provenance: CodeProvenance,
        search_policy: ResearchSearchPolicy | None,
        search_truncated: bool,
        val_ratio: float,
        oos_ratio: float,
        wf_train_size: int | None,
        wf_test_size: int | None,
        persist_evidence: bool,
        persist_registry_dir: str | Path | None,
        memory_store: ResearchRegistryStore | Sequence[DoNotRepeatConstraint] | None,
        enable_memory_governance: bool,
        knowledge_patterns: Sequence[ResearchKnowledgePattern] | None,
        enable_discovery_feedback: bool,
        execution_registry: StrategyRegistry,
    ) -> DiscoveryRunResult:
        """SINGLE CANONICAL GOVERNED DISCOVERY LIFECYCLE.

        Orchestrates campaign definition, trial planning, discovery feedback governance,
        memory governance, hypothesis generation/acceptance/validation, research execution,
        robustness assessment, qualification, selection governance, registry/learning recording,
        and campaign lifecycle tracking.
        """
        effective_search_policy = search_policy or ResearchSearchPolicy(max_trials=max(len(eval_candidates), 1))
        search_policy_fp = compute_search_policy_fingerprint(
            max_trials=effective_search_policy.max_trials,
            fail_fast=effective_search_policy.fail_fast,
        )
        criteria_fp = compute_criteria_fingerprint(self.criteria)
        cand_ids = tuple(c.candidate_id for c in eval_candidates)

        campaign_id = compute_campaign_fingerprint(
            search_space_fingerprint=search_fingerprint,
            search_policy_fingerprint=search_policy_fp,
            criteria_fingerprint=criteria_fp,
            dataset_scope=dataset_scope,
            execution_assumptions=execution_assumptions,
            code_provenance=code_provenance,
            candidate_ids=cand_ids,
        )

        campaign_store = (
            ResearchCampaignStore(base_dir=persist_registry_dir)
            if persist_evidence and persist_registry_dir
            else ResearchCampaignStore()
        )

        definition = ResearchCampaignDefinition(
            search_space_fingerprint=search_fingerprint,
            search_policy_fingerprint=search_policy_fp,
            criteria_fingerprint=criteria_fp,
            dataset_scope=dataset_scope,
            execution_assumptions=execution_assumptions,
            code_provenance=code_provenance,
            methodology_version=self.criteria.methodology_version,
            candidate_ids=cand_ids,
            trial_count=len(cand_ids),
        )

        planned_trials = [
            ResearchPlannedTrial(
                campaign_id=campaign_id,
                trial_id=f"{search_id}_trial_{idx}",
                trial_index=idx,
                candidate_id=cand.candidate_id,
                candidate_fingerprint=cand.candidate_id if isinstance(cand, CandidateSpec) else cand.fingerprint,
                hypothesis_fingerprint=cand.candidate_id if isinstance(cand, CandidateSpec) else cand.fingerprint,
                strategy_name=cand.strategy_name if isinstance(cand, CandidateSpec) else "mathematical_expression",
                strategy_version=self.criteria.strategy_version,
                dataset_id=dataset_scope.dataset_id,
                execution_assumptions_id=f"ea_{hashlib.sha256(str(execution_assumptions).encode('utf-8')).hexdigest()[:8]}",
            )
            for idx, cand in enumerate(eval_candidates)
        ]

        plan = ResearchTrialPlan(
            campaign_id=campaign_id,
            definition_fingerprint=definition.definition_fingerprint,
            trials=tuple(planned_trials),
        )

        if persist_evidence:
            campaign_store.save_definition(definition)
            campaign_store.save_trial_plan(plan)
            try:
                curr_state = campaign_store.load_lifecycle_state(campaign_id)
                current_status = ResearchCampaignStatus(curr_state["status"])
            except Exception:
                current_status = None

            if current_status not in (
                ResearchCampaignStatus.COMPLETED,
                ResearchCampaignStatus.TRUNCATED,
                ResearchCampaignStatus.FAILED,
                ResearchCampaignStatus.CANCELLED,
            ):
                campaign_store.save_lifecycle_state(campaign_id, ResearchCampaignStatus.RUNNING)

        # Resolve effective WalkForwardProtocol from dataset length
        n = len(data)
        wf_protocol = resolve_walk_forward_protocol(
            n_observations=n,
            train_size=wf_train_size,
            test_size=wf_test_size,
        )

        # Chronological partitioning
        is_ratio = 1.0 - val_ratio - oos_ratio
        if is_ratio <= 0:
            raise ValueError(
                f"Invalid partition ratios: val_ratio ({val_ratio}) + oos_ratio ({oos_ratio}) >= 1.0"
            )

        n_is = int(n * is_ratio)
        n_val = int(n * val_ratio)
        n_oos = n - n_is - n_val

        df_is = data.iloc[:n_is]
        df_val = data.iloc[n_is : n_is + n_val]
        df_oos = data.iloc[n_is + n_val :]

        promoted: list[ResearchEvidence] = []
        rejected: list[ResearchEvidence] = []
        trial_records: list[ResearchTrialRecord] = []
        research_candidates: list[ResearchCandidate] = []
        memory_governance_results: list[DiscoveryMemoryGovernanceResult] = []
        registry_records: list[ResearchRegistryRecord] = []
        learning_records: list[ResearchLearningRecord] = []
        robustness_assessment_by_evidence_id: dict[str, ResearchRobustnessAssessment] = {}
        governance_decision_by_evidence_id: dict[str, Any] = {}

        registry_store = (
            ResearchRegistryStore(base_dir=persist_registry_dir)
            if persist_evidence and persist_registry_dir
            else (ResearchRegistryStore() if persist_evidence else None)
        )

        # Resolve active constraints without silent filtering or fail-open fallbacks
        active_constraints: Sequence[DoNotRepeatConstraint] = ()
        memory_store_error: str | None = None
        if enable_memory_governance:
            if isinstance(memory_store, (list, tuple)):
                active_constraints = memory_store  # Unfiltered sequence
            else:
                eff_store = memory_store if memory_store is not None else self.memory_store
                if eff_store is not None:
                    try:
                        active_constraints = eff_store.get_active_do_not_repeat_constraints(
                            symbol=dataset_scope.symbol,
                            timeframe=dataset_scope.timeframe,
                        )
                    except Exception as exc:
                        memory_store_error = f"Governance store constraint resolution failed: {exc}"

        # Resolve knowledge patterns without silent filtering or fail-open fallbacks
        effective_patterns: Sequence[ResearchKnowledgePattern] = ()
        knowledge_store_error: str | None = None
        if enable_discovery_feedback:
            if knowledge_patterns is not None:
                effective_patterns = knowledge_patterns  # Unfiltered sequence
            else:
                eff_store = memory_store if isinstance(memory_store, ResearchRegistryStore) else self.memory_store
                if eff_store is not None and hasattr(eff_store, "list_patterns"):
                    try:
                        effective_patterns = eff_store.list_patterns()
                    except Exception as exc:
                        knowledge_store_error = f"Governance store knowledge pattern resolution failed: {exc}"

        discovery_feedback_records: list[ResearchDiscoveryFeedback] = []
        seen_candidate_fingerprints: set[str] = set()

        for idx, cand in enumerate(eval_candidates):
            trial_id = f"{search_id}_trial_{idx}"

            # Fail closed on governance store resolution failures
            if memory_store_error or knowledge_store_error:
                store_err_msg = memory_store_error or knowledge_store_error or "Governance store resolution failed."
                if effective_search_policy.fail_fast:
                    raise RegistryValidationError(store_err_msg)

                trial_record = ResearchTrialRecord(
                    search_id=search_id,
                    trial_id=trial_id,
                    trial_index=idx,
                    candidate_id=cand.candidate_id,
                    candidate_fingerprint=cand.candidate_id if isinstance(cand, CandidateSpec) else cand.fingerprint,
                    experiment_fingerprint="",
                    evidence_fingerprint=None,
                    qualification_status=PromotionStatus.REJECTED,
                    rejection_reasons=(RejectionReason.SPECIFICATION_INVALID, RejectionReason.GOVERNANCE_BLOCKED),
                    status="FAILED",
                    error_message=store_err_msg,
                    campaign_id=campaign_id,
                )
                trial_records.append(trial_record)

                dummy_hyp = (
                    cand.to_hypothesis(walk_forward_protocol=wf_protocol)
                    if isinstance(cand, MathematicalExpressionCandidate)
                    else ResearchHypothesis(
                        statement=f"Hypothesis for candidate {cand.candidate_id}",
                        methodology_version=self.criteria.methodology_version,
                        strategy_name=cand.strategy_name,
                        strategy_version=self.criteria.strategy_version,
                        dataset_scope=dataset_scope,
                        execution_assumptions=execution_assumptions,
                        code_provenance=code_provenance,
                        benchmark_reference=self.criteria.benchmark_reference,
                        parameters=dict(cand.parameters),
                        random_seed=cand.random_seed,
                        walk_forward_protocol=wf_protocol,
                    )
                )

                research_cand = ResearchCandidate(
                    candidate_id=cand.candidate_id,
                    hypothesis=dummy_hyp,
                    evidence=None,
                    validation_status=PromotionStatus.REJECTED,
                    promotion_status=PromotionStatus.REJECTED,
                    rejection_reasons=(RejectionReason.SPECIFICATION_INVALID, RejectionReason.GOVERNANCE_BLOCKED),
                )
                research_candidates.append(research_cand)
                continue

            # Pre-validation for MathematicalExpressionCandidate
            if isinstance(cand, MathematicalExpressionCandidate):
                try:
                    cand.validate()
                    if cand.search_space.fingerprint != search_fingerprint:
                        raise MathematicalSearchError("Candidate search_space.fingerprint does not match SearchSpace fingerprint.")
                    if (
                        cand.generator_id != cand.search_space.generator_id
                        or cand.generator_version != cand.search_space.generator_version
                        or cand.random_seed != cand.search_space.random_seed
                    ):
                        raise MathematicalSearchError("Candidate generator metadata does not match SearchSpace generator metadata.")
                    if cand.dataset_scope != dataset_scope or cand.execution_assumptions != execution_assumptions or cand.code_provenance != code_provenance:
                        raise MathematicalSearchError("Candidate research lineage does not match input lineage.")
                except Exception as exc:
                    trial_record = ResearchTrialRecord(
                        search_id=search_id,
                        trial_id=trial_id,
                        trial_index=idx,
                        candidate_id=cand.candidate_id,
                        candidate_fingerprint=cand.fingerprint,
                        experiment_fingerprint="",
                        evidence_fingerprint=None,
                        qualification_status=PromotionStatus.REJECTED,
                        rejection_reasons=(RejectionReason.SPECIFICATION_INVALID,),
                        status="FAILED",
                        error_message=str(exc),
                        campaign_id=campaign_id,
                    )
                    trial_records.append(trial_record)
                    if effective_search_policy.fail_fast:
                        raise
                    continue

            # Construct initial GENERATED hypothesis
            if isinstance(cand, CandidateSpec):
                hypothesis_stmt = (
                    cand.hypothesis_template.replace("{candidate_id}", cand.candidate_id)
                    if cand.hypothesis_template
                    else f"Hypothesis for candidate {cand.candidate_id}"
                )
                hypothesis = ResearchHypothesis(
                    statement=hypothesis_stmt,
                    methodology_version=self.criteria.methodology_version,
                    strategy_name=cand.strategy_name,
                    strategy_version=self.criteria.strategy_version,
                    dataset_scope=dataset_scope,
                    execution_assumptions=execution_assumptions,
                    code_provenance=code_provenance,
                    benchmark_reference=self.criteria.benchmark_reference,
                    parameters=dict(cand.parameters),
                    random_seed=cand.random_seed,
                    walk_forward_protocol=wf_protocol,
                )
            else:
                hypothesis = cand.to_hypothesis(
                    walk_forward_protocol=wf_protocol,
                    benchmark_reference=self.criteria.benchmark_reference,
                    methodology_version=self.criteria.methodology_version,
                    strategy_version=self.criteria.strategy_version,
                )

            # Stage 0a: Controlled Discovery Feedback
            feedback_fail_closed = False
            fail_closed_reason = ""
            if enable_discovery_feedback:
                cand_feedbacks = evaluate_candidate_discovery_feedback(
                    candidate=cand,
                    dataset_scope=dataset_scope,
                    execution_assumptions=execution_assumptions,
                    code_provenance=code_provenance,
                    knowledge_patterns=effective_patterns,
                    hypothesis=hypothesis,
                    search_id=search_id,
                    search_fingerprint=search_fingerprint,
                    registry_store=registry_store,
                    methodology_version=self.criteria.methodology_version,
                )
                for fb in cand_feedbacks:
                    discovery_feedback_records.append(fb)
                    if registry_store is not None:
                        try:
                            registry_store.register_feedback(fb)
                        except Exception:
                            pass
                    if fb.feedback_type == DiscoveryFeedbackType.FAIL_CLOSED:
                        feedback_fail_closed = True
                        fail_closed_reason = fb.reason

            # HARD FAIL-CLOSED BOUNDARY FOR DISCOVERY FEEDBACK
            if feedback_fail_closed:
                if effective_search_policy.fail_fast:
                    raise RegistryValidationError(
                        f"Discovery feedback failed closed for candidate '{cand.candidate_id}': {fail_closed_reason}"
                    )

                trial_record = ResearchTrialRecord(
                    search_id=search_id,
                    trial_id=trial_id,
                    trial_index=idx,
                    candidate_id=cand.candidate_id,
                    candidate_fingerprint=cand.candidate_id if isinstance(cand, CandidateSpec) else cand.fingerprint,
                    experiment_fingerprint="",
                    evidence_fingerprint=None,
                    qualification_status=PromotionStatus.REJECTED,
                    rejection_reasons=(RejectionReason.SPECIFICATION_INVALID, RejectionReason.GOVERNANCE_BLOCKED),
                    status="FAILED",
                    error_message=fail_closed_reason,
                    campaign_id=campaign_id,
                )
                trial_records.append(trial_record)

                research_cand = ResearchCandidate(
                    candidate_id=cand.candidate_id,
                    hypothesis=hypothesis,
                    evidence=None,
                    validation_status=PromotionStatus.REJECTED,
                    promotion_status=PromotionStatus.REJECTED,
                    rejection_reasons=(RejectionReason.SPECIFICATION_INVALID, RejectionReason.GOVERNANCE_BLOCKED),
                )
                research_candidates.append(research_cand)
                continue

            # Stage 0b: Memory-Aware Discovery Governance
            if enable_memory_governance and active_constraints:
                gov_res = evaluate_candidate_memory_governance(
                    candidate=cand,
                    dataset_scope=dataset_scope,
                    execution_assumptions=execution_assumptions,
                    code_provenance=code_provenance,
                    active_constraints=active_constraints,
                    hypothesis=hypothesis,
                    search_id=search_id,
                    search_fingerprint=search_fingerprint,
                    methodology_version=self.criteria.methodology_version,
                )
                memory_governance_results.append(gov_res)

                if gov_res.decision == MemoryGovernanceDecision.BLOCKED:
                    trial_record = ResearchTrialRecord(
                        search_id=search_id,
                        trial_id=trial_id,
                        trial_index=idx,
                        candidate_id=cand.candidate_id,
                        candidate_fingerprint=cand.candidate_id if isinstance(cand, CandidateSpec) else cand.fingerprint,
                        experiment_fingerprint=gov_res.experiment_fingerprint,
                        evidence_fingerprint=None,
                        qualification_status=PromotionStatus.REJECTED,
                        rejection_reasons=(RejectionReason.GOVERNANCE_BLOCKED,),
                        status="BLOCKED",
                        error_message=gov_res.reason,
                        campaign_id=campaign_id,
                    )
                    trial_records.append(trial_record)

                    research_cand = ResearchCandidate(
                        candidate_id=cand.candidate_id,
                        hypothesis=hypothesis,
                        evidence=None,
                        validation_status=PromotionStatus.REJECTED,
                        promotion_status=PromotionStatus.REJECTED,
                        rejection_reasons=(RejectionReason.GOVERNANCE_BLOCKED,),
                    )
                    research_candidates.append(research_cand)

                    from src.evaluation.research_registry import (
                        RegistryStatus,
                        ResearchEvidenceLineage,
                        ResearchReproducibilityDescriptor,
                    )
                    ds_id = _compute_scope_id(dataset_scope)
                    ea_id = _compute_ea_id(execution_assumptions)
                    cp_id = _compute_cp_id(code_provenance)

                    repro = ResearchReproducibilityDescriptor(
                        experiment_fingerprint=gov_res.experiment_fingerprint,
                        evidence_fingerprint=None,
                        dataset_scope_id=ds_id,
                        execution_assumptions_id=ea_id,
                        code_provenance_id=cp_id,
                        methodology_version=self.criteria.methodology_version,
                        search_space_fingerprint=search_fingerprint,
                        trial_id=trial_id,
                        candidate_id=cand.candidate_id,
                    )
                    lin = ResearchEvidenceLineage(
                        search_id=search_id,
                        search_fingerprint=search_fingerprint,
                        trial_id=trial_id,
                        trial_index=idx,
                        candidate_id=cand.candidate_id,
                        experiment_fingerprint=gov_res.experiment_fingerprint,
                        evidence_fingerprint=None,
                        qualification_status="REJECTED",
                        selection_assessment_id=None,
                        robustness_assessment_id=None,
                        promotion_status=PromotionStatus.REJECTED.value,
                    )
                    blocked_key = f"blocked:{trial_id}:{cand.candidate_id}"
                    blocked_rec_id = hashlib.sha256(blocked_key.encode("utf-8")).hexdigest()[:24]
                    blocked_rec = ResearchRegistryRecord(
                        record_id=blocked_rec_id,
                        experiment_fingerprint=gov_res.experiment_fingerprint,
                        evidence_fingerprint=None,
                        candidate_id=cand.candidate_id,
                        search_fingerprint=search_fingerprint,
                        search_id=search_id,
                        trial_id=trial_id,
                        trial_index=idx,
                        status=RegistryStatus.REJECTED,
                        qualification_status="REJECTED",
                        promotion_status=PromotionStatus.REJECTED.value,
                        rejection_reasons=("GOVERNANCE_BLOCKED",),
                        dataset_scope_id=ds_id,
                        execution_assumptions_id=ea_id,
                        code_provenance_id=cp_id,
                        methodology_version=self.criteria.methodology_version,
                        selection_assessment_id=None,
                        robustness_assessment_id=None,
                        benchmark_status=None,
                        regime_status=None,
                        error_message=gov_res.reason,
                        reproducibility=repro,
                        lineage=lin,
                    )
                    registry_records.append(blocked_rec)
                    if registry_store is not None:
                        registry_store.register(blocked_rec)

                    continue

                elif gov_res.decision == MemoryGovernanceDecision.FAIL_CLOSED:
                    if effective_search_policy.fail_fast:
                        raise RegistryValidationError(
                            f"Memory governance failed closed for candidate '{cand.candidate_id}': {gov_res.reason}"
                        )

                    trial_record = ResearchTrialRecord(
                        search_id=search_id,
                        trial_id=trial_id,
                        trial_index=idx,
                        candidate_id=cand.candidate_id,
                        candidate_fingerprint=cand.candidate_id if isinstance(cand, CandidateSpec) else cand.fingerprint,
                        experiment_fingerprint=gov_res.experiment_fingerprint,
                        evidence_fingerprint=None,
                        qualification_status=PromotionStatus.REJECTED,
                        rejection_reasons=(RejectionReason.SPECIFICATION_INVALID, RejectionReason.GOVERNANCE_BLOCKED),
                        status="FAILED",
                        error_message=gov_res.reason,
                        campaign_id=campaign_id,
                    )
                    trial_records.append(trial_record)

                    research_cand = ResearchCandidate(
                        candidate_id=cand.candidate_id,
                        hypothesis=hypothesis,
                        evidence=None,
                        validation_status=PromotionStatus.REJECTED,
                        promotion_status=PromotionStatus.REJECTED,
                        rejection_reasons=(RejectionReason.SPECIFICATION_INVALID, RejectionReason.GOVERNANCE_BLOCKED),
                    )
                    research_candidates.append(research_cand)
                    continue

            # Transition candidate hypothesis to ACCEPTED_FOR_RESEARCH
            try:
                accepted_hypothesis = accept_hypothesis_for_research(hypothesis)
            except Exception as exc:
                trial_record = ResearchTrialRecord(
                    search_id=search_id,
                    trial_id=trial_id,
                    trial_index=idx,
                    candidate_id=cand.candidate_id,
                    candidate_fingerprint=cand.candidate_id if isinstance(cand, CandidateSpec) else cand.fingerprint,
                    experiment_fingerprint="",
                    evidence_fingerprint=None,
                    qualification_status=PromotionStatus.REJECTED,
                    rejection_reasons=(RejectionReason.GOVERNANCE_BLOCKED,),
                    status="BLOCKED",
                    error_message=str(exc),
                    campaign_id=campaign_id,
                )
                trial_records.append(trial_record)

                research_cand = ResearchCandidate(
                    candidate_id=cand.candidate_id,
                    hypothesis=hypothesis,
                    evidence=None,
                    validation_status=PromotionStatus.REJECTED,
                    promotion_status=PromotionStatus.REJECTED,
                    rejection_reasons=(RejectionReason.GOVERNANCE_BLOCKED,),
                )
                research_candidates.append(research_cand)

                if effective_search_policy.fail_fast:
                    raise

                continue

            # Boundary validation for accepted hypothesis against MathematicalExpressionCandidate
            if isinstance(cand, MathematicalExpressionCandidate):
                try:
                    validate_accepted_hypothesis_against_candidate(
                        candidate=cand,
                        accepted_hypothesis=accepted_hypothesis,
                        walk_forward_protocol=wf_protocol,
                    )
                except Exception as exc:
                    trial_record = ResearchTrialRecord(
                        search_id=search_id,
                        trial_id=trial_id,
                        trial_index=idx,
                        candidate_id=cand.candidate_id,
                        candidate_fingerprint=cand.fingerprint,
                        experiment_fingerprint="",
                        evidence_fingerprint=None,
                        qualification_status=PromotionStatus.REJECTED,
                        rejection_reasons=(RejectionReason.SPECIFICATION_INVALID,),
                        status="FAILED",
                        error_message=str(exc),
                        campaign_id=campaign_id,
                    )
                    trial_records.append(trial_record)
                    if effective_search_policy.fail_fast:
                        raise
                    continue

            # Research experiment execution
            try:
                if isinstance(cand, CandidateSpec):
                    evidence = self._evaluate_candidate(
                        cand=cand,
                        hypothesis=accepted_hypothesis,
                        df_full=data,
                        df_is=df_is,
                        df_val=df_val,
                        df_oos=df_oos,
                        dataset_scope=dataset_scope,
                        execution_assumptions=execution_assumptions,
                        code_provenance=code_provenance,
                        wf_protocol=wf_protocol,
                        wf_train_size=wf_train_size,
                        wf_test_size=wf_test_size,
                        seen_fingerprints=seen_candidate_fingerprints,
                    )
                else:
                    evidence = run_research_experiment(
                        spec=accepted_hypothesis,
                        df=data,
                        criteria=self.criteria,
                        registry=execution_registry,
                        wf_train_size=wf_train_size,
                        wf_test_size=wf_test_size,
                        persist_evidence=False,
                    )
            except Exception as exc:
                if effective_search_policy.fail_fast:
                    raise

                trial_record = ResearchTrialRecord(
                    search_id=search_id,
                    trial_id=trial_id,
                    trial_index=idx,
                    candidate_id=cand.candidate_id,
                    candidate_fingerprint=cand.candidate_id if isinstance(cand, CandidateSpec) else cand.fingerprint,
                    experiment_fingerprint="",
                    evidence_fingerprint=None,
                    qualification_status=PromotionStatus.REJECTED,
                    rejection_reasons=(RejectionReason.SPECIFICATION_INVALID,),
                    status="FAILED",
                    error_message=str(exc),
                    campaign_id=campaign_id,
                )
                trial_records.append(trial_record)

                research_cand = ResearchCandidate(
                    candidate_id=cand.candidate_id,
                    hypothesis=accepted_hypothesis,
                    evidence=None,
                    validation_status=PromotionStatus.REJECTED,
                    promotion_status=PromotionStatus.REJECTED,
                    rejection_reasons=(RejectionReason.SPECIFICATION_INVALID,),
                )
                research_candidates.append(research_cand)
                continue

            # Identity chain post-execution verification
            if evidence.experiment_fingerprint != accepted_hypothesis.fingerprint:
                raise MathematicalSearchError(
                    f"Identity chain broken post-execution: evidence.experiment_fingerprint ({evidence.experiment_fingerprint}) "
                    f"does not match accepted_hypothesis fingerprint ({accepted_hypothesis.fingerprint})."
                )

            if evidence.experiment_fingerprint in seen_candidate_fingerprints and RejectionReason.DUPLICATE_CANDIDATE not in evidence.rejection_reasons:
                rejection_reasons = list(evidence.rejection_reasons) + [RejectionReason.DUPLICATE_CANDIDATE]
                evidence = ResearchEvidence(
                    experiment_fingerprint=evidence.experiment_fingerprint,
                    spec=evidence.spec,
                    partitions=evidence.partitions,
                    robustness_verdict=evidence.robustness_verdict,
                    benchmark_comparison=evidence.benchmark_comparison,
                    promotion_status=PromotionStatus.REJECTED,
                    rejection_reasons=tuple(dict.fromkeys(rejection_reasons)),
                    critique_notes=evidence.critique_notes,
                    created_at_utc=evidence.created_at_utc,
                )

            seen_candidate_fingerprints.add(evidence.experiment_fingerprint)

            if persist_evidence:
                save_research_experiment(
                    evidence,
                    base_dir=persist_registry_dir if persist_registry_dir else DEFAULT_RESEARCH_DIR,
                )

            # Robustness Assessment and Qualification
            rob_assessment = assess_research_robustness(
                evidence=evidence,
                robustness_criteria=self.criteria.robustness_criteria,
            )
            robustness_assessment_by_evidence_id[evidence.experiment_fingerprint] = rob_assessment

            from src.evaluation.research_qualification import qualify_research_evidence
            qual_res = qualify_research_evidence(evidence, robustness_assessment=rob_assessment)
            governance_decision_by_evidence_id[evidence.experiment_fingerprint] = qual_res

            if qual_res.qualified:
                promoted.append(evidence)
                trial_status = "QUALIFIED"
            else:
                rejected.append(evidence)
                trial_status = "REJECTED"

            trial_record = ResearchTrialRecord(
                search_id=search_id,
                trial_id=trial_id,
                trial_index=idx,
                candidate_id=cand.candidate_id,
                candidate_fingerprint=cand.candidate_id if isinstance(cand, CandidateSpec) else cand.fingerprint,
                experiment_fingerprint=evidence.experiment_fingerprint,
                evidence_fingerprint=evidence.evidence_id,
                qualification_status=qual_res.status,
                rejection_reasons=qual_res.rejection_reasons,
                status=trial_status,
                error_message="",
                campaign_id=campaign_id,
            )
            trial_records.append(trial_record)

            research_cand = ResearchCandidate(
                candidate_id=cand.candidate_id,
                hypothesis=accepted_hypothesis,
                evidence=evidence,
                validation_status=qual_res.status,
                promotion_status=qual_res.status,
                rejection_reasons=qual_res.rejection_reasons,
                created_at_utc=evidence.created_at_utc,
            )
            research_candidates.append(research_cand)

        # Deterministic ranking
        def _evidence_rank_key(ev: ResearchEvidence) -> tuple[float, float, str]:
            oos_sharpe = next(
                (p.sharpe_ratio for p in ev.partitions if p.role == EvidencePartitionRole.OUT_OF_SAMPLE),
                -999.0,
            )
            total_return = next(
                (p.total_return for p in ev.partitions if p.role == EvidencePartitionRole.OUT_OF_SAMPLE),
                -999.0,
            )
            return (-oos_sharpe, -total_return, ev.experiment_fingerprint)

        promoted_sorted = sorted(promoted, key=_evidence_rank_key)
        rejected_sorted = sorted(rejected, key=_evidence_rank_key)

        selection_assessments: list[ResearchSelectionAssessment] = []
        robustness_assessments: list[ResearchRobustnessAssessment] = []

        all_evidence = promoted_sorted + rejected_sorted
        for ev in all_evidence:
            assessment = assess_research_selection(
                evidence=ev,
                trial_records=trial_records,
                search_fingerprint=search_fingerprint,
                evidence_collection=all_evidence,
            )
            selection_assessments.append(assessment)

            rob_assessment = robustness_assessment_by_evidence_id.get(ev.experiment_fingerprint)
            if rob_assessment is None:
                raise RuntimeError(
                    f"Canonical robustness assessment missing for evidence '{ev.experiment_fingerprint}'."
                )
            robustness_assessments.append(rob_assessment)

            tr = next((t for t in trial_records if t.experiment_fingerprint == ev.experiment_fingerprint), None)
            cand_id = tr.candidate_id if tr else None
            tr_id = tr.trial_id if tr else None
            tr_idx = tr.trial_index if tr else None
            qual_stat = tr.status if tr else None

            rec = construct_registry_record_from_evidence(
                evidence=ev,
                candidate_id=cand_id,
                search_id=search_id,
                search_fingerprint=search_fingerprint,
                trial_id=tr_id,
                trial_index=tr_idx,
                selection_assessment=assessment,
                robustness_assessment=rob_assessment,
                qualification_status=qual_stat,
            )
            registry_records.append(rec)
            learning_rec = construct_learning_record_from_registry_record(rec)
            learning_records.append(learning_rec)
            if registry_store is not None:
                registry_store.register(rec)
                registry_store.register_learning_record(learning_rec)

        # Handle failed trials in registry
        for tr in trial_records:
            if tr.status == "FAILED":
                failed_key = f"failed:{tr.trial_id}:{tr.candidate_id}"
                failed_rec_id = hashlib.sha256(failed_key.encode("utf-8")).hexdigest()[:24]
                ds_id = _compute_scope_id(dataset_scope)
                ea_id = _compute_ea_id(execution_assumptions)
                cp_id = _compute_cp_id(code_provenance)

                from src.evaluation.research_registry import (
                    RegistryStatus,
                    ResearchEvidenceLineage,
                    ResearchReproducibilityDescriptor,
                )
                repro = ResearchReproducibilityDescriptor(
                    experiment_fingerprint=f"failed_{tr.candidate_id}",
                    evidence_fingerprint=None,
                    dataset_scope_id=ds_id,
                    execution_assumptions_id=ea_id,
                    code_provenance_id=cp_id,
                    methodology_version=self.criteria.methodology_version,
                    search_space_fingerprint=search_fingerprint,
                    trial_id=tr.trial_id,
                    candidate_id=tr.candidate_id,
                )
                lin = ResearchEvidenceLineage(
                    search_id=search_id,
                    search_fingerprint=search_fingerprint,
                    trial_id=tr.trial_id,
                    trial_index=tr.trial_index,
                    candidate_id=tr.candidate_id,
                    experiment_fingerprint=f"failed_{tr.candidate_id}",
                    evidence_fingerprint=None,
                    qualification_status="REJECTED",
                    selection_assessment_id=None,
                    robustness_assessment_id=None,
                    promotion_status=PromotionStatus.REJECTED.value,
                )
                failed_rec = ResearchRegistryRecord(
                    record_id=failed_rec_id,
                    experiment_fingerprint=f"failed_{tr.candidate_id}",
                    evidence_fingerprint=None,
                    candidate_id=tr.candidate_id,
                    search_fingerprint=search_fingerprint,
                    search_id=search_id,
                    trial_id=tr.trial_id,
                    trial_index=tr.trial_index,
                    status=RegistryStatus.FAILED,
                    qualification_status="REJECTED",
                    promotion_status=PromotionStatus.REJECTED.value,
                    rejection_reasons=("SPECIFICATION_INVALID",),
                    dataset_scope_id=ds_id,
                    execution_assumptions_id=ea_id,
                    code_provenance_id=cp_id,
                    methodology_version=self.criteria.methodology_version,
                    selection_assessment_id=None,
                    robustness_assessment_id=None,
                    benchmark_status=None,
                    regime_status=None,
                    error_message=tr.error_message,
                    reproducibility=repro,
                    lineage=lin,
                )
                registry_records.append(failed_rec)
                failed_learning_rec = construct_learning_record_from_registry_record(failed_rec)
                learning_records.append(failed_learning_rec)
                if registry_store is not None:
                    registry_store.register(failed_rec)
                    registry_store.register_learning_record(failed_learning_rec)

        ev_fps = tuple(ev.evidence_id for ev in all_evidence if ev.evidence_id)
        selected_ids = tuple(
            tr.candidate_id for tr in trial_records if tr.status == "QUALIFIED"
        )
        campaign_status = (
            ResearchCampaignStatus.TRUNCATED
            if search_truncated
            else ResearchCampaignStatus.COMPLETED
        )

        campaign = ResearchCampaign(
            campaign_id=campaign_id,
            search_space_fingerprint=search_fingerprint,
            search_policy_fingerprint=search_policy_fp,
            criteria_fingerprint=criteria_fp,
            dataset_scope=dataset_scope,
            execution_assumptions=execution_assumptions,
            code_provenance=code_provenance,
            candidate_ids=cand_ids,
            evidence_fingerprints=ev_fps,
            selected_candidate_ids=selected_ids,
            status=campaign_status,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            definition_fingerprint=definition.definition_fingerprint,
            trial_plan_fingerprint=plan.plan_fingerprint,
            executed_trial_count=sum(1 for t in trial_records if t.status in ("COMPLETED", "QUALIFIED", "REJECTED")),
            failed_trial_count=sum(1 for t in trial_records if t.status == "FAILED"),
            blocked_trial_count=sum(1 for t in trial_records if t.status == "BLOCKED"),
        )

        if persist_evidence:
            try:
                curr_state = campaign_store.load_lifecycle_state(campaign_id)
                curr_status = ResearchCampaignStatus(curr_state["status"])
            except Exception:
                curr_status = None

            if curr_status != campaign_status:
                campaign_store.save_lifecycle_state(campaign_id, campaign_status)

            save_research_campaign(
                campaign,
                base_dir=persist_registry_dir if persist_registry_dir else DEFAULT_CAMPAIGN_DIR,
            )

        return DiscoveryRunResult(
            dataset_scope=dataset_scope,
            execution_assumptions=execution_assumptions,
            code_provenance=code_provenance,
            candidates_evaluated=len(eval_candidates),
            promoted_evidence=tuple(promoted_sorted),
            rejected_evidence=tuple(rejected_sorted),
            search_space_fingerprint=search_fingerprint,
            search_id=search_id,
            trial_ledger=tuple(trial_records),
            search_truncated=search_truncated,
            selection_assessments=tuple(selection_assessments),
            robustness_assessments=tuple(robustness_assessments),
            registry_records=tuple(registry_records),
            learning_records=tuple(learning_records),
            research_candidates=tuple(research_candidates),
            memory_governance_results=tuple(memory_governance_results),
            discovery_feedback=tuple(discovery_feedback_records),
            campaign=campaign,
        )

    def _validate_dataset_scope(
        self, df: pd.DataFrame, dataset_scope: DatasetScope
    ) -> None:
        """Fail closed if dataset scope start/end dates violate data limits."""
        if "timestamp" in df.columns:
            ts = df["timestamp"]
        else:
            ts = df.index

        df_start = pd.to_datetime(ts.min()).strftime("%Y-%m-%d")
        df_end = pd.to_datetime(ts.max()).strftime("%Y-%m-%d")

        scope_start = dataset_scope.start_date[:10]
        scope_end = dataset_scope.end_date[:10]

        if scope_start < df_start or scope_end > df_end:
            raise ValueError(
                f"DatasetScope dates [{scope_start}, {scope_end}] extend beyond actual "
                f"data boundaries [{df_start}, {df_end}]."
            )

    def _evaluate_candidate(
        self,
        *,
        cand: CandidateSpec,
        hypothesis: ResearchHypothesis,
        df_full: pd.DataFrame,
        df_is: pd.DataFrame,
        df_val: pd.DataFrame,
        df_oos: pd.DataFrame,
        dataset_scope: DatasetScope,
        execution_assumptions: ExecutionAssumptions,
        code_provenance: CodeProvenance,
        wf_protocol: WalkForwardProtocol | None,
        wf_train_size: int | None,
        wf_test_size: int | None,
        seen_fingerprints: set[str],
    ) -> ResearchEvidence:
        """Evaluate a single candidate by delegating its accepted hypothesis to run_research_experiment."""
        evidence = run_research_experiment(
            spec=hypothesis,
            df=df_full,
            criteria=self.criteria,
            registry=self.registry,
            wf_train_size=wf_train_size,
            wf_test_size=wf_test_size,
            persist_evidence=False,
        )

        if evidence.experiment_fingerprint in seen_fingerprints and RejectionReason.DUPLICATE_CANDIDATE not in evidence.rejection_reasons:
            rejection_reasons = list(evidence.rejection_reasons) + [RejectionReason.DUPLICATE_CANDIDATE]
            evidence = ResearchEvidence(
                experiment_fingerprint=evidence.experiment_fingerprint,
                spec=evidence.spec,
                partitions=evidence.partitions,
                robustness_verdict=evidence.robustness_verdict,
                benchmark_comparison=evidence.benchmark_comparison,
                promotion_status=PromotionStatus.REJECTED,
                rejection_reasons=tuple(dict.fromkeys(rejection_reasons)),
                critique_notes=evidence.critique_notes,
                created_at_utc=evidence.created_at_utc,
            )

        return evidence
