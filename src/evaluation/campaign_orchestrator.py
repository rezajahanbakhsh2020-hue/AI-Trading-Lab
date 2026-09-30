"""Authoritative Research Campaign Orchestration & Lifecycle Control Plane.

This module evolves the Research Discovery Engine into a durable, deterministic,
resumable, and auditable research campaign lifecycle manager.

Authoritative Research Lifecycle:
    Campaign Definition
    -> Campaign Planning
    -> Deterministic Trial Plan
    -> Campaign Execution
    -> Durable Trial Checkpoints
    -> Resume / Retry
    -> Campaign Completion or Failure
    -> Immutable Campaign Evidence Manifest
    -> Research Registry / Learning Memory

Preserves the existing governed research path:
    ResearchHypothesis
    -> accept_hypothesis_for_research()
    -> ACCEPTED_FOR_RESEARCH
    -> run_research_experiment()
    -> canonical evidence
    -> canonical robustness
    -> qualification
    -> registry / learning memory
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.candidate_generator import ResearchSearchSpace
from src.evaluation.discovery_feedback import (
    evaluate_candidate_discovery_feedback,
)
from src.evaluation.hypothesis_generator import accept_hypothesis_for_research
from src.evaluation.memory_governance import (
    MemoryGovernanceDecision,
    evaluate_candidate_memory_governance,
)
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
    ResearchCampaign,
    ResearchCampaignDefinition,
    ResearchCampaignStatus,
    ResearchEvidence,
    ResearchHypothesis,
    ResearchPlannedTrial,
    ResearchTrialCheckpoint,
    ResearchTrialPlan,
    RobustnessCriteria,
    WalkForwardProtocol,
    compute_campaign_fingerprint,
    compute_criteria_fingerprint,
    compute_search_policy_fingerprint,
    resolve_walk_forward_protocol,
)
from src.evaluation.research_knowledge import ResearchKnowledgePattern
from src.evaluation.research_qualification import (
    qualify_research_evidence,
)
from src.evaluation.research_registry import (
    DoNotRepeatConstraint,
    ResearchRegistryStore,
    construct_learning_record_from_registry_record,
    construct_registry_record_from_evidence,
)
from src.evaluation.research_robustness import (
    assess_research_robustness,
)
from src.evaluation.campaign_learning import (
    CampaignLearningIntegrityError,
    CampaignLearningIntegrityValidator,
    materialize_governed_campaign_feedback,
)
from src.evaluation.campaign_synthesis import (
    CampaignSelectionIntegrityError,
    CampaignSelectionStatus,
    ResearchCampaignEvidenceSynthesis,
    ResearchCampaignSelectionDecision,
    ResearchCampaignSelectionPolicy,
    select_campaign_candidate,
    synthesize_campaign_evidence,
)
from src.evaluation.research_runner import (
    run_research_experiment,
    validate_and_prepare_dataset,
)
from src.evaluation.research_store import (
    DEFAULT_RESEARCH_DIR,
    ResearchCampaignStore,
    load_research_experiment,
    save_research_campaign,
    save_research_experiment,
)
from src.strategies.registry import DEFAULT_REGISTRY, StrategyRegistry


class CampaignIntegrityError(ValueError):
    """Raised when campaign state, plan, checkpoints, or evidence fail integrity validation."""


@dataclass(frozen=True)
class ResearchCampaignExecutionPolicy:
    """Execution policy for Research Campaign orchestration."""

    max_trials: int = 100
    fail_fast: bool = False
    max_retries: int = 1  # Number of attempts allowed for failed trials (1 = initial attempt only, no retries)
    persist_evidence: bool = True

    def __post_init__(self) -> None:
        if self.max_trials <= 0:
            raise ValueError("max_trials must be a positive integer.")
        if self.max_retries <= 0:
            raise ValueError("max_retries must be a positive integer.")


def validate_campaign_integrity(
    campaign_id: str,
    store: ResearchCampaignStore | None = None,
    research_dir: str | Path = DEFAULT_RESEARCH_DIR,
) -> None:
    """Validate persisted campaign state, trial plan, checkpoints, and evidence integrity.

    Fails closed with CampaignIntegrityError on any mismatch, inconsistency, corruption,
    or invalid governance state.
    """
    if store is None:
        store = ResearchCampaignStore()

    definition = store.load_definition(campaign_id)
    if definition.campaign_id != campaign_id:
        raise CampaignIntegrityError(
            f"Campaign ID mismatch: expected '{campaign_id}', got '{definition.campaign_id}' in definition."
        )

    # Re-verify definition fingerprint
    expected_cid = compute_campaign_fingerprint(
        search_space_fingerprint=definition.search_space_fingerprint,
        search_policy_fingerprint=definition.search_policy_fingerprint,
        criteria_fingerprint=definition.criteria_fingerprint,
        dataset_scope=definition.dataset_scope,
        execution_assumptions=definition.execution_assumptions,
        code_provenance=definition.code_provenance,
        candidate_ids=definition.candidate_ids,
    )
    if expected_cid != campaign_id:
        raise CampaignIntegrityError(
            f"Campaign definition fingerprint corrupt: recomputed '{expected_cid}' != '{campaign_id}'."
        )

    plan = store.load_trial_plan(campaign_id)
    if plan.campaign_id != campaign_id:
        raise CampaignIntegrityError(
            f"Trial plan campaign_id mismatch: expected '{campaign_id}', got '{plan.campaign_id}'."
        )
    if plan.definition_fingerprint != definition.definition_fingerprint:
        raise CampaignIntegrityError(
            f"Trial plan definition_fingerprint mismatch: expected '{definition.definition_fingerprint}', "
            f"got '{plan.definition_fingerprint}'."
        )

    # Validate trial IDs uniqueness and identity match
    planned_trial_map = {t.trial_id: t for t in plan.trials}
    if len(planned_trial_map) != len(plan.trials):
        raise CampaignIntegrityError(f"Duplicate trial IDs detected in trial plan for campaign '{campaign_id}'.")

    checkpoints = store.list_trial_checkpoints(campaign_id)
    checkpoint_map = {cp.trial_id: cp for cp in checkpoints}

    for trial_id, cp in checkpoint_map.items():
        if trial_id not in planned_trial_map:
            raise CampaignIntegrityError(
                f"Unknown trial_id '{trial_id}' in checkpoints not found in trial plan for campaign '{campaign_id}'."
            )
        planned_trial = planned_trial_map[trial_id]
        if cp.candidate_id != planned_trial.candidate_id:
            raise CampaignIntegrityError(
                f"Candidate identity mismatch in trial '{trial_id}': checkpoint has '{cp.candidate_id}', "
                f"planned trial has '{planned_trial.candidate_id}'."
            )
        if cp.campaign_id != campaign_id:
            raise CampaignIntegrityError(
                f"Campaign ID mismatch in trial checkpoint '{trial_id}': expected '{campaign_id}', got '{cp.campaign_id}'."
            )

        # Completed/Qualified/Rejected trials must have valid evidence
        if cp.status in ("COMPLETED", "QUALIFIED", "REJECTED"):
            if not cp.experiment_fingerprint:
                raise CampaignIntegrityError(
                    f"Completed trial '{trial_id}' missing experiment_fingerprint in checkpoint."
                )
            if not cp.evidence_fingerprint:
                raise CampaignIntegrityError(
                    f"Completed trial '{trial_id}' missing evidence_fingerprint in checkpoint."
                )
            # Load evidence artifact and verify fingerprint matching
            ev_path = Path(research_dir) / cp.experiment_fingerprint / "evidence.json"
            if not ev_path.exists():
                raise CampaignIntegrityError(
                    f"Missing research evidence artifact for completed trial '{trial_id}' at: {ev_path}"
                )
            evidence = load_research_experiment(ev_path, base_dir=research_dir)
            if evidence.experiment_fingerprint != cp.experiment_fingerprint:
                raise CampaignIntegrityError(
                    f"Trial '{trial_id}' evidence fingerprint inconsistency: file has "
                    f"'{evidence.experiment_fingerprint}', checkpoint has '{cp.experiment_fingerprint}'."
                )

    state_data = store.load_lifecycle_state(campaign_id)
    status_str = state_data.get("status")
    if status_str not in ResearchCampaignStatus.__members__:
        raise CampaignIntegrityError(f"Invalid campaign lifecycle state '{status_str}' in store.")

    current_status = ResearchCampaignStatus(status_str)

    if current_status in (ResearchCampaignStatus.COMPLETED, ResearchCampaignStatus.TRUNCATED):
        # All planned trials must be resolved (no PENDING or RUNNING trials allowed)
        for planned_trial in plan.trials:
            cp = checkpoint_map.get(planned_trial.trial_id)
            if cp is None or cp.status in ("PENDING", "RUNNING"):
                raise CampaignIntegrityError(
                    f"Campaign '{campaign_id}' is marked '{current_status.value}' but trial "
                    f"'{planned_trial.trial_id}' remains unresolved (status: '{cp.status if cp else 'MISSING'}')."
                )


class ResearchCampaignOrchestrator:
    """Authoritative orchestration engine for durable Research Campaigns.

    Manages campaign lifecycle state transitions, trial plan generation, checkpoint persistence,
    idempotent execution/resume, safe retries, and integrity validation.
    """

    def __init__(
        self,
        store: ResearchCampaignStore | None = None,
        registry: StrategyRegistry = DEFAULT_REGISTRY,
        memory_store: ResearchRegistryStore | None = None,
    ) -> None:
        self.store = store or ResearchCampaignStore()
        self.registry = registry
        self.memory_store = memory_store

    def plan_campaign(
        self,
        search_space: ResearchSearchSpace,
        dataset_scope: DatasetScope,
        execution_assumptions: ExecutionAssumptions,
        code_provenance: CodeProvenance,
        criteria: Any,
        execution_policy: ResearchCampaignExecutionPolicy | None = None,
        methodology_version: str = "campaign_v1.0",
        walk_forward_protocol: WalkForwardProtocol | None = None,
        governance_constraints: dict[str, Any] | None = None,
        memory_policy_id: str = "default_memory_policy_v1",
    ) -> tuple[ResearchCampaignDefinition, ResearchTrialPlan]:
        """Create and materialize a deterministic campaign definition and trial plan."""
        policy = execution_policy or ResearchCampaignExecutionPolicy()
        policy_fp = compute_search_policy_fingerprint(
            max_trials=policy.max_trials,
            fail_fast=policy.fail_fast,
        )
        criteria_fp = compute_criteria_fingerprint(criteria)

        eval_candidates = search_space.candidate_definitions
        if len(eval_candidates) > policy.max_trials:
            eval_candidates = eval_candidates[: policy.max_trials]

        cand_ids = tuple(c.candidate_id for c in eval_candidates)

        definition = ResearchCampaignDefinition(
            search_space_fingerprint=search_space.search_fingerprint,
            search_policy_fingerprint=policy_fp,
            criteria_fingerprint=criteria_fp,
            dataset_scope=dataset_scope,
            execution_assumptions=execution_assumptions,
            code_provenance=code_provenance,
            methodology_version=methodology_version,
            candidate_ids=cand_ids,
            trial_count=len(cand_ids),
            walk_forward_protocol=walk_forward_protocol,
            governance_constraints=governance_constraints or {},
            memory_policy_id=memory_policy_id,
        )

        planned_trials = []
        for idx, cand in enumerate(eval_candidates):
            trial_id = f"{definition.campaign_id}_tr_{idx:04d}_{cand.candidate_id}"
            pt = ResearchPlannedTrial(
                campaign_id=definition.campaign_id,
                trial_id=trial_id,
                trial_index=idx,
                candidate_id=cand.candidate_id,
                candidate_fingerprint=cand.candidate_id,
                hypothesis_fingerprint=cand.candidate_id,
                strategy_name=cand.strategy_name,
                strategy_version=getattr(criteria, "strategy_version", "1.0.0"),
                dataset_id=dataset_scope.dataset_id,
                execution_assumptions_id=f"ea_{hashlib.sha256(str(execution_assumptions).encode('utf-8')).hexdigest()[:8]}",
                planned_status="PENDING",
            )
            planned_trials.append(pt)

        plan = ResearchTrialPlan(
            campaign_id=definition.campaign_id,
            definition_fingerprint=definition.definition_fingerprint,
            trials=tuple(planned_trials),
        )

        # Save definition, trial plan, and initial PLANNED state in store
        self.store.save_definition(definition)
        self.store.save_trial_plan(plan)
        self.store.save_lifecycle_state(definition.campaign_id, ResearchCampaignStatus.PLANNED)

        # Initialize trial checkpoints as PENDING
        for pt in planned_trials:
            cp = ResearchTrialCheckpoint(
                trial_id=pt.trial_id,
                campaign_id=definition.campaign_id,
                candidate_id=pt.candidate_id,
                trial_index=pt.trial_index,
                attempt_number=1,
                status="PENDING",
            )
            self.store.save_trial_checkpoint(cp)

        return definition, plan

    def execute_campaign(
        self,
        campaign_id: str,
        df: pd.DataFrame,
        search_space: ResearchSearchSpace,
        criteria: Any,
        execution_policy: ResearchCampaignExecutionPolicy | None = None,
        selection_policy: ResearchCampaignSelectionPolicy | None = None,
        val_ratio: float = 0.2,
        oos_ratio: float = 0.3,
        wf_train_size: int | None = None,
        wf_test_size: int | None = None,
        persist_registry_dir: str | Path | None = None,
        memory_store: ResearchRegistryStore | Sequence[DoNotRepeatConstraint] | None = None,
        enable_memory_governance: bool = True,
        knowledge_patterns: Sequence[ResearchKnowledgePattern] | None = None,
        enable_discovery_feedback: bool = True,
    ) -> ResearchCampaign:
        """Execute or resume a Research Campaign using durable trial checkpoints and governed research."""
        policy = execution_policy or ResearchCampaignExecutionPolicy()

        # Validate campaign integrity before execution
        validate_campaign_integrity(campaign_id, store=self.store)

        definition = self.store.load_definition(campaign_id)
        plan = self.store.load_trial_plan(campaign_id)
        curr_state = self.store.load_lifecycle_state(campaign_id)

        current_status = ResearchCampaignStatus(curr_state["status"])
        if current_status in (
            ResearchCampaignStatus.COMPLETED,
            ResearchCampaignStatus.TRUNCATED,
        ):
            # Finalize completed campaign (idempotent / resumes missing finalization phase)
            eff_registry = (
                persist_registry_dir
                if persist_registry_dir
                else (self.memory_store if isinstance(self.memory_store, ResearchRegistryStore) else None)
            )
            self.finalize_completed_campaign(
                campaign_id,
                selection_policy=selection_policy,
                registry_store=eff_registry,
            )
            try:
                from src.evaluation.research_store import load_research_campaign
                return load_research_campaign(campaign_id, base_dir=self.store.base_dir)
            except Exception:
                all_cps = self.store.list_trial_checkpoints(campaign_id)
                executed_count = sum(1 for c in all_cps if c.status in ("COMPLETED", "QUALIFIED", "REJECTED"))
                failed_count = sum(1 for c in all_cps if c.status == "FAILED")
                blocked_count = sum(1 for c in all_cps if c.status == "BLOCKED")
                selected_ids = tuple(c.candidate_id for c in all_cps if c.status == "QUALIFIED")
                ev_fps = tuple(c.evidence_fingerprint for c in all_cps if c.evidence_fingerprint)

                return ResearchCampaign(
                    campaign_id=campaign_id,
                    search_space_fingerprint=definition.search_space_fingerprint,
                    search_policy_fingerprint=definition.search_policy_fingerprint,
                    criteria_fingerprint=definition.criteria_fingerprint,
                    dataset_scope=definition.dataset_scope,
                    execution_assumptions=definition.execution_assumptions,
                    code_provenance=definition.code_provenance,
                    candidate_ids=definition.candidate_ids,
                    evidence_fingerprints=ev_fps,
                    selected_candidate_ids=selected_ids,
                    status=current_status,
                    created_at_utc=datetime.now(timezone.utc).isoformat(),
                    definition_fingerprint=definition.definition_fingerprint,
                    trial_plan_fingerprint=plan.plan_fingerprint,
                    executed_trial_count=executed_count,
                    failed_trial_count=failed_count,
                    blocked_trial_count=blocked_count,
                )

        if current_status in (
            ResearchCampaignStatus.FAILED,
            ResearchCampaignStatus.CANCELLED,
        ):
            # Already terminal (failed/cancelled) -> load and return existing campaign manifest
            try:
                from src.evaluation.research_store import load_research_campaign
                return load_research_campaign(campaign_id, base_dir=self.store.base_dir)
            except Exception:
                all_cps = self.store.list_trial_checkpoints(campaign_id)
                executed_count = sum(1 for c in all_cps if c.status in ("COMPLETED", "QUALIFIED", "REJECTED"))
                failed_count = sum(1 for c in all_cps if c.status == "FAILED")
                blocked_count = sum(1 for c in all_cps if c.status == "BLOCKED")
                selected_ids = tuple(c.candidate_id for c in all_cps if c.status == "QUALIFIED")
                ev_fps = tuple(c.evidence_fingerprint for c in all_cps if c.evidence_fingerprint)

                return ResearchCampaign(
                    campaign_id=campaign_id,
                    search_space_fingerprint=definition.search_space_fingerprint,
                    search_policy_fingerprint=definition.search_policy_fingerprint,
                    criteria_fingerprint=definition.criteria_fingerprint,
                    dataset_scope=definition.dataset_scope,
                    execution_assumptions=definition.execution_assumptions,
                    code_provenance=definition.code_provenance,
                    candidate_ids=definition.candidate_ids,
                    evidence_fingerprints=ev_fps,
                    selected_candidate_ids=selected_ids,
                    status=current_status,
                    created_at_utc=datetime.now(timezone.utc).isoformat(),
                    definition_fingerprint=definition.definition_fingerprint,
                    trial_plan_fingerprint=plan.plan_fingerprint,
                    executed_trial_count=executed_count,
                    failed_trial_count=failed_count,
                    blocked_trial_count=blocked_count,
                )

        # Transition state to RUNNING for PLANNED or PAUSED campaigns
        self.store.save_lifecycle_state(
            campaign_id,
            ResearchCampaignStatus.RUNNING,
            reason="Campaign execution started",
        )

        data = validate_and_prepare_dataset(df, definition.dataset_scope)
        n = len(data)
        wf_protocol = definition.walk_forward_protocol or resolve_walk_forward_protocol(
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

        df_is = data.iloc[:n_is]
        df_val = data.iloc[n_is : n_is + n_val]
        df_oos = data.iloc[n_is + n_val :]

        candidate_map = {c.candidate_id: c for c in search_space.candidate_definitions}

        registry_store = (
            ResearchRegistryStore(base_dir=persist_registry_dir)
            if policy.persist_evidence and persist_registry_dir
            else (ResearchRegistryStore() if policy.persist_evidence else None)
        )

        active_constraints: tuple[DoNotRepeatConstraint, ...] = ()
        if enable_memory_governance:
            if isinstance(memory_store, (list, tuple)):
                active_constraints = tuple(c for c in memory_store if isinstance(c, DoNotRepeatConstraint))
            else:
                eff_store = memory_store if memory_store is not None else self.memory_store
                if eff_store is not None:
                    try:
                        active_constraints = eff_store.get_active_do_not_repeat_constraints(
                            symbol=definition.dataset_scope.symbol,
                            timeframe=definition.dataset_scope.timeframe,
                        )
                    except Exception:
                        active_constraints = ()

        effective_patterns: tuple[ResearchKnowledgePattern, ...] = ()
        if enable_discovery_feedback:
            if knowledge_patterns is not None:
                effective_patterns = tuple(k for k in knowledge_patterns if isinstance(k, ResearchKnowledgePattern))
            else:
                eff_store = memory_store if isinstance(memory_store, ResearchRegistryStore) else self.memory_store
                if eff_store is not None and hasattr(eff_store, "list_patterns"):
                    try:
                        effective_patterns = eff_store.list_patterns()
                    except Exception:
                        effective_patterns = ()

        evidence_list: list[ResearchEvidence] = []
        selected_candidate_ids: list[str] = []

        # Load existing checkpoints
        checkpoints_map = {cp.trial_id: cp for cp in self.store.list_trial_checkpoints(campaign_id)}

        for planned_trial in plan.trials:
            cp = checkpoints_map.get(planned_trial.trial_id)
            if cp is None:
                cp = ResearchTrialCheckpoint(
                    trial_id=planned_trial.trial_id,
                    campaign_id=campaign_id,
                    candidate_id=planned_trial.candidate_id,
                    trial_index=planned_trial.trial_index,
                    attempt_number=1,
                    status="PENDING",
                )

            # Idempotent skip for completed/qualified/rejected/blocked trials
            if cp.status in ("COMPLETED", "QUALIFIED", "REJECTED", "BLOCKED"):
                if cp.status == "QUALIFIED":
                    selected_candidate_ids.append(cp.candidate_id)
                if cp.experiment_fingerprint and policy.persist_evidence:
                    try:
                        ev_path = Path(DEFAULT_RESEARCH_DIR) / cp.experiment_fingerprint / "evidence.json"
                        if ev_path.exists():
                            ev = load_research_experiment(ev_path, base_dir=DEFAULT_RESEARCH_DIR)
                            evidence_list.append(ev)
                    except Exception:
                        pass
                continue

            # Handle retries for failed trials
            if cp.status == "FAILED":
                if cp.attempt_number >= policy.max_retries:
                    # Retry threshold reached, keep failed status
                    continue
                # Increment attempt counter for retry
                cp = ResearchTrialCheckpoint(
                    trial_id=cp.trial_id,
                    campaign_id=cp.campaign_id,
                    candidate_id=cp.candidate_id,
                    trial_index=cp.trial_index,
                    attempt_number=cp.attempt_number + 1,
                    status="RUNNING",
                    execution_history=cp.execution_history + ({
                        "attempt": cp.attempt_number,
                        "status": "FAILED",
                        "error": cp.error_message,
                        "failed_at_utc": cp.updated_at_utc,
                    },),
                )
            else:
                cp = ResearchTrialCheckpoint(
                    trial_id=cp.trial_id,
                    campaign_id=cp.campaign_id,
                    candidate_id=cp.candidate_id,
                    trial_index=cp.trial_index,
                    attempt_number=cp.attempt_number,
                    status="RUNNING",
                )

            # Persist RUNNING trial checkpoint
            self.store.save_trial_checkpoint(cp)

            cand = candidate_map.get(planned_trial.candidate_id)
            if cand is None:
                # Missing candidate spec -> fail trial
                cp_failed = ResearchTrialCheckpoint(
                    trial_id=cp.trial_id,
                    campaign_id=campaign_id,
                    candidate_id=cp.candidate_id,
                    trial_index=cp.trial_index,
                    attempt_number=cp.attempt_number,
                    status="FAILED",
                    rejection_reasons=("SPECIFICATION_INVALID",),
                    error_message=f"Candidate spec '{planned_trial.candidate_id}' missing in search space.",
                    execution_history=cp.execution_history,
                )
                self.store.save_trial_checkpoint(cp_failed)
                continue

            hypothesis_stmt = (
                cand.hypothesis_template.replace("{candidate_id}", cand.candidate_id)
                if cand.hypothesis_template
                else f"Hypothesis for candidate {cand.candidate_id}"
            )
            raw_hypothesis = ResearchHypothesis(
                statement=hypothesis_stmt,
                methodology_version=definition.methodology_version,
                strategy_name=cand.strategy_name,
                strategy_version=getattr(criteria, "strategy_version", "1.0.0"),
                dataset_scope=definition.dataset_scope,
                execution_assumptions=definition.execution_assumptions,
                code_provenance=definition.code_provenance,
                benchmark_reference=getattr(criteria, "benchmark_reference", "buy_and_hold"),
                parameters=dict(cand.parameters),
                random_seed=cand.random_seed,
                walk_forward_protocol=wf_protocol,
            )

            # Accept hypothesis for research governance
            hypothesis = accept_hypothesis_for_research(raw_hypothesis)

            # Stage 0a: Controlled Discovery Feedback
            if enable_discovery_feedback:
                cand_feedbacks = evaluate_candidate_discovery_feedback(
                    candidate=cand,
                    dataset_scope=definition.dataset_scope,
                    execution_assumptions=definition.execution_assumptions,
                    code_provenance=definition.code_provenance,
                    knowledge_patterns=effective_patterns,
                    search_id=search_space.search_id,
                    search_fingerprint=search_space.search_fingerprint,
                    registry_store=registry_store,
                    methodology_version=definition.methodology_version,
                )
                if registry_store is not None:
                    for fb in cand_feedbacks:
                        try:
                            registry_store.register_feedback(fb)
                        except Exception:
                            pass

            # Stage 0b: Memory Governance
            if enable_memory_governance and active_constraints:
                gov_res = evaluate_candidate_memory_governance(
                    candidate=cand,
                    dataset_scope=definition.dataset_scope,
                    execution_assumptions=definition.execution_assumptions,
                    code_provenance=definition.code_provenance,
                    active_constraints=active_constraints,
                    search_id=search_space.search_id,
                    search_fingerprint=search_space.search_fingerprint,
                    methodology_version=definition.methodology_version,
                )

                if gov_res.decision == MemoryGovernanceDecision.BLOCKED:
                    cp_blocked = ResearchTrialCheckpoint(
                        trial_id=cp.trial_id,
                        campaign_id=campaign_id,
                        candidate_id=cp.candidate_id,
                        trial_index=cp.trial_index,
                        attempt_number=cp.attempt_number,
                        status="BLOCKED",
                        qualification_status="REJECTED",
                        rejection_reasons=("GOVERNANCE_BLOCKED",),
                        error_message=gov_res.reason,
                        execution_history=cp.execution_history,
                    )
                    self.store.save_trial_checkpoint(cp_blocked)
                    continue

                elif gov_res.decision == MemoryGovernanceDecision.FAIL_CLOSED:
                    cp_failed = ResearchTrialCheckpoint(
                        trial_id=cp.trial_id,
                        campaign_id=campaign_id,
                        candidate_id=cp.candidate_id,
                        trial_index=cp.trial_index,
                        attempt_number=cp.attempt_number,
                        status="FAILED",
                        qualification_status="REJECTED",
                        rejection_reasons=("SPECIFICATION_INVALID", "GOVERNANCE_BLOCKED"),
                        error_message=gov_res.reason,
                        execution_history=cp.execution_history,
                    )
                    self.store.save_trial_checkpoint(cp_failed)
                    if policy.fail_fast:
                        self.store.save_lifecycle_state(
                            campaign_id, ResearchCampaignStatus.FAILED, reason=gov_res.reason
                        )
                        raise CampaignIntegrityError(f"Memory governance failed closed: {gov_res.reason}")
                    continue

            # Execute research experiment
            try:
                spec = hypothesis.to_experiment_spec()
                evidence = run_research_experiment(
                    spec=spec,
                    df=data,
                    criteria=criteria,
                    registry=self.registry,
                    wf_train_size=wf_train_size,
                    wf_test_size=wf_test_size,
                    persist_evidence=False,
                )
            except Exception as exc:
                cp_failed = ResearchTrialCheckpoint(
                    trial_id=cp.trial_id,
                    campaign_id=campaign_id,
                    candidate_id=cp.candidate_id,
                    trial_index=cp.trial_index,
                    attempt_number=cp.attempt_number,
                    status="FAILED",
                    rejection_reasons=("SPECIFICATION_INVALID",),
                    error_message=str(exc),
                    execution_history=cp.execution_history,
                )
                self.store.save_trial_checkpoint(cp_failed)
                if policy.fail_fast:
                    self.store.save_lifecycle_state(
                        campaign_id, ResearchCampaignStatus.FAILED, reason=str(exc)
                    )
                    raise
                continue

            # CRASH RECOVERY: Persist evidence to disk BEFORE updating trial checkpoint to COMPLETED/QUALIFIED/REJECTED
            if policy.persist_evidence:
                save_research_experiment(evidence, base_dir=DEFAULT_RESEARCH_DIR)

            # Canonical Robustness Assessment & Qualification
            rob_assessment = assess_research_robustness(
                evidence=evidence,
                robustness_criteria=getattr(criteria, "robustness_criteria", RobustnessCriteria()),
            )
            qual_res = qualify_research_evidence(evidence, robustness_assessment=rob_assessment)

            evidence_list.append(evidence)

            if qual_res.qualified:
                final_status = "QUALIFIED"
                selected_candidate_ids.append(cp.candidate_id)
            else:
                final_status = "REJECTED"

            cp_complete = ResearchTrialCheckpoint(
                trial_id=cp.trial_id,
                campaign_id=campaign_id,
                candidate_id=cp.candidate_id,
                trial_index=cp.trial_index,
                attempt_number=cp.attempt_number,
                status=final_status,
                experiment_fingerprint=evidence.experiment_fingerprint,
                evidence_fingerprint=evidence.evidence_id,
                qualification_status=qual_res.status.value if hasattr(qual_res.status, "value") else str(qual_res.status),
                rejection_reasons=tuple(r.value if hasattr(r, "value") else str(r) for r in qual_res.rejection_reasons),
                execution_history=cp.execution_history,
            )
            self.store.save_trial_checkpoint(cp_complete)

            # Register in learning / registry memory
            if registry_store is not None:
                rec = construct_registry_record_from_evidence(
                    evidence=evidence,
                    candidate_id=cp.candidate_id,
                    search_id=search_space.search_id,
                    search_fingerprint=search_space.search_fingerprint,
                    trial_id=cp.trial_id,
                    trial_index=cp.trial_index,
                    robustness_assessment=rob_assessment,
                    qualification_status=final_status,
                )
                registry_store.register(rec)
                learning_rec = construct_learning_record_from_registry_record(rec)
                registry_store.register_learning_record(learning_rec)

        # Re-evaluate all checkpoints
        all_cps = self.store.list_trial_checkpoints(campaign_id)
        executed_count = sum(1 for c in all_cps if c.status in ("COMPLETED", "QUALIFIED", "REJECTED"))
        failed_count = sum(1 for c in all_cps if c.status == "FAILED")
        blocked_count = sum(1 for c in all_cps if c.status == "BLOCKED")

        is_truncated = len(search_space.candidate_definitions) > len(definition.candidate_ids)
        final_campaign_status = (
            ResearchCampaignStatus.TRUNCATED if is_truncated else ResearchCampaignStatus.COMPLETED
        )

        ev_fps = tuple(ev.evidence_id for ev in evidence_list if ev.evidence_id)

        campaign = ResearchCampaign(
            campaign_id=campaign_id,
            search_space_fingerprint=definition.search_space_fingerprint,
            search_policy_fingerprint=definition.search_policy_fingerprint,
            criteria_fingerprint=definition.criteria_fingerprint,
            dataset_scope=definition.dataset_scope,
            execution_assumptions=definition.execution_assumptions,
            code_provenance=definition.code_provenance,
            candidate_ids=definition.candidate_ids,
            evidence_fingerprints=ev_fps,
            selected_candidate_ids=tuple(selected_candidate_ids),
            status=final_campaign_status,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            definition_fingerprint=definition.definition_fingerprint,
            trial_plan_fingerprint=plan.plan_fingerprint,
            executed_trial_count=executed_count,
            failed_trial_count=failed_count,
            blocked_trial_count=blocked_count,
        )

        if policy.persist_evidence:
            save_research_campaign(campaign, base_dir=self.store.base_dir)

        self.store.save_lifecycle_state(
            campaign_id,
            final_campaign_status,
            reason="Campaign execution completed cleanly",
        )

        if final_campaign_status in (
            ResearchCampaignStatus.COMPLETED,
            ResearchCampaignStatus.TRUNCATED,
        ):
            eff_registry = (
                registry_store
                if registry_store is not None
                else (self.memory_store if isinstance(self.memory_store, ResearchRegistryStore) else None)
            )
            self.finalize_completed_campaign(
                campaign_id,
                selection_policy=selection_policy,
                registry_store=eff_registry,
            )

        return campaign

    def finalize_completed_campaign(
        self,
        campaign_id: str,
        *,
        selection_policy: ResearchCampaignSelectionPolicy | None = None,
        registry_store: ResearchRegistryStore | None = None,
        experiment_store_dir: str | Path | None = None,
    ) -> ResearchCampaignSelectionDecision:
        """Canonical orchestration finalization boundary for completed or truncated campaigns.

        Responsibilities:
        1. Load durable campaign lifecycle state.
        2. Allow finalization only for COMPLETED and TRUNCATED campaigns; fail closed for others.
        3. Validate campaign integrity before finalization.
        4. Synthesize campaign evidence using canonical synthesis API & check/save synthesis artifact idempotently.
        5. Execute candidate selection using canonical selection API & check/save decision artifact idempotently.
        6. If decision status is SELECTED, derive/materialize governed campaign learning and validate artifact.
        7. If decision status is non-selected (e.g. NO_ELIGIBLE_CANDIDATE, TIE_UNRESOLVED), persist decision without learning.
        8. Return authoritative selection decision.
        """
        # 1. Load lifecycle state
        try:
            state_data = self.store.load_lifecycle_state(campaign_id)
            status_str = state_data.get("status")
        except Exception as exc:
            raise CampaignIntegrityError(
                f"Failed to load campaign lifecycle state for '{campaign_id}': {exc}"
            ) from exc

        # 2. Validate terminal completion state
        if status_str not in (
            ResearchCampaignStatus.COMPLETED.value,
            ResearchCampaignStatus.TRUNCATED.value,
        ):
            raise CampaignIntegrityError(
                f"Cannot finalize campaign '{campaign_id}' in lifecycle state '{status_str}'. "
                "Only COMPLETED or TRUNCATED campaigns can be finalized."
            )

        # 3. Validate campaign structural & evidence integrity
        validate_campaign_integrity(campaign_id, store=self.store, research_dir=experiment_store_dir or DEFAULT_RESEARCH_DIR)

        sel_policy = selection_policy or ResearchCampaignSelectionPolicy()
        eff_registry_store = (
            registry_store
            if registry_store is not None
            else (self.memory_store if isinstance(self.memory_store, ResearchRegistryStore) else ResearchRegistryStore())
        )

        # 4. Canonical evidence synthesis
        try:
            derived_synthesis = synthesize_campaign_evidence(
                campaign_id=campaign_id,
                selection_policy=sel_policy,
                store=self.store,
                experiment_store_dir=experiment_store_dir,
            )
        except Exception as exc:
            if isinstance(exc, (CampaignIntegrityError, CampaignSelectionIntegrityError)):
                raise CampaignIntegrityError(str(exc)) from exc
            raise CampaignIntegrityError(
                f"Campaign evidence synthesis failed for campaign '{campaign_id}': {exc}"
            ) from exc

        try:
            existing_synthesis = self.store.load_evidence_synthesis(campaign_id)
            if existing_synthesis.synthesis_fingerprint != derived_synthesis.synthesis_fingerprint:
                raise CampaignIntegrityError(
                    f"Conflicting persisted evidence synthesis for campaign '{campaign_id}': "
                    f"existing fingerprint '{existing_synthesis.synthesis_fingerprint}', "
                    f"derived fingerprint '{derived_synthesis.synthesis_fingerprint}'."
                )
            synthesis = existing_synthesis
        except FileNotFoundError:
            try:
                self.store.save_evidence_synthesis(derived_synthesis)
                synthesis = derived_synthesis
            except Exception as exc:
                raise CampaignIntegrityError(
                    f"Failed to persist evidence synthesis for campaign '{campaign_id}': {exc}"
                ) from exc
        except CampaignIntegrityError:
            raise
        except Exception as exc:
            raise CampaignIntegrityError(
                f"Conflicting persisted evidence synthesis for campaign '{campaign_id}': {exc}"
            ) from exc

        # 5. Canonical selection decision
        try:
            derived_decision = select_campaign_candidate(
                synthesis=synthesis,
                selection_policy=sel_policy,
            )
        except Exception as exc:
            if isinstance(exc, (CampaignIntegrityError, CampaignSelectionIntegrityError)):
                raise CampaignIntegrityError(str(exc)) from exc
            raise CampaignIntegrityError(
                f"Campaign candidate selection failed for campaign '{campaign_id}': {exc}"
            ) from exc

        try:
            existing_decision = self.store.load_selection_decision(campaign_id)
            if existing_decision.decision_fingerprint != derived_decision.decision_fingerprint:
                raise CampaignIntegrityError(
                    f"Conflicting persisted selection decision for campaign '{campaign_id}': "
                    f"existing fingerprint '{existing_decision.decision_fingerprint}', "
                    f"derived fingerprint '{derived_decision.decision_fingerprint}'."
                )
            decision = existing_decision
        except FileNotFoundError:
            try:
                self.store.save_selection_decision(derived_decision)
                decision = derived_decision
            except Exception as exc:
                raise CampaignIntegrityError(
                    f"Failed to persist selection decision for campaign '{campaign_id}': {exc}"
                ) from exc
        except CampaignIntegrityError:
            raise
        except Exception as exc:
            raise CampaignIntegrityError(
                f"Conflicting persisted selection decision for campaign '{campaign_id}': {exc}"
            ) from exc

        # 6. Governed learning materialization (ONLY when status is SELECTED)
        if decision.decision_status == CampaignSelectionStatus.SELECTED.value:
            try:
                existing_learning = self.store.load_campaign_learning(campaign_id)
                # Verify existing learning artifact integrity and matching decision/synthesis fingerprints
                if existing_learning.synthesis_fingerprint != decision.synthesis_fingerprint:
                    raise CampaignIntegrityError(
                        f"Conflicting persisted campaign learning artifact synthesis_fingerprint for '{campaign_id}': "
                        f"existing '{existing_learning.synthesis_fingerprint}', expected '{decision.synthesis_fingerprint}'."
                    )
                if existing_learning.campaign_selection_decision_fingerprint != decision.decision_fingerprint:
                    raise CampaignIntegrityError(
                        f"Conflicting persisted campaign learning artifact decision_fingerprint for '{campaign_id}': "
                        f"existing '{existing_learning.campaign_selection_decision_fingerprint}', expected '{decision.decision_fingerprint}'."
                    )
                CampaignLearningIntegrityValidator.validate_campaign_learning_integrity(
                    existing_learning,
                    store=self.store,
                    registry_store=eff_registry_store,
                )
            except FileNotFoundError:
                try:
                    learning_art = materialize_governed_campaign_feedback(
                        campaign_id=campaign_id,
                        store=self.store,
                        registry_store=eff_registry_store,
                    )
                    CampaignLearningIntegrityValidator.validate_campaign_learning_integrity(
                        learning_art,
                        store=self.store,
                        registry_store=eff_registry_store,
                    )
                except Exception as exc:
                    if isinstance(exc, (CampaignIntegrityError, CampaignLearningIntegrityError)):
                        raise CampaignIntegrityError(str(exc)) from exc
                    raise CampaignIntegrityError(
                        f"Failed to materialize governed campaign learning for campaign '{campaign_id}': {exc}"
                    ) from exc
            except CampaignIntegrityError:
                raise
            except Exception as exc:
                raise CampaignIntegrityError(
                    f"Failed to load or verify existing campaign learning artifact for '{campaign_id}': {exc}"
                ) from exc

        return decision

    def resume_campaign(
        self,
        campaign_id: str,
        df: pd.DataFrame,
        search_space: ResearchSearchSpace,
        criteria: Any,
        execution_policy: ResearchCampaignExecutionPolicy | None = None,
        selection_policy: ResearchCampaignSelectionPolicy | None = None,
        persist_registry_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> ResearchCampaign:
        """Resume an interrupted or paused Research Campaign."""
        state = self.store.load_lifecycle_state(campaign_id)
        status = ResearchCampaignStatus(state["status"])
        if status in (
            ResearchCampaignStatus.COMPLETED,
            ResearchCampaignStatus.TRUNCATED,
        ):
            # Finalize completed campaign (idempotent / resumes missing finalization phase)
            eff_registry = (
                persist_registry_dir
                if persist_registry_dir
                else (self.memory_store if isinstance(self.memory_store, ResearchRegistryStore) else None)
            )
            self.finalize_completed_campaign(
                campaign_id,
                selection_policy=selection_policy,
                registry_store=eff_registry,
            )
            try:
                from src.evaluation.research_store import load_research_campaign
                return load_research_campaign(campaign_id, base_dir=self.store.base_dir)
            except Exception:
                pass

        if status in (
            ResearchCampaignStatus.FAILED,
            ResearchCampaignStatus.CANCELLED,
        ):
            # Already terminal -> load and return campaign manifest
            try:
                from src.evaluation.research_store import load_research_campaign
                return load_research_campaign(campaign_id, base_dir=self.store.base_dir)
            except Exception:
                pass

        return self.execute_campaign(
            campaign_id=campaign_id,
            df=df,
            search_space=search_space,
            criteria=criteria,
            execution_policy=execution_policy,
            selection_policy=selection_policy,
            persist_registry_dir=persist_registry_dir,
            **kwargs,
        )

    def pause_campaign(self, campaign_id: str, reason: str = "") -> None:
        """Pause a running campaign."""
        self.store.save_lifecycle_state(
            campaign_id,
            ResearchCampaignStatus.PAUSED,
            reason=reason or "Campaign execution paused by user",
        )

    def cancel_campaign(self, campaign_id: str, reason: str = "") -> None:
        """Cancel a planned, running, or paused campaign."""
        self.store.save_lifecycle_state(
            campaign_id,
            ResearchCampaignStatus.CANCELLED,
            reason=reason or "Campaign execution cancelled by user",
        )

    def get_campaign_status(self, campaign_id: str) -> dict[str, Any]:
        """Query current campaign lifecycle state, definition, plan, and progress."""
        validate_campaign_integrity(campaign_id, store=self.store)
        state = self.store.load_lifecycle_state(campaign_id)
        defn = self.store.load_definition(campaign_id)
        plan = self.store.load_trial_plan(campaign_id)
        cps = self.store.list_trial_checkpoints(campaign_id)

        completed_count = sum(1 for c in cps if c.status in ("COMPLETED", "QUALIFIED", "REJECTED"))
        failed_count = sum(1 for c in cps if c.status == "FAILED")
        blocked_count = sum(1 for c in cps if c.status == "BLOCKED")
        pending_count = sum(1 for c in cps if c.status in ("PENDING", "RUNNING"))

        return {
            "campaign_id": campaign_id,
            "definition_fingerprint": defn.definition_fingerprint,
            "plan_fingerprint": plan.plan_fingerprint,
            "status": state["status"],
            "reason": state.get("reason", ""),
            "updated_at_utc": state.get("updated_at_utc", ""),
            "total_planned_trials": len(plan.trials),
            "completed_trials": completed_count,
            "failed_trials": failed_count,
            "blocked_trials": blocked_count,
            "pending_trials": pending_count,
        }
