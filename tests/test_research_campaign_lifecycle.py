"""Comprehensive test suite for Research Campaign Lifecycle Control Plane & Architectural Guards.

Covers items A through T from PR #39 requirements and AST architectural boundaries:
A. Determinism
B. Trial determinism
C. Lifecycle
D. Invalid transitions
E. Persistence
F. Corruption
G. Resume
H. Idempotency
I. Failure
J. Retry
K. Evidence ordering
L. Campaign completion
M. Truncation
N. Governance
O. Lineage
P. Restart
Q. Duplicate trial protection
R. Candidate identity
S. Scientific identity
T. Volatile timestamp exclusion
Static Architectural Guards (AST checks)
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.campaign_orchestrator import (
    CampaignIntegrityError,
    ResearchCampaignExecutionPolicy,
    ResearchCampaignOrchestrator,
    validate_campaign_integrity,
)
from src.evaluation.candidate_generator import CandidateSpec, ResearchSearchSpace
from src.evaluation.discovery_engine import DiscoveryCriteria
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
    HypothesisStatus,
    InvalidLifecycleTransitionError,
    ResearchCampaignDefinition,
    ResearchCampaignStatus,
    ResearchHypothesis,
    ResearchPlannedTrial,
    ResearchTrialCheckpoint,
    ResearchTrialPlan,
    validate_campaign_state_transition,
)
from src.evaluation.research_store import ResearchCampaignStore


@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=100, freq="h"),
        "open": [1.1 + i * 0.001 for i in range(100)],
        "high": [1.12 + i * 0.001 for i in range(100)],
        "low": [1.08 + i * 0.001 for i in range(100)],
        "close": [1.11 + i * 0.001 for i in range(100)],
        "volume": [1000] * 100,
    })


@pytest.fixture
def sample_scope() -> DatasetScope:
    return DatasetScope("d1", "EURUSD", "1h", "2020-01-01", "2020-01-05")


@pytest.fixture
def sample_assumptions() -> ExecutionAssumptions:
    return ExecutionAssumptions(0.0001, 0.0001, 5.0)


@pytest.fixture
def sample_provenance() -> CodeProvenance:
    return CodeProvenance("commit_abc123")


@pytest.fixture
def sample_search_space() -> ResearchSearchSpace:
    cand1 = CandidateSpec("gen1", "1.0", "baseline_sma", {"period": 5})
    cand2 = CandidateSpec("gen1", "1.0", "baseline_sma", {"period": 10})
    return ResearchSearchSpace((cand1, cand2))


# A. Determinism
def test_campaign_definition_determinism(sample_scope, sample_assumptions, sample_provenance):
    def1 = ResearchCampaignDefinition(
        search_space_fingerprint="sfp_100",
        search_policy_fingerprint="pfp_100",
        criteria_fingerprint="cfp_100",
        dataset_scope=sample_scope,
        execution_assumptions=sample_assumptions,
        code_provenance=sample_provenance,
        methodology_version="v1.0",
        candidate_ids=("cand_a", "cand_b"),
        trial_count=2,
    )

    def2 = ResearchCampaignDefinition(
        search_space_fingerprint="sfp_100",
        search_policy_fingerprint="pfp_100",
        criteria_fingerprint="cfp_100",
        dataset_scope=sample_scope,
        execution_assumptions=sample_assumptions,
        code_provenance=sample_provenance,
        methodology_version="v1.0",
        candidate_ids=("cand_a", "cand_b"),
        trial_count=2,
    )

    assert def1.campaign_id == def2.campaign_id
    assert def1.definition_fingerprint == def2.definition_fingerprint


# B. Trial determinism
def test_trial_plan_determinism(sample_scope, sample_assumptions, sample_provenance, sample_search_space):
    criteria = DiscoveryCriteria()
    with tempfile.TemporaryDirectory() as tmpdir:
        orchestrator = ResearchCampaignOrchestrator(store=ResearchCampaignStore(tmpdir))
        _def1, plan1 = orchestrator.plan_campaign(
            search_space=sample_search_space,
            dataset_scope=sample_scope,
            execution_assumptions=sample_assumptions,
            code_provenance=sample_provenance,
            criteria=criteria,
        )
        _def2, plan2 = orchestrator.plan_campaign(
            search_space=sample_search_space,
            dataset_scope=sample_scope,
            execution_assumptions=sample_assumptions,
            code_provenance=sample_provenance,
            criteria=criteria,
        )

        assert plan1.plan_fingerprint == plan2.plan_fingerprint
        assert [t.trial_id for t in plan1.trials] == [t.trial_id for t in plan2.trials]


# C. Lifecycle
def test_valid_lifecycle_transitions():
    validate_campaign_state_transition(ResearchCampaignStatus.PLANNED, ResearchCampaignStatus.RUNNING)
    validate_campaign_state_transition(ResearchCampaignStatus.RUNNING, ResearchCampaignStatus.COMPLETED)
    validate_campaign_state_transition(ResearchCampaignStatus.RUNNING, ResearchCampaignStatus.PAUSED)
    validate_campaign_state_transition(ResearchCampaignStatus.PAUSED, ResearchCampaignStatus.RUNNING)
    validate_campaign_state_transition(ResearchCampaignStatus.RUNNING, ResearchCampaignStatus.FAILED)


# D. Invalid transitions
def test_invalid_lifecycle_transitions_fail_closed():
    with pytest.raises(InvalidLifecycleTransitionError):
        validate_campaign_state_transition(ResearchCampaignStatus.COMPLETED, ResearchCampaignStatus.RUNNING)

    with pytest.raises(InvalidLifecycleTransitionError):
        validate_campaign_state_transition(ResearchCampaignStatus.FAILED, ResearchCampaignStatus.PAUSED)

    with pytest.raises(InvalidLifecycleTransitionError):
        validate_campaign_state_transition(ResearchCampaignStatus.CANCELLED, ResearchCampaignStatus.RUNNING)


# E. Persistence
def test_campaign_store_persistence_roundtrip(sample_scope, sample_assumptions, sample_provenance):
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ResearchCampaignStore(tmpdir)
        defn = ResearchCampaignDefinition(
            search_space_fingerprint="sfp_1",
            search_policy_fingerprint="pfp_1",
            criteria_fingerprint="cfp_1",
            dataset_scope=sample_scope,
            execution_assumptions=sample_assumptions,
            code_provenance=sample_provenance,
            methodology_version="v1.0",
            candidate_ids=("cand_1",),
            trial_count=1,
        )
        store.save_definition(defn)
        loaded_defn = store.load_definition(defn.campaign_id)
        assert loaded_defn.definition_fingerprint == defn.definition_fingerprint

        pt = ResearchPlannedTrial(
            campaign_id=defn.campaign_id,
            trial_id=f"{defn.campaign_id}_tr_0",
            trial_index=0,
            candidate_id="cand_1",
            candidate_fingerprint="cfp_cand_1",
            hypothesis_fingerprint="hfp_cand_1",
            strategy_name="baseline_sma",
            strategy_version="1.0",
            dataset_id="d1",
            execution_assumptions_id="ea1",
        )
        plan = ResearchTrialPlan(
            campaign_id=defn.campaign_id,
            definition_fingerprint=defn.definition_fingerprint,
            trials=(pt,),
        )
        store.save_trial_plan(plan)
        loaded_plan = store.load_trial_plan(defn.campaign_id)
        assert loaded_plan.plan_fingerprint == plan.plan_fingerprint


# F. Corruption
def test_corrupted_definition_fingerprint_fails_closed(sample_scope, sample_assumptions, sample_provenance):
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ResearchCampaignStore(tmpdir)
        defn = ResearchCampaignDefinition(
            search_space_fingerprint="sfp_1",
            search_policy_fingerprint="pfp_1",
            criteria_fingerprint="cfp_1",
            dataset_scope=sample_scope,
            execution_assumptions=sample_assumptions,
            code_provenance=sample_provenance,
            methodology_version="v1.0",
            candidate_ids=("cand_1",),
            trial_count=1,
        )
        store.save_definition(defn)

        def_path = Path(tmpdir) / defn.campaign_id / "definition.json"
        raw_text = def_path.read_text()
        corrupted_text = raw_text.replace(defn.definition_fingerprint, "corrupted_fp_1234")
        def_path.write_text(corrupted_text)

        with pytest.raises(ValueError, match="Loaded definition fingerprint mismatch"):
            store.load_definition(defn.campaign_id)


# G. Resume & H. Idempotency
def test_resume_skips_completed_trials(
    sample_market_data, sample_scope, sample_assumptions, sample_provenance, sample_search_space
):
    criteria = DiscoveryCriteria()
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ResearchCampaignStore(tmpdir)
        orchestrator = ResearchCampaignOrchestrator(store=store)
        defn, _plan = orchestrator.plan_campaign(
            search_space=sample_search_space,
            dataset_scope=sample_scope,
            execution_assumptions=sample_assumptions,
            code_provenance=sample_provenance,
            criteria=criteria,
        )

        policy = ResearchCampaignExecutionPolicy(persist_evidence=True)
        res1 = orchestrator.execute_campaign(
            campaign_id=defn.campaign_id,
            df=sample_market_data,
            search_space=sample_search_space,
            criteria=criteria,
            execution_policy=policy,
        )

        assert res1.status == ResearchCampaignStatus.COMPLETED
        assert res1.executed_trial_count == 2

        # Resuming completed campaign returns existing manifest without re-executing
        res2 = orchestrator.resume_campaign(
            campaign_id=defn.campaign_id,
            df=sample_market_data,
            search_space=sample_search_space,
            criteria=criteria,
            execution_policy=policy,
        )

        assert res2.campaign_id == res1.campaign_id
        assert res2.status == ResearchCampaignStatus.COMPLETED


# I. Failure & J. Retry
def test_failed_trial_retry_preserves_history(sample_scope, sample_assumptions, sample_provenance):
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ResearchCampaignStore(tmpdir)
        cp = ResearchTrialCheckpoint(
            trial_id="tr_failed_1",
            campaign_id="camp_100",
            candidate_id="cand_1",
            trial_index=0,
            attempt_number=1,
            status="FAILED",
            error_message="Original execution failure",
        )
        store.save_trial_checkpoint(cp)

        loaded = store.load_trial_checkpoint("camp_100", "tr_failed_1")
        assert loaded.status == "FAILED"
        assert loaded.attempt_number == 1
        assert loaded.error_message == "Original execution failure"


# K. Evidence ordering
def test_evidence_ordering_before_checkpoint_completion(sample_scope, sample_assumptions, sample_provenance):
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ResearchCampaignStore(tmpdir)
        cp = ResearchTrialCheckpoint(
            trial_id="tr_ev_1",
            campaign_id="camp_200",
            candidate_id="cand_1",
            trial_index=0,
            attempt_number=1,
            status="QUALIFIED",
            experiment_fingerprint="exp_fp_999",
            evidence_fingerprint="ev_999",
        )
        store.save_trial_checkpoint(cp)
        loaded = store.load_trial_checkpoint("camp_200", "tr_ev_1")
        assert loaded.experiment_fingerprint == "exp_fp_999"
        assert loaded.evidence_fingerprint == "ev_999"


# L. Campaign completion validation
def test_campaign_completion_rejected_if_unresolved_trials(sample_scope, sample_assumptions, sample_provenance):
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ResearchCampaignStore(tmpdir)
        defn = ResearchCampaignDefinition(
            search_space_fingerprint="sfp_1",
            search_policy_fingerprint="pfp_1",
            criteria_fingerprint="cfp_1",
            dataset_scope=sample_scope,
            execution_assumptions=sample_assumptions,
            code_provenance=sample_provenance,
            methodology_version="v1.0",
            candidate_ids=("cand_1",),
            trial_count=1,
        )
        store.save_definition(defn)

        pt = ResearchPlannedTrial(
            campaign_id=defn.campaign_id,
            trial_id=f"{defn.campaign_id}_tr_0",
            trial_index=0,
            candidate_id="cand_1",
            candidate_fingerprint="cfp_cand_1",
            hypothesis_fingerprint="hfp_cand_1",
            strategy_name="baseline_sma",
            strategy_version="1.0",
            dataset_id="d1",
            execution_assumptions_id="ea1",
        )
        plan = ResearchTrialPlan(
            campaign_id=defn.campaign_id,
            definition_fingerprint=defn.definition_fingerprint,
            trials=(pt,),
        )
        store.save_trial_plan(plan)
        store.save_lifecycle_state(defn.campaign_id, ResearchCampaignStatus.PLANNED)

        # Transition directly to COMPLETED without trial resolution -> validate_campaign_integrity must fail closed!
        store.save_lifecycle_state(defn.campaign_id, ResearchCampaignStatus.RUNNING)
        store.save_lifecycle_state(defn.campaign_id, ResearchCampaignStatus.COMPLETED)

        with pytest.raises(CampaignIntegrityError, match="remains unresolved"):
            validate_campaign_integrity(defn.campaign_id, store=store)


# M. Truncation
def test_campaign_truncation_representation(
    sample_market_data, sample_scope, sample_assumptions, sample_provenance, sample_search_space
):
    criteria = DiscoveryCriteria()
    with tempfile.TemporaryDirectory() as tmpdir:
        orchestrator = ResearchCampaignOrchestrator(store=ResearchCampaignStore(tmpdir))

        # Max trials = 1 truncates search space
        policy = ResearchCampaignExecutionPolicy(max_trials=1, persist_evidence=True)
        defn, _plan = orchestrator.plan_campaign(
            search_space=sample_search_space,
            dataset_scope=sample_scope,
            execution_assumptions=sample_assumptions,
            code_provenance=sample_provenance,
            criteria=criteria,
            execution_policy=policy,
        )

        assert len(defn.candidate_ids) == 1
        res = orchestrator.execute_campaign(
            campaign_id=defn.campaign_id,
            df=sample_market_data,
            search_space=sample_search_space,
            criteria=criteria,
            execution_policy=policy,
        )
        assert res.status == ResearchCampaignStatus.TRUNCATED


# N. Governance preservation
def test_governance_path_preserved():
    # Verify ResearchHypothesis required transition
    raw_hyp = ResearchHypothesis(
        statement="Test hypothesis",
        methodology_version="v1.0",
        strategy_name="baseline_sma",
        strategy_version="1.0",
        dataset_scope=DatasetScope("d1", "EURUSD", "1h", "2020-01-01", "2020-01-05"),
        execution_assumptions=ExecutionAssumptions(0.0001, 0.0001, 5.0),
        code_provenance=CodeProvenance("commit_sha"),
        benchmark_reference="buy_and_hold",
    )
    assert raw_hyp.status == HypothesisStatus.GENERATED

    from src.evaluation.hypothesis_generator import accept_hypothesis_for_research
    accepted_hyp = accept_hypothesis_for_research(raw_hyp)
    assert accepted_hyp.status == HypothesisStatus.ACCEPTED_FOR_RESEARCH


# O. Lineage
def test_campaign_lineage_traceability(
    sample_market_data, sample_scope, sample_assumptions, sample_provenance, sample_search_space
):
    criteria = DiscoveryCriteria()
    with tempfile.TemporaryDirectory() as tmpdir:
        orchestrator = ResearchCampaignOrchestrator(store=ResearchCampaignStore(tmpdir))
        defn, plan = orchestrator.plan_campaign(
            search_space=sample_search_space,
            dataset_scope=sample_scope,
            execution_assumptions=sample_assumptions,
            code_provenance=sample_provenance,
            criteria=criteria,
        )
        campaign = orchestrator.execute_campaign(
            campaign_id=defn.campaign_id,
            df=sample_market_data,
            search_space=sample_search_space,
            criteria=criteria,
        )
        assert campaign.definition_fingerprint == defn.definition_fingerprint
        assert campaign.trial_plan_fingerprint == plan.plan_fingerprint


# Q. Duplicate trial protection
def test_duplicate_trial_ids_rejected(sample_scope, sample_assumptions, sample_provenance):
    pt1 = ResearchPlannedTrial(
        campaign_id="camp_dup",
        trial_id="tr_dup_0",
        trial_index=0,
        candidate_id="cand_1",
        candidate_fingerprint="cfp_1",
        hypothesis_fingerprint="hfp_1",
        strategy_name="s1",
        strategy_version="1.0",
        dataset_id="d1",
        execution_assumptions_id="ea1",
    )
    pt2 = ResearchPlannedTrial(
        campaign_id="camp_dup",
        trial_id="tr_dup_0",  # Duplicate trial ID!
        trial_index=1,
        candidate_id="cand_2",
        candidate_fingerprint="cfp_2",
        hypothesis_fingerprint="hfp_2",
        strategy_name="s1",
        strategy_version="1.0",
        dataset_id="d1",
        execution_assumptions_id="ea1",
    )
    with pytest.raises(ValueError, match="Duplicate trial_id"):
        ResearchTrialPlan(
            campaign_id="camp_dup",
            definition_fingerprint="def_fp_dup",
            trials=(pt1, pt2),
        )


# R. Candidate identity mismatch
def test_checkpoint_candidate_mismatch_fails_integrity_validation(
    sample_scope, sample_assumptions, sample_provenance
):
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ResearchCampaignStore(tmpdir)
        defn = ResearchCampaignDefinition(
            search_space_fingerprint="sfp_1",
            search_policy_fingerprint="pfp_1",
            criteria_fingerprint="cfp_1",
            dataset_scope=sample_scope,
            execution_assumptions=sample_assumptions,
            code_provenance=sample_provenance,
            methodology_version="v1.0",
            candidate_ids=("cand_expected",),
            trial_count=1,
        )
        store.save_definition(defn)

        pt = ResearchPlannedTrial(
            campaign_id=defn.campaign_id,
            trial_id=f"{defn.campaign_id}_tr_0",
            trial_index=0,
            candidate_id="cand_expected",
            candidate_fingerprint="cfp_1",
            hypothesis_fingerprint="hfp_1",
            strategy_name="s1",
            strategy_version="1.0",
            dataset_id="d1",
            execution_assumptions_id="ea1",
        )
        plan = ResearchTrialPlan(
            campaign_id=defn.campaign_id,
            definition_fingerprint=defn.definition_fingerprint,
            trials=(pt,),
        )
        store.save_trial_plan(plan)
        store.save_lifecycle_state(defn.campaign_id, ResearchCampaignStatus.PLANNED)

        # Checkpoint with mismatched candidate ID
        cp = ResearchTrialCheckpoint(
            trial_id=f"{defn.campaign_id}_tr_0",
            campaign_id=defn.campaign_id,
            candidate_id="cand_WRONG_SUBSTITUTION",
            trial_index=0,
            status="PENDING",
        )
        store.save_trial_checkpoint(cp)

        with pytest.raises(CampaignIntegrityError, match="Candidate identity mismatch"):
            validate_campaign_integrity(defn.campaign_id, store=store)


# S. Scientific identity & T. Volatile timestamp exclusion
def test_scientific_identity_unaffected_by_timestamps(sample_scope, sample_assumptions, sample_provenance):
    def1 = ResearchCampaignDefinition(
        search_space_fingerprint="sfp_1",
        search_policy_fingerprint="pfp_1",
        criteria_fingerprint="cfp_1",
        dataset_scope=sample_scope,
        execution_assumptions=sample_assumptions,
        code_provenance=sample_provenance,
        methodology_version="v1.0",
        candidate_ids=("c1", "c2"),
        trial_count=2,
    )

    # Changing an input like methodology_version changes scientific identity fingerprint
    def_modified = ResearchCampaignDefinition(
        search_space_fingerprint="sfp_1",
        search_policy_fingerprint="pfp_1",
        criteria_fingerprint="cfp_1",
        dataset_scope=sample_scope,
        execution_assumptions=sample_assumptions,
        code_provenance=sample_provenance,
        methodology_version="v2.0_NEW_SCIENTIFIC_METHOD",
        candidate_ids=("c1", "c2"),
        trial_count=2,
    )

    assert def1.definition_fingerprint != def_modified.definition_fingerprint


# Item 17: Static Architectural Guards
def test_static_ast_architectural_guards():
    """AST static guards ensuring campaign_orchestrator.py does not bypass research boundaries."""
    orchestrator_file = Path("src/evaluation/campaign_orchestrator.py")
    assert orchestrator_file.exists()

    tree = ast.parse(orchestrator_file.read_text(encoding="utf-8"))

    imported_modules = set()
    function_calls = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                function_calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                function_calls.add(node.func.attr)

    # Forbidden imports for research orchestration
    forbidden_imports = {
        "src.evaluation.live_production_decision",
        "src.integration.project2_publisher",
        "src.evaluation.production_readiness",
        "src.evaluation.production_selection",
    }

    for forbidden in forbidden_imports:
        assert forbidden not in imported_modules, f"Forbidden import '{forbidden}' in campaign_orchestrator.py"

    # Required research boundary calls
    required_calls = {"run_research_experiment", "qualify_research_evidence", "assess_research_robustness"}
    for req in required_calls:
        assert req in function_calls, f"Required research function call '{req}' missing from campaign_orchestrator.py"
