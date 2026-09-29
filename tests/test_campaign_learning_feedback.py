"""Tests for Governed Campaign Learning & Feedback Loop Lifecycle.

Verifies end-to-end feedback loop connecting:
Completed Research Campaign -> Campaign Evidence Synthesis -> Campaign Selection Decision
-> Governed Campaign Learning Artifact -> Research Knowledge Patterns -> Governed Research Hypothesis (GENERATED)
-> Explicit ACCEPTED_FOR_RESEARCH -> run_research_experiment().

Covers positive path, complete fail-closed negative matrix, SHA-256 determinism,
and strict governance boundaries preventing production authority leakage.
"""

import tempfile
from pathlib import Path
import pytest
import pandas as pd

from src.evaluation.campaign_learning import (
    CampaignLearningIntegrityError,
    GovernedCampaignLearningArtifact,
    derive_governed_campaign_learning,
    generate_hypotheses_from_campaign_learning,
    materialize_governed_campaign_feedback,
    register_governed_campaign_learning,
    validate_feedback_loop_lineage,
)
from src.evaluation.campaign_synthesis import (
    ResearchCampaignSelectionDecision,
)
from src.evaluation.campaign_orchestrator import (
    ResearchCampaignExecutionPolicy,
    ResearchCampaignOrchestrator,
)
from src.evaluation.campaign_synthesis import (
    ResearchCampaignSelectionPolicy,
    select_campaign_candidate,
    synthesize_campaign_evidence,
)
from src.evaluation.candidate_generator import (
    CandidateGenerator,
    CandidateGeneratorSpec,
    ResearchSearchSpace,
)
from src.evaluation.hypothesis_generator import (
    HypothesisGenerationContext,
    HypothesisGenerationError,
    UnacceptedHypothesisError,
    accept_hypothesis_for_research,
)
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
    HypothesisStatus,
    ResearchCampaignStatus,
)
from src.evaluation.research_runner import DiscoveryCriteria, run_research_experiment
from src.evaluation.research_store import ResearchCampaignStore
from src.evaluation.research_registry import ResearchRegistryStore
from src.strategies.registry import DEFAULT_REGISTRY


@pytest.fixture
def tmp_stores():
    with tempfile.TemporaryDirectory() as tmp_dir:
        c_dir = Path(tmp_dir) / "campaigns"
        r_dir = Path(tmp_dir) / "registry"
        e_dir = Path(tmp_dir) / "experiments"
        c_dir.mkdir()
        r_dir.mkdir()
        e_dir.mkdir()

        c_store = ResearchCampaignStore(base_dir=c_dir)
        r_store = ResearchRegistryStore(base_dir=r_dir)
        yield c_store, r_store, e_dir


@pytest.fixture
def sample_dataset():
    dates = pd.date_range("2023-01-01", periods=100, freq="1h")
    df = pd.DataFrame(
        {
            "open": [100.0 + i * 0.1 for i in range(100)],
            "high": [101.0 + i * 0.1 for i in range(100)],
            "low": [99.0 + i * 0.1 for i in range(100)],
            "close": [100.5 + i * 0.1 for i in range(100)],
            "volume": [1000.0] * 100,
        },
        index=dates,
    )
    return df


@pytest.fixture
def base_context():
    return HypothesisGenerationContext(
        dataset_scope=DatasetScope("ds_test", "XAUUSD", "1h", "2023-01-01", "2023-01-05"),
        execution_assumptions=ExecutionAssumptions(0.0001, 0.0001, 10.0),
        code_provenance=CodeProvenance("sha1234567890", "clean", "author"),
    )


def test_positive_campaign_learning_feedback_loop(tmp_stores, sample_dataset, base_context):
    c_store, r_store, e_dir = tmp_stores

    # 1. Setup and execute a campaign
    spec = CandidateGeneratorSpec(
        generator_name="test_gen",
        generator_version="1.0",
        strategy_name="baseline",
        parameter_grid={"fast_window": [5], "slow_window": [20]},
    )
    gen = CandidateGenerator(spec)
    candidates = gen.generate_candidates()
    search_space = ResearchSearchSpace(candidate_definitions=candidates)

    criteria = DiscoveryCriteria(min_is_sharpe=-10.0, min_oos_sharpe=-10.0)
    orchestrator = ResearchCampaignOrchestrator(
        store=c_store, registry=DEFAULT_REGISTRY, memory_store=r_store
    )

    defn, plan = orchestrator.plan_campaign(
        search_space=search_space,
        dataset_scope=base_context.dataset_scope,
        execution_assumptions=base_context.execution_assumptions,
        code_provenance=base_context.code_provenance,
        criteria=criteria,
    )

    campaign = orchestrator.execute_campaign(
        campaign_id=defn.campaign_id,
        df=sample_dataset,
        search_space=search_space,
        criteria=criteria,
        execution_policy=ResearchCampaignExecutionPolicy(max_trials=10),
        persist_registry_dir=r_store.base_dir,
    )

    assert campaign.status == ResearchCampaignStatus.COMPLETED

    # 2. Synthesize evidence and make selection decision
    synthesis = synthesize_campaign_evidence(defn.campaign_id, store=c_store)
    c_store.save_evidence_synthesis(synthesis)

    sel_policy = ResearchCampaignSelectionPolicy(
        required_governance_states=("QUALIFIED", "PROMOTABLE", "VALIDATED", "REJECTED"),
        required_oos_evidence=False,
        required_walk_forward_evidence=False,
    )
    decision = select_campaign_candidate(synthesis, selection_policy=sel_policy)
    if decision.decision_status != "SELECTED":
        comps_info = [(c.candidate_id, c.qualification_status, c.robustness_status, c.rejection_reasons) for c in synthesis.candidate_comparisons]
        raise ValueError(f"Selection failed status={decision.decision_status}, reason={decision.decision_reason}, comps={comps_info}")
    c_store.save_selection_decision(decision)
    assert decision.decision_status == "SELECTED"

    # 3. Derive and register campaign learning artifact
    learning_art = derive_governed_campaign_learning(
        campaign_id=defn.campaign_id,
        store=c_store,
        registry_store=r_store,
    )

    registered_art = register_governed_campaign_learning(
        learning_art,
        store=c_store,
        registry_store=r_store,
    )

    assert registered_art.artifact_fingerprint == learning_art.artifact_fingerprint

    # 4. Generate future hypotheses from campaign learning memory
    hypotheses = generate_hypotheses_from_campaign_learning(
        campaign_id=defn.campaign_id,
        context=base_context,
        store=c_store,
        registry_store=r_store,
    )

    assert len(hypotheses) > 0
    hyp = hypotheses[0]
    assert hyp.status == HypothesisStatus.GENERATED
    assert hyp.constraints["campaign_id"] == defn.campaign_id

    # 5. Validate unbroken feedback loop lineage
    validate_feedback_loop_lineage(defn.campaign_id, hyp, store=c_store, registry_store=r_store)

    # 6. Explicitly accept hypothesis for research execution
    accepted_hyp = accept_hypothesis_for_research(hyp)
    assert accepted_hyp.status == HypothesisStatus.ACCEPTED_FOR_RESEARCH

    # 7. Execute research experiment via canonical runner
    evidence = run_research_experiment(
        spec=accepted_hyp,
        df=sample_dataset,
        criteria=criteria,
        registry=DEFAULT_REGISTRY,
    )

    assert evidence.spec.hypothesis == accepted_hyp.statement
    assert evidence.spec.strategy_name == accepted_hyp.strategy_name
    assert evidence.evidence_id is not None


def test_fail_closed_uncompleted_campaign(tmp_stores, base_context):
    c_store, r_store, _ = tmp_stores
    cid = "uncompleted_campaign_001"

    c_store.save_lifecycle_state(cid, ResearchCampaignStatus.RUNNING)

    with pytest.raises(CampaignLearningIntegrityError, match="is in status 'RUNNING', expected COMPLETED/TRUNCATED"):
        derive_governed_campaign_learning(cid, store=c_store, registry_store=r_store)


def test_fail_closed_missing_selection_decision(tmp_stores, sample_dataset, base_context):
    c_store, r_store, _ = tmp_stores

    spec = CandidateGeneratorSpec(
        generator_name="test_gen",
        generator_version="1.0",
        strategy_name="baseline",
        parameter_grid={"fast_window": [5]},
    )
    gen = CandidateGenerator(spec)
    candidates = gen.generate_candidates()
    search_space = ResearchSearchSpace(candidate_definitions=candidates)

    criteria = DiscoveryCriteria(min_is_sharpe=-10.0, min_oos_sharpe=-10.0)
    orchestrator = ResearchCampaignOrchestrator(store=c_store)
    defn, _ = orchestrator.plan_campaign(
        search_space=search_space,
        dataset_scope=base_context.dataset_scope,
        execution_assumptions=base_context.execution_assumptions,
        code_provenance=base_context.code_provenance,
        criteria=criteria,
    )
    orchestrator.execute_campaign(defn.campaign_id, sample_dataset, search_space, criteria)

    # Synthesis saved, but selection decision NOT saved
    synthesis = synthesize_campaign_evidence(defn.campaign_id, store=c_store)
    c_store.save_evidence_synthesis(synthesis)

    with pytest.raises(FileNotFoundError, match="Selection decision not found"):
        derive_governed_campaign_learning(defn.campaign_id, store=c_store, registry_store=r_store)


def test_fail_closed_decision_from_another_campaign(tmp_stores, base_context):
    c_store, r_store, _ = tmp_stores
    c1 = "campaign_alpha"
    c2 = "campaign_beta"

    art = GovernedCampaignLearningArtifact(
        campaign_id=c1,
        campaign_selection_decision_fingerprint="dec_fp_123",
        synthesis_fingerprint="syn_fp_123",
        selected_candidate_ids=("cand1",),
        supporting_evidence_fingerprints=("ev1",),
        robustness_fingerprints=("rob1",),
        qualification_fingerprints=("qual1",),
        learning_record_ids=("lr1",),
        knowledge_pattern_ids=("kp1",),
    )

    # Art is for c1, but attempting to validate for c2
    c_store.save_lifecycle_state(c2, ResearchCampaignStatus.COMPLETED)

    with pytest.raises(CampaignLearningIntegrityError):
        register_governed_campaign_learning(art, store=c_store, registry_store=r_store)


def test_unaccepted_hypothesis_cannot_execute(base_context, sample_dataset):
    # Construct hypothesis in GENERATED status
    from src.evaluation.hypothesis_generator import KnowledgeHypothesisGenerator
    from src.evaluation.research_knowledge import PatternObservationSummary, ResearchKnowledgePattern, ResearchPatternCategory

    pat = ResearchKnowledgePattern(
        pattern_id="pat_001",
        category=ResearchPatternCategory.SUCCESS_PATTERN,
        statement="Test success pattern",
        normalized_conditions=PatternObservationSummary(
            symbol="XAUUSD", timeframe="1h", strategy_name="baseline", strategy_version="1.0.0",
            dataset_scope_id="ds1", execution_assumptions_id="ea1", code_provenance_id="cp1", methodology_version="1.0"
        ),
        supporting_learning_ids=("lr1",),
        supporting_experiment_fingerprints=("exp1",),
        supporting_evidence_fingerprints=("ev1",),
        observation_count=2, success_count=2, failure_count=0, inconclusive_count=0, is_contradictory=False
    )

    gen = KnowledgeHypothesisGenerator()
    hyps = gen.generate([pat], context=base_context)
    hyp = hyps[0]

    assert hyp.status == HypothesisStatus.GENERATED

    with pytest.raises(UnacceptedHypothesisError, match="is not accepted for research execution"):
        run_research_experiment(hyp, df=sample_dataset)


def test_sha256_determinism_and_timestamp_independence(tmp_stores, base_context):
    c_store, r_store, _ = tmp_stores

    art1 = GovernedCampaignLearningArtifact(
        campaign_id="camp_100",
        campaign_selection_decision_fingerprint="dec_100",
        synthesis_fingerprint="syn_100",
        selected_candidate_ids=("cand_a", "cand_b"),
        supporting_evidence_fingerprints=("ev_1", "ev_2"),
        robustness_fingerprints=("rob_1",),
        qualification_fingerprints=("qual_1",),
        learning_record_ids=("lr_1",),
        knowledge_pattern_ids=("kp_1",),
    )

    art2 = GovernedCampaignLearningArtifact(
        campaign_id="camp_100",
        campaign_selection_decision_fingerprint="dec_100",
        synthesis_fingerprint="syn_100",
        selected_candidate_ids=("cand_b", "cand_a"),  # Reversed list order
        supporting_evidence_fingerprints=("ev_2", "ev_1"),
        robustness_fingerprints=("rob_1",),
        qualification_fingerprints=("qual_1",),
        learning_record_ids=("lr_1",),
        knowledge_pattern_ids=("kp_1",),
    )

    assert art1.artifact_fingerprint == art2.artifact_fingerprint
    assert art1.learning_artifact_id == art2.learning_artifact_id


def test_governance_boundary_isolation(tmp_stores, base_context, sample_dataset):
    """Verify that feedback loop cannot produce live runtime authorization or production publication."""
    c_store, r_store, _ = tmp_stores

    # Verify no ProductionRuntimeAuthorization or live trade function is imported or called
    import src.evaluation.campaign_learning as cl_mod
    assert not hasattr(cl_mod, "ProductionRuntimeAuthorization")
    assert not hasattr(cl_mod, "ProductionDecision")
    assert not hasattr(cl_mod, "Project2Publisher")


# =============================================================================
# Exact Domain Invariant Unit Tests for Campaign Learning Materialization Boundary
# =============================================================================

def _save_mock_definition_and_plan(c_store, cid):
    from src.evaluation.research_constitution import (
        ResearchCampaignDefinition,
        ResearchTrialPlan,
        DatasetScope,
        ExecutionAssumptions,
        CodeProvenance,
    )
    defn = ResearchCampaignDefinition(
        search_space_fingerprint="ss_fp_001",
        search_policy_fingerprint="sp_fp_001",
        criteria_fingerprint="crit_fp_001",
        dataset_scope=DatasetScope("ds_1", "XAUUSD", "1h", "2023-01-01", "2023-01-05"),
        execution_assumptions=ExecutionAssumptions(0.0001, 0.0001, 10.0),
        code_provenance=CodeProvenance("sha1234567890", "clean", "author"),
        methodology_version="1.0",
        candidate_ids=("cand_1", "cand_2"),
        trial_count=2,
    )
    object.__setattr__(defn, "campaign_id", cid)
    # Write definition directly using as_dict to match expected campaign_id
    cdir = c_store._campaign_dir(cid)
    cdir.mkdir(parents=True, exist_ok=True)
    import json
    (cdir / "definition.json").write_text(json.dumps(defn.as_dict(), indent=2), encoding="utf-8")

    plan = ResearchTrialPlan(
        campaign_id=cid,
        definition_fingerprint=defn.definition_fingerprint,
        trials=(),
    )
    c_store.save_trial_plan(plan)
    return defn, plan


def test_learning_artifact_selected_candidates_must_exactly_match_decision(tmp_stores):
    c_store, r_store, _ = tmp_stores
    cid = "camp_exact_match_001"

    defn, plan = _save_mock_definition_and_plan(c_store, cid)
    c_store.save_lifecycle_state(cid, ResearchCampaignStatus.COMPLETED)

    from src.evaluation.campaign_synthesis import ResearchCampaignEvidenceSynthesis
    syn = ResearchCampaignEvidenceSynthesis(
        campaign_id=cid,
        campaign_definition_fingerprint=defn.definition_fingerprint,
        trial_plan_fingerprint=plan.plan_fingerprint,
        search_space_fingerprint="ss_fp_001",
        search_policy_fingerprint="sp_fp_001",
        criteria_fingerprint="crit_fp_001",
        dataset_identity={},
        execution_assumptions={},
        code_provenance={},
        methodology_version="1.0",
        ordered_trial_identities=(),
        ordered_evidence_fingerprints=("ev_1",),
        completed_trial_count=1,
        failed_trial_count=0,
        blocked_trial_count=0,
        qualified_candidate_count=1,
        rejected_candidate_count=0,
        selection_governance_status="SELECTION_NOT_APPLICABLE",
        selection_policy_fingerprint="pol_fp_001",
        candidate_comparisons=(),
    )
    c_store.save_evidence_synthesis(syn)

    dec = ResearchCampaignSelectionDecision(
        campaign_id=cid,
        synthesis_fingerprint=syn.synthesis_fingerprint,
        selection_policy_fingerprint="pol_fp_001",
        selected_candidate_ids=("cand_1", "cand_2"),
        eligible_candidate_ids=("cand_1", "cand_2"),
        rejected_candidate_ids=(),
        blocked_candidate_ids=(),
        candidate_comparison_fingerprints=("comp_1",),
        selection_governance_fingerprints=(),
        qualification_fingerprints=("qual_1",),
        robustness_fingerprints=("rob_1",),
        evidence_fingerprints=("ev_1",),
        decision_status="SELECTED",
        decision_reason="Top candidate",
        deterministic_ordering=("cand_1", "cand_2"),
    )
    c_store.save_selection_decision(dec)

    art_mismatched = GovernedCampaignLearningArtifact(
        campaign_id=cid,
        campaign_selection_decision_fingerprint=dec.decision_fingerprint,
        synthesis_fingerprint=syn.synthesis_fingerprint,
        selected_candidate_ids=("cand_1",),  # Mismatched candidate set
        supporting_evidence_fingerprints=("ev_1",),
        robustness_fingerprints=("rob_1",),
        qualification_fingerprints=("qual_1",),
        learning_record_ids=(),
        knowledge_pattern_ids=(),
    )

    with pytest.raises(CampaignLearningIntegrityError, match="selected_candidate_ids"):
        register_governed_campaign_learning(art_mismatched, store=c_store, registry_store=r_store)


def test_learning_artifact_evidence_lineage_must_exactly_match_decision(tmp_stores):
    c_store, r_store, _ = tmp_stores
    cid = "camp_ev_match_001"

    defn, plan = _save_mock_definition_and_plan(c_store, cid)
    c_store.save_lifecycle_state(cid, ResearchCampaignStatus.COMPLETED)

    from src.evaluation.campaign_synthesis import ResearchCampaignEvidenceSynthesis
    syn = ResearchCampaignEvidenceSynthesis(
        campaign_id=cid,
        campaign_definition_fingerprint=defn.definition_fingerprint,
        trial_plan_fingerprint=plan.plan_fingerprint,
        search_space_fingerprint="ss_fp_001",
        search_policy_fingerprint="sp_fp_001",
        criteria_fingerprint="crit_fp_001",
        dataset_identity={},
        execution_assumptions={},
        code_provenance={},
        methodology_version="1.0",
        ordered_trial_identities=(),
        ordered_evidence_fingerprints=("ev_1", "ev_2"),
        completed_trial_count=2,
        failed_trial_count=0,
        blocked_trial_count=0,
        qualified_candidate_count=1,
        rejected_candidate_count=0,
        selection_governance_status="SELECTION_NOT_APPLICABLE",
        selection_policy_fingerprint="pol_fp_001",
        candidate_comparisons=(),
    )
    c_store.save_evidence_synthesis(syn)

    dec = ResearchCampaignSelectionDecision(
        campaign_id=cid,
        synthesis_fingerprint=syn.synthesis_fingerprint,
        selection_policy_fingerprint="pol_fp_001",
        selected_candidate_ids=("cand_1",),
        eligible_candidate_ids=("cand_1",),
        rejected_candidate_ids=(),
        blocked_candidate_ids=(),
        candidate_comparison_fingerprints=("comp_1",),
        selection_governance_fingerprints=(),
        qualification_fingerprints=("qual_1",),
        robustness_fingerprints=("rob_1",),
        evidence_fingerprints=("ev_1", "ev_2"),
        decision_status="SELECTED",
        decision_reason="Top candidate",
        deterministic_ordering=("cand_1",),
    )
    c_store.save_selection_decision(dec)

    art_mismatched = GovernedCampaignLearningArtifact(
        campaign_id=cid,
        campaign_selection_decision_fingerprint=dec.decision_fingerprint,
        synthesis_fingerprint=syn.synthesis_fingerprint,
        selected_candidate_ids=("cand_1",),
        supporting_evidence_fingerprints=("ev_1",),  # Missing ev_2
        robustness_fingerprints=("rob_1",),
        qualification_fingerprints=("qual_1",),
        learning_record_ids=(),
        knowledge_pattern_ids=(),
    )

    with pytest.raises(CampaignLearningIntegrityError, match="supporting_evidence_fingerprints"):
        register_governed_campaign_learning(art_mismatched, store=c_store, registry_store=r_store)


def test_learning_artifact_robustness_lineage_must_exactly_match_decision(tmp_stores):
    c_store, r_store, _ = tmp_stores
    cid = "camp_rob_match_001"

    defn, plan = _save_mock_definition_and_plan(c_store, cid)
    c_store.save_lifecycle_state(cid, ResearchCampaignStatus.COMPLETED)

    from src.evaluation.campaign_synthesis import ResearchCampaignEvidenceSynthesis
    syn = ResearchCampaignEvidenceSynthesis(
        campaign_id=cid,
        campaign_definition_fingerprint=defn.definition_fingerprint,
        trial_plan_fingerprint=plan.plan_fingerprint,
        search_space_fingerprint="ss_fp_001",
        search_policy_fingerprint="sp_fp_001",
        criteria_fingerprint="crit_fp_001",
        dataset_identity={},
        execution_assumptions={},
        code_provenance={},
        methodology_version="1.0",
        ordered_trial_identities=(),
        ordered_evidence_fingerprints=("ev_1",),
        completed_trial_count=1,
        failed_trial_count=0,
        blocked_trial_count=0,
        qualified_candidate_count=1,
        rejected_candidate_count=0,
        selection_governance_status="SELECTION_NOT_APPLICABLE",
        selection_policy_fingerprint="pol_fp_001",
        candidate_comparisons=(),
    )
    c_store.save_evidence_synthesis(syn)

    dec = ResearchCampaignSelectionDecision(
        campaign_id=cid,
        synthesis_fingerprint=syn.synthesis_fingerprint,
        selection_policy_fingerprint="pol_fp_001",
        selected_candidate_ids=("cand_1",),
        eligible_candidate_ids=("cand_1",),
        rejected_candidate_ids=(),
        blocked_candidate_ids=(),
        candidate_comparison_fingerprints=("comp_1",),
        selection_governance_fingerprints=(),
        qualification_fingerprints=("qual_1",),
        robustness_fingerprints=("rob_1", "rob_2"),
        evidence_fingerprints=("ev_1",),
        decision_status="SELECTED",
        decision_reason="Top candidate",
        deterministic_ordering=("cand_1",),
    )
    c_store.save_selection_decision(dec)

    art_mismatched = GovernedCampaignLearningArtifact(
        campaign_id=cid,
        campaign_selection_decision_fingerprint=dec.decision_fingerprint,
        synthesis_fingerprint=syn.synthesis_fingerprint,
        selected_candidate_ids=("cand_1",),
        supporting_evidence_fingerprints=("ev_1",),
        robustness_fingerprints=("rob_1",),  # Missing rob_2
        qualification_fingerprints=("qual_1",),
        learning_record_ids=(),
        knowledge_pattern_ids=(),
    )

    with pytest.raises(CampaignLearningIntegrityError, match="robustness_fingerprints"):
        register_governed_campaign_learning(art_mismatched, store=c_store, registry_store=r_store)


def test_learning_artifact_qualification_lineage_must_exactly_match_decision(tmp_stores):
    c_store, r_store, _ = tmp_stores
    cid = "camp_qual_match_001"

    defn, plan = _save_mock_definition_and_plan(c_store, cid)
    c_store.save_lifecycle_state(cid, ResearchCampaignStatus.COMPLETED)

    from src.evaluation.campaign_synthesis import ResearchCampaignEvidenceSynthesis
    syn = ResearchCampaignEvidenceSynthesis(
        campaign_id=cid,
        campaign_definition_fingerprint=defn.definition_fingerprint,
        trial_plan_fingerprint=plan.plan_fingerprint,
        search_space_fingerprint="ss_fp_001",
        search_policy_fingerprint="sp_fp_001",
        criteria_fingerprint="crit_fp_001",
        dataset_identity={},
        execution_assumptions={},
        code_provenance={},
        methodology_version="1.0",
        ordered_trial_identities=(),
        ordered_evidence_fingerprints=("ev_1",),
        completed_trial_count=1,
        failed_trial_count=0,
        blocked_trial_count=0,
        qualified_candidate_count=1,
        rejected_candidate_count=0,
        selection_governance_status="SELECTION_NOT_APPLICABLE",
        selection_policy_fingerprint="pol_fp_001",
        candidate_comparisons=(),
    )
    c_store.save_evidence_synthesis(syn)

    dec = ResearchCampaignSelectionDecision(
        campaign_id=cid,
        synthesis_fingerprint=syn.synthesis_fingerprint,
        selection_policy_fingerprint="pol_fp_001",
        selected_candidate_ids=("cand_1",),
        eligible_candidate_ids=("cand_1",),
        rejected_candidate_ids=(),
        blocked_candidate_ids=(),
        candidate_comparison_fingerprints=("comp_1",),
        selection_governance_fingerprints=(),
        qualification_fingerprints=("qual_1", "qual_2"),
        robustness_fingerprints=("rob_1",),
        evidence_fingerprints=("ev_1",),
        decision_status="SELECTED",
        decision_reason="Top candidate",
        deterministic_ordering=("cand_1",),
    )
    c_store.save_selection_decision(dec)

    art_mismatched = GovernedCampaignLearningArtifact(
        campaign_id=cid,
        campaign_selection_decision_fingerprint=dec.decision_fingerprint,
        synthesis_fingerprint=syn.synthesis_fingerprint,
        selected_candidate_ids=("cand_1",),
        supporting_evidence_fingerprints=("ev_1",),
        robustness_fingerprints=("rob_1",),
        qualification_fingerprints=("qual_1",),  # Missing qual_2
        learning_record_ids=(),
        knowledge_pattern_ids=(),
    )

    with pytest.raises(CampaignLearningIntegrityError, match="qualification_fingerprints"):
        register_governed_campaign_learning(art_mismatched, store=c_store, registry_store=r_store)


def test_materialization_is_deterministic(tmp_stores, sample_dataset, base_context):
    c_store, r_store, _ = tmp_stores

    spec = CandidateGeneratorSpec(
        generator_name="test_gen",
        generator_version="1.0",
        strategy_name="baseline",
        parameter_grid={"fast_window": [5]},
    )
    gen = CandidateGenerator(spec)
    candidates = gen.generate_candidates()
    search_space = ResearchSearchSpace(candidate_definitions=candidates)

    criteria = DiscoveryCriteria(min_is_sharpe=-10.0, min_oos_sharpe=-10.0)
    orchestrator = ResearchCampaignOrchestrator(
        store=c_store, registry=DEFAULT_REGISTRY, memory_store=r_store
    )

    defn, _ = orchestrator.plan_campaign(
        search_space=search_space,
        dataset_scope=base_context.dataset_scope,
        execution_assumptions=base_context.execution_assumptions,
        code_provenance=base_context.code_provenance,
        criteria=criteria,
    )

    orchestrator.execute_campaign(
        campaign_id=defn.campaign_id,
        df=sample_dataset,
        search_space=search_space,
        criteria=criteria,
        execution_policy=ResearchCampaignExecutionPolicy(max_trials=10),
        persist_registry_dir=r_store.base_dir,
    )

    synthesis = synthesize_campaign_evidence(defn.campaign_id, store=c_store)
    c_store.save_evidence_synthesis(synthesis)

    sel_policy = ResearchCampaignSelectionPolicy(
        required_governance_states=("QUALIFIED", "PROMOTABLE", "VALIDATED", "REJECTED"),
        required_oos_evidence=False,
        required_walk_forward_evidence=False,
    )
    decision = select_campaign_candidate(synthesis, selection_policy=sel_policy)
    c_store.save_selection_decision(decision)

    art1 = derive_governed_campaign_learning(defn.campaign_id, store=c_store, registry_store=r_store)
    art2 = derive_governed_campaign_learning(defn.campaign_id, store=c_store, registry_store=r_store)

    assert art1.artifact_fingerprint == art2.artifact_fingerprint
    assert art1.learning_artifact_id == art2.learning_artifact_id


def test_materialization_is_idempotent(tmp_stores, sample_dataset, base_context):
    c_store, r_store, _ = tmp_stores

    spec = CandidateGeneratorSpec(
        generator_name="test_gen",
        generator_version="1.0",
        strategy_name="baseline",
        parameter_grid={"fast_window": [5]},
    )
    gen = CandidateGenerator(spec)
    candidates = gen.generate_candidates()
    search_space = ResearchSearchSpace(candidate_definitions=candidates)

    criteria = DiscoveryCriteria(min_is_sharpe=-10.0, min_oos_sharpe=-10.0)
    orchestrator = ResearchCampaignOrchestrator(
        store=c_store, registry=DEFAULT_REGISTRY, memory_store=r_store
    )

    defn, _ = orchestrator.plan_campaign(
        search_space=search_space,
        dataset_scope=base_context.dataset_scope,
        execution_assumptions=base_context.execution_assumptions,
        code_provenance=base_context.code_provenance,
        criteria=criteria,
    )

    orchestrator.execute_campaign(
        campaign_id=defn.campaign_id,
        df=sample_dataset,
        search_space=search_space,
        criteria=criteria,
        execution_policy=ResearchCampaignExecutionPolicy(max_trials=10),
        persist_registry_dir=r_store.base_dir,
    )

    synthesis = synthesize_campaign_evidence(defn.campaign_id, store=c_store)
    c_store.save_evidence_synthesis(synthesis)

    sel_policy = ResearchCampaignSelectionPolicy(
        required_governance_states=("QUALIFIED", "PROMOTABLE", "VALIDATED", "REJECTED"),
        required_oos_evidence=False,
        required_walk_forward_evidence=False,
    )
    decision = select_campaign_candidate(synthesis, selection_policy=sel_policy)
    c_store.save_selection_decision(decision)

    art1 = materialize_governed_campaign_feedback(defn.campaign_id, store=c_store, registry_store=r_store)
    art2 = materialize_governed_campaign_feedback(defn.campaign_id, store=c_store, registry_store=r_store)

    assert art1.artifact_fingerprint == art2.artifact_fingerprint


def test_generated_hypothesis_preserves_canonical_learning_lineage(tmp_stores, sample_dataset, base_context):
    c_store, r_store, _ = tmp_stores

    spec = CandidateGeneratorSpec(
        generator_name="test_gen",
        generator_version="1.0",
        strategy_name="baseline",
        parameter_grid={"fast_window": [5]},
    )
    gen = CandidateGenerator(spec)
    candidates = gen.generate_candidates()
    search_space = ResearchSearchSpace(candidate_definitions=candidates)

    criteria = DiscoveryCriteria(min_is_sharpe=-10.0, min_oos_sharpe=-10.0)
    orchestrator = ResearchCampaignOrchestrator(
        store=c_store, registry=DEFAULT_REGISTRY, memory_store=r_store
    )

    defn, _ = orchestrator.plan_campaign(
        search_space=search_space,
        dataset_scope=base_context.dataset_scope,
        execution_assumptions=base_context.execution_assumptions,
        code_provenance=base_context.code_provenance,
        criteria=criteria,
    )

    orchestrator.execute_campaign(
        campaign_id=defn.campaign_id,
        df=sample_dataset,
        search_space=search_space,
        criteria=criteria,
        execution_policy=ResearchCampaignExecutionPolicy(max_trials=10),
        persist_registry_dir=r_store.base_dir,
    )

    synthesis = synthesize_campaign_evidence(defn.campaign_id, store=c_store)
    c_store.save_evidence_synthesis(synthesis)

    sel_policy = ResearchCampaignSelectionPolicy(
        required_governance_states=("QUALIFIED", "PROMOTABLE", "VALIDATED", "REJECTED"),
        required_oos_evidence=False,
        required_walk_forward_evidence=False,
    )
    decision = select_campaign_candidate(synthesis, selection_policy=sel_policy)
    c_store.save_selection_decision(decision)

    artifact = materialize_governed_campaign_feedback(defn.campaign_id, store=c_store, registry_store=r_store)

    hypotheses = generate_hypotheses_from_campaign_learning(
        campaign_id=defn.campaign_id,
        context=base_context,
        store=c_store,
        registry_store=r_store,
    )

    assert len(hypotheses) > 0
    hyp = hypotheses[0]
    assert hyp.status == HypothesisStatus.GENERATED
    assert hyp.constraints["campaign_id"] == defn.campaign_id
    assert hyp.constraints["campaign_selection_decision_fingerprint"] == decision.decision_fingerprint
    assert hyp.constraints["synthesis_fingerprint"] == synthesis.synthesis_fingerprint
    assert hyp.constraints["campaign_learning_artifact_fingerprint"] == artifact.artifact_fingerprint


def test_feedback_loop_rejects_tampered_learning_artifact(tmp_stores, sample_dataset, base_context):
    c_store, r_store, _ = tmp_stores

    spec = CandidateGeneratorSpec(
        generator_name="test_gen",
        generator_version="1.0",
        strategy_name="baseline",
        parameter_grid={"fast_window": [5]},
    )
    gen = CandidateGenerator(spec)
    candidates = gen.generate_candidates()
    search_space = ResearchSearchSpace(candidate_definitions=candidates)

    criteria = DiscoveryCriteria(min_is_sharpe=-10.0, min_oos_sharpe=-10.0)
    orchestrator = ResearchCampaignOrchestrator(
        store=c_store, registry=DEFAULT_REGISTRY, memory_store=r_store
    )

    defn, _ = orchestrator.plan_campaign(
        search_space=search_space,
        dataset_scope=base_context.dataset_scope,
        execution_assumptions=base_context.execution_assumptions,
        code_provenance=base_context.code_provenance,
        criteria=criteria,
    )

    orchestrator.execute_campaign(
        campaign_id=defn.campaign_id,
        df=sample_dataset,
        search_space=search_space,
        criteria=criteria,
        execution_policy=ResearchCampaignExecutionPolicy(max_trials=10),
        persist_registry_dir=r_store.base_dir,
    )

    synthesis = synthesize_campaign_evidence(defn.campaign_id, store=c_store)
    c_store.save_evidence_synthesis(synthesis)

    sel_policy = ResearchCampaignSelectionPolicy(
        required_governance_states=("QUALIFIED", "PROMOTABLE", "VALIDATED", "REJECTED"),
        required_oos_evidence=False,
        required_walk_forward_evidence=False,
    )
    decision = select_campaign_candidate(synthesis, selection_policy=sel_policy)
    c_store.save_selection_decision(decision)

    materialize_governed_campaign_feedback(defn.campaign_id, store=c_store, registry_store=r_store)

    hypotheses = generate_hypotheses_from_campaign_learning(
        campaign_id=defn.campaign_id,
        context=base_context,
        store=c_store,
        registry_store=r_store,
    )

    hyp = hypotheses[0]

    # Tamper with stored campaign learning artifact
    art = c_store.load_campaign_learning(defn.campaign_id)
    tampered_art = GovernedCampaignLearningArtifact(
        campaign_id=art.campaign_id,
        campaign_selection_decision_fingerprint="TAMPERED_DECISION_FP",
        synthesis_fingerprint=art.synthesis_fingerprint,
        selected_candidate_ids=art.selected_candidate_ids,
        supporting_evidence_fingerprints=art.supporting_evidence_fingerprints,
        robustness_fingerprints=art.robustness_fingerprints,
        qualification_fingerprints=art.qualification_fingerprints,
        learning_record_ids=art.learning_record_ids,
        knowledge_pattern_ids=art.knowledge_pattern_ids,
    )
    # Force save tampered learning artifact
    art_path = c_store._campaign_dir(defn.campaign_id) / "campaign_learning.json"
    import json
    art_path.write_text(json.dumps(tampered_art.as_dict()), encoding="utf-8")

    with pytest.raises(CampaignLearningIntegrityError):
        validate_feedback_loop_lineage(defn.campaign_id, hyp, store=c_store, registry_store=r_store)


def test_feedback_loop_rejects_cross_campaign_lineage(tmp_stores, sample_dataset, base_context):
    c_store, r_store, _ = tmp_stores

    spec = CandidateGeneratorSpec(
        generator_name="test_gen",
        generator_version="1.0",
        strategy_name="baseline",
        parameter_grid={"fast_window": [5]},
    )
    gen = CandidateGenerator(spec)
    candidates = gen.generate_candidates()
    search_space = ResearchSearchSpace(candidate_definitions=candidates)

    criteria = DiscoveryCriteria(min_is_sharpe=-10.0, min_oos_sharpe=-10.0)
    orchestrator = ResearchCampaignOrchestrator(
        store=c_store, registry=DEFAULT_REGISTRY, memory_store=r_store
    )

    defn, _ = orchestrator.plan_campaign(
        search_space=search_space,
        dataset_scope=base_context.dataset_scope,
        execution_assumptions=base_context.execution_assumptions,
        code_provenance=base_context.code_provenance,
        criteria=criteria,
    )

    orchestrator.execute_campaign(
        campaign_id=defn.campaign_id,
        df=sample_dataset,
        search_space=search_space,
        criteria=criteria,
        execution_policy=ResearchCampaignExecutionPolicy(max_trials=10),
        persist_registry_dir=r_store.base_dir,
    )

    synthesis = synthesize_campaign_evidence(defn.campaign_id, store=c_store)
    c_store.save_evidence_synthesis(synthesis)

    sel_policy = ResearchCampaignSelectionPolicy(
        required_governance_states=("QUALIFIED", "PROMOTABLE", "VALIDATED", "REJECTED"),
        required_oos_evidence=False,
        required_walk_forward_evidence=False,
    )
    decision = select_campaign_candidate(synthesis, selection_policy=sel_policy)
    c_store.save_selection_decision(decision)

    materialize_governed_campaign_feedback(defn.campaign_id, store=c_store, registry_store=r_store)

    hypotheses = generate_hypotheses_from_campaign_learning(
        campaign_id=defn.campaign_id,
        context=base_context,
        store=c_store,
        registry_store=r_store,
    )

    hyp = hypotheses[0]

    # Validate against wrong campaign ID
    with pytest.raises(CampaignLearningIntegrityError, match="campaign_id mismatch"):
        validate_feedback_loop_lineage("OTHER_CAMPAIGN_ID", hyp, store=c_store, registry_store=r_store)
