"""Dedicated test suite for Governed Research Campaign Evidence Synthesis & Selection Lifecycle."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
import pytest

from src.evaluation.campaign_synthesis import (
    CampaignSelectionIntegrityError,
    CampaignSelectionStatus,
    ResearchCampaignEvidenceSynthesis,
    ResearchCampaignSelectionPolicy,
    collect_and_synthesize_campaign_evidence,
    select_campaign_candidate,
    validate_campaign_selection_integrity,
)
from src.evaluation.hypothesis_generator import accept_hypothesis_for_research
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
    PromotionStatus,
    ResearchCampaignDefinition,
    ResearchCampaignStatus,
    ResearchEvidence,
    ResearchExperimentSpec,
    ResearchHypothesis,
    ResearchPlannedTrial,
    ResearchTrialCheckpoint,
    ResearchTrialPlan,
    WalkForwardProtocol,
)
from src.evaluation.research_runner import run_research_experiment
from src.evaluation.research_store import (
    ResearchCampaignStore,
    save_research_experiment,
)


@pytest.fixture
def temp_research_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _create_mock_evidence(
    tmp_dir: Path,
    strategy_name: str = "momentum",
    candidate_id: str = "cand_1",
    sharpe: float = 1.5,
    profit_factor: float = 1.8,
    dataset_id: str = "test_ds",
    symbol: str = "XAUUSD",
) -> ResearchEvidence:
    spec = ResearchExperimentSpec(
        hypothesis=f"Test hypothesis for {candidate_id}",
        methodology_version="1.0",
        strategy_name=strategy_name,
        strategy_version="1.0.0",
        dataset_scope=DatasetScope(
            dataset_id=dataset_id,
            symbol=symbol,
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-06-01",
        ),
        execution_assumptions=ExecutionAssumptions(
            transaction_cost=0.0001,
            slippage=0.0001,
            latency_ms=10.0,
        ),
        code_provenance=CodeProvenance(
            commit_sha="abc123456789def123456789abc123456789def1",
            repository_status="clean",
            author="tester",
        ),
        benchmark_reference="buy_and_hold",
        parameters={"window": 10, "stop_loss_pct": 0.02, "take_profit_pct": 0.04},
        walk_forward_protocol=WalkForwardProtocol(train_size=40, test_size=15),
    )

    hyp = accept_hypothesis_for_research(ResearchHypothesis.from_experiment_spec(spec))
    orig_evidence = run_research_experiment(hyp)

    from src.evaluation.research_constitution import EvidencePartition, EvidencePartitionRole

    parts = (
        EvidencePartition(
            role=EvidencePartitionRole.IN_SAMPLE,
            start_date="2024-01-01",
            end_date="2024-03-01",
            total_return=0.1,
            max_drawdown=0.05,
            sharpe_ratio=sharpe,
            win_rate=0.6,
            profit_factor=profit_factor,
            observations=100,
            start_timestamp_utc="2024-01-01T00:00:00+00:00",
            end_timestamp_utc="2024-03-01T00:00:00+00:00",
        ),
        EvidencePartition(
            role=EvidencePartitionRole.OUT_OF_SAMPLE,
            start_date="2024-03-02",
            end_date="2024-05-01",
            total_return=0.1,
            max_drawdown=0.05,
            sharpe_ratio=sharpe,
            win_rate=0.6,
            profit_factor=profit_factor,
            observations=100,
            start_timestamp_utc="2024-03-02T00:00:00+00:00",
            end_timestamp_utc="2024-05-01T00:00:00+00:00",
        ),
        EvidencePartition(
            role=EvidencePartitionRole.WALK_FORWARD,
            start_date="2024-05-01",
            end_date="2024-06-01",
            total_return=0.1,
            max_drawdown=0.05,
            sharpe_ratio=sharpe,
            win_rate=0.6,
            profit_factor=profit_factor,
            observations=100,
            start_timestamp_utc="2024-05-01T00:00:00+00:00",
            end_timestamp_utc="2024-06-01T00:00:00+00:00",
        ),
    )

    rob_verdict = {
        "passed": True,
        "parameter_sensitivity": {"passed": True},
        "subsample_stability": {"passed": True},
        "execution_cost_stress": {"passed": True},
        "statistical_validation": {"passed": True, "p_value": 0.01},
        "anti_overfitting": {"passed": True},
    }

    evidence = ResearchEvidence(
        experiment_fingerprint=orig_evidence.experiment_fingerprint,
        spec=spec,
        partitions=parts,
        robustness_verdict=rob_verdict,
        benchmark_comparison=orig_evidence.benchmark_comparison,
        promotion_status=PromotionStatus.PROMOTABLE,
        rejection_reasons=(),
        critique_notes=orig_evidence.critique_notes,
    )

    save_research_experiment(evidence, base_dir=tmp_dir)
    return evidence


def _setup_mock_campaign(
    campaign_dir: Path,
    exp_dir: Path,
    candidate_specs: list[dict],
) -> tuple[str, ResearchCampaignStore]:
    store = ResearchCampaignStore(base_dir=campaign_dir)

    ds = DatasetScope(
        dataset_id="test_ds",
        symbol="XAUUSD",
        timeframe="5m",
        start_date="2024-01-01",
        end_date="2024-06-01",
    )
    ea = ExecutionAssumptions(transaction_cost=0.0001, slippage=0.0001, latency_ms=10.0)
    cp = CodeProvenance(
        commit_sha="abc123456789def123456789abc123456789def1",
        repository_status="clean",
        author="tester",
    )

    cand_ids = [c["candidate_id"] for c in candidate_specs]

    defn = ResearchCampaignDefinition(
        search_space_fingerprint="sp_fp_1",
        search_policy_fingerprint="pol_fp_1",
        criteria_fingerprint="crit_fp_1",
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        methodology_version="1.0",
        candidate_ids=tuple(cand_ids),
        trial_count=len(cand_ids),
    )
    campaign_id = defn.campaign_id
    store.save_definition(defn)

    planned_trials = []
    for idx, c_spec in enumerate(candidate_specs):
        cid = c_spec["candidate_id"]
        t = ResearchPlannedTrial(
            campaign_id=campaign_id,
            trial_id=f"trial_{idx}",
            trial_index=idx,
            candidate_id=cid,
            candidate_fingerprint=f"cand_fp_{cid}",
            hypothesis_fingerprint=f"hyp_fp_{cid}",
            strategy_name=c_spec.get("strategy_name", "momentum"),
            strategy_version="1.0.0",
            dataset_id="test_ds",
            execution_assumptions_id="ea_1",
            planned_status="PENDING",
        )
        planned_trials.append(t)

    plan = ResearchTrialPlan(
        campaign_id=campaign_id,
        definition_fingerprint=defn.definition_fingerprint,
        trials=tuple(planned_trials),
    )
    store.save_trial_plan(plan)
    store.save_lifecycle_state(campaign_id, ResearchCampaignStatus.RUNNING)

    for idx, c_spec in enumerate(candidate_specs):
        cid = c_spec["candidate_id"]
        trial_id = f"trial_{idx}"
        status = c_spec.get("status", "COMPLETED")

        if status in ("FAILED", "ERROR"):
            cp = ResearchTrialCheckpoint(
                trial_id=trial_id,
                campaign_id=campaign_id,
                candidate_id=cid,
                trial_index=idx,
                attempt_number=1,
                status=status,
                error_message=c_spec.get("error_message", "Execution error"),
            )
        elif status == "BLOCKED":
            cp = ResearchTrialCheckpoint(
                trial_id=trial_id,
                campaign_id=campaign_id,
                candidate_id=cid,
                trial_index=idx,
                attempt_number=1,
                status="BLOCKED",
                error_message="Governance blocked",
            )
        else:
            ev = _create_mock_evidence(
                tmp_dir=exp_dir,
                strategy_name=c_spec.get("strategy_name", "momentum"),
                candidate_id=cid,
                sharpe=c_spec.get("sharpe", 1.5),
                profit_factor=c_spec.get("profit_factor", 1.8),
                dataset_id="test_ds",
            )
            cp = ResearchTrialCheckpoint(
                trial_id=trial_id,
                campaign_id=campaign_id,
                candidate_id=cid,
                trial_index=idx,
                attempt_number=1,
                status="COMPLETED",
                experiment_fingerprint=ev.spec.fingerprint,
                evidence_fingerprint=ev.evidence_id,
                qualification_status="QUALIFIED",
            )

        store.save_trial_checkpoint(cp)

    store.save_lifecycle_state(campaign_id, ResearchCampaignStatus.COMPLETED)
    return campaign_id, store


# =============================================================================
# 1. Campaign Identity & Fingerprinting Tests
# =============================================================================

def test_campaign_synthesis_fingerprint_deterministic(temp_research_dir):
    camp_dir = temp_research_dir / "campaigns"
    exp_dir = temp_research_dir / "experiments"

    cid, store = _setup_mock_campaign(
        camp_dir, exp_dir,
        [
            {"candidate_id": "c1", "sharpe": 1.5, "profit_factor": 1.8},
            {"candidate_id": "c2", "sharpe": 2.0, "profit_factor": 2.2},
        ]
    )

    synthesis1 = collect_and_synthesize_campaign_evidence(cid, store=store, experiment_store_dir=exp_dir)
    synthesis2 = collect_and_synthesize_campaign_evidence(cid, store=store, experiment_store_dir=exp_dir)

    assert synthesis1.synthesis_fingerprint == synthesis2.synthesis_fingerprint
    assert len(synthesis1.synthesis_fingerprint) == 64


def test_selection_decision_fingerprint_deterministic(temp_research_dir):
    camp_dir = temp_research_dir / "campaigns"
    exp_dir = temp_research_dir / "experiments"

    cid, store = _setup_mock_campaign(
        camp_dir, exp_dir,
        [
            {"candidate_id": "c1", "sharpe": 1.5},
            {"candidate_id": "c2", "sharpe": 2.0},
        ]
    )

    synthesis = collect_and_synthesize_campaign_evidence(cid, store=store, experiment_store_dir=exp_dir)
    policy = ResearchCampaignSelectionPolicy()

    decision1 = select_campaign_candidate(synthesis, selection_policy=policy)
    decision2 = select_campaign_candidate(synthesis, selection_policy=policy)

    assert decision1.decision_fingerprint == decision2.decision_fingerprint
    assert decision1.decision_status == CampaignSelectionStatus.SELECTED.value
    assert decision1.selected_candidate_ids == ("c2",)


def test_policy_change_changes_fingerprints(temp_research_dir):
    camp_dir = temp_research_dir / "campaigns"
    exp_dir = temp_research_dir / "experiments"

    cid, store = _setup_mock_campaign(
        camp_dir, exp_dir,
        [{"candidate_id": "c1", "sharpe": 1.5}]
    )

    synthesis = collect_and_synthesize_campaign_evidence(cid, store=store, experiment_store_dir=exp_dir)

    policy1 = ResearchCampaignSelectionPolicy(required_oos_evidence=True)
    policy2 = ResearchCampaignSelectionPolicy(required_oos_evidence=False)

    assert policy1.policy_fingerprint != policy2.policy_fingerprint

    dec1 = select_campaign_candidate(synthesis, selection_policy=policy1)
    dec2 = select_campaign_candidate(synthesis, selection_policy=policy2)

    assert dec1.decision_fingerprint != dec2.decision_fingerprint


# =============================================================================
# 2. Evidence Collection & Integrity Tests
# =============================================================================

def test_missing_trial_checkpoint_fails_closed(temp_research_dir):
    camp_dir = temp_research_dir / "campaigns"
    exp_dir = temp_research_dir / "experiments"

    cid, store = _setup_mock_campaign(
        camp_dir, exp_dir,
        [
            {"candidate_id": "c1", "sharpe": 1.5},
            {"candidate_id": "c2", "sharpe": 2.0},
        ]
    )

    # Delete checkpoint for c2
    cp_file = camp_dir / cid / "checkpoints" / "trial_1.json"
    cp_file.unlink()

    with pytest.raises(ValueError, match="Missing expected checkpoint for trial"):
        collect_and_synthesize_campaign_evidence(cid, store=store, experiment_store_dir=exp_dir)


def test_cross_campaign_evidence_contamination_fails_closed(temp_research_dir):
    camp_dir = temp_research_dir / "campaigns"
    exp_dir = temp_research_dir / "experiments"

    cid, store = _setup_mock_campaign(
        camp_dir, exp_dir,
        [{"candidate_id": "c1", "sharpe": 1.5}]
    )

    # Tamper checkpoint campaign_id
    cp = store.load_trial_checkpoint(cid, "trial_0")
    tampered_cp = ResearchTrialCheckpoint(
        trial_id=cp.trial_id,
        campaign_id="OTHER_CAMPAIGN",
        candidate_id=cp.candidate_id,
        trial_index=cp.trial_index,
        attempt_number=cp.attempt_number,
        status=cp.status,
        experiment_fingerprint=cp.experiment_fingerprint,
        evidence_fingerprint=cp.evidence_fingerprint,
    )

    cp_file = camp_dir / cid / "checkpoints" / "trial_0.json"
    cp_file.write_text(json.dumps(tampered_cp.as_dict()), encoding="utf-8")

    with pytest.raises(ValueError, match="belongs to campaign 'OTHER_CAMPAIGN'"):
        collect_and_synthesize_campaign_evidence(cid, store=store, experiment_store_dir=exp_dir)


def test_candidate_mismatch_fails_closed(temp_research_dir):
    camp_dir = temp_research_dir / "campaigns"
    exp_dir = temp_research_dir / "experiments"

    cid, store = _setup_mock_campaign(
        camp_dir, exp_dir,
        [{"candidate_id": "c1", "sharpe": 1.5}]
    )

    cp = store.load_trial_checkpoint(cid, "trial_0")
    tampered_cp = ResearchTrialCheckpoint(
        trial_id=cp.trial_id,
        campaign_id=cp.campaign_id,
        candidate_id="WRONG_CANDIDATE",
        trial_index=cp.trial_index,
        attempt_number=cp.attempt_number,
        status=cp.status,
        experiment_fingerprint=cp.experiment_fingerprint,
        evidence_fingerprint=cp.evidence_fingerprint,
    )

    cp_file = camp_dir / cid / "checkpoints" / "trial_0.json"
    cp_file.write_text(json.dumps(tampered_cp.as_dict()), encoding="utf-8")

    with pytest.raises(ValueError, match="Candidate ID mismatch in trial"):
        collect_and_synthesize_campaign_evidence(cid, store=store, experiment_store_dir=exp_dir)


# =============================================================================
# 3. Governance & Selection Rules Tests
# =============================================================================

def test_selection_no_eligible_candidates(temp_research_dir):
    camp_dir = temp_research_dir / "campaigns"
    exp_dir = temp_research_dir / "experiments"

    cid, store = _setup_mock_campaign(
        camp_dir, exp_dir,
        [
            {"candidate_id": "c1", "status": "FAILED"},
            {"candidate_id": "c2", "status": "BLOCKED"},
        ]
    )

    synthesis = collect_and_synthesize_campaign_evidence(cid, store=store, experiment_store_dir=exp_dir)
    decision = select_campaign_candidate(synthesis)

    assert decision.decision_status == CampaignSelectionStatus.NO_ELIGIBLE_CANDIDATE.value
    assert decision.selected_candidate_ids == ()
    assert "c1" in decision.rejected_candidate_ids
    assert "c2" in decision.blocked_candidate_ids


def test_deterministic_tie_and_unresolved_tie(temp_research_dir):
    camp_dir = temp_research_dir / "campaigns"
    exp_dir = temp_research_dir / "experiments"

    cid, store = _setup_mock_campaign(
        camp_dir, exp_dir,
        [
            {"candidate_id": "c1", "sharpe": 2.0, "profit_factor": 1.8},
            {"candidate_id": "c2", "sharpe": 2.0, "profit_factor": 1.8},
        ]
    )

    synthesis = collect_and_synthesize_campaign_evidence(cid, store=store, experiment_store_dir=exp_dir)

    # Policy without secondary tie-breaker -> TIE_UNRESOLVED
    policy_no_tiebreak = ResearchCampaignSelectionPolicy()
    dec1 = select_campaign_candidate(synthesis, selection_policy=policy_no_tiebreak)
    assert dec1.decision_status == CampaignSelectionStatus.TIE_UNRESOLVED.value
    assert dec1.selected_candidate_ids == ()

    # Policy with secondary tie-breaker -> RESOLVED if metrics differ or TIE_UNRESOLVED if identical
    policy_tiebreak = ResearchCampaignSelectionPolicy(
        tie_handling_policy={"secondary_metric": "total_trades"}
    )
    dec2 = select_campaign_candidate(synthesis, selection_policy=policy_tiebreak)
    # Total trades are identical in mock evidence -> still unresolved
    assert dec2.decision_status == CampaignSelectionStatus.TIE_UNRESOLVED.value


# =============================================================================
# 4. Persistence & Integrity Round-Trip Tests
# =============================================================================

def test_persistence_round_trip_and_idempotency(temp_research_dir):
    camp_dir = temp_research_dir / "campaigns"
    exp_dir = temp_research_dir / "experiments"

    cid, store = _setup_mock_campaign(
        camp_dir, exp_dir,
        [{"candidate_id": "c1", "sharpe": 1.5}]
    )

    synthesis = collect_and_synthesize_campaign_evidence(cid, store=store, experiment_store_dir=exp_dir)
    decision = select_campaign_candidate(synthesis)

    p_syn = store.save_evidence_synthesis(synthesis)
    p_dec = store.save_selection_decision(decision)

    assert p_syn.exists()
    assert p_dec.exists()

    loaded_syn = store.load_evidence_synthesis(cid)
    loaded_dec = store.load_selection_decision(cid)

    assert loaded_syn.synthesis_fingerprint == synthesis.synthesis_fingerprint
    assert loaded_dec.decision_fingerprint == decision.decision_fingerprint

    # Idempotent save succeeds
    store.save_evidence_synthesis(synthesis)
    store.save_selection_decision(decision)


def test_corrupted_synthesis_fails_integrity(temp_research_dir):
    camp_dir = temp_research_dir / "campaigns"
    exp_dir = temp_research_dir / "experiments"

    cid, store = _setup_mock_campaign(
        camp_dir, exp_dir,
        [{"candidate_id": "c1", "sharpe": 1.5}]
    )

    synthesis = collect_and_synthesize_campaign_evidence(cid, store=store, experiment_store_dir=exp_dir)

    # Create tampered synthesis object with mismatched fingerprint
    tampered = ResearchCampaignEvidenceSynthesis(
        campaign_id=synthesis.campaign_id,
        campaign_definition_fingerprint=synthesis.campaign_definition_fingerprint,
        trial_plan_fingerprint=synthesis.trial_plan_fingerprint,
        search_space_fingerprint=synthesis.search_space_fingerprint,
        search_policy_fingerprint=synthesis.search_policy_fingerprint,
        criteria_fingerprint=synthesis.criteria_fingerprint,
        dataset_identity=synthesis.dataset_identity,
        execution_assumptions=synthesis.execution_assumptions,
        code_provenance=synthesis.code_provenance,
        methodology_version=synthesis.methodology_version,
        ordered_trial_identities=synthesis.ordered_trial_identities,
        ordered_evidence_fingerprints=synthesis.ordered_evidence_fingerprints,
        completed_trial_count=999,  # TAMPERED COUNT
        failed_trial_count=synthesis.failed_trial_count,
        blocked_trial_count=synthesis.blocked_trial_count,
        qualified_candidate_count=synthesis.qualified_candidate_count,
        rejected_candidate_count=synthesis.rejected_candidate_count,
        selection_governance_status=synthesis.selection_governance_status,
        selection_policy_fingerprint=synthesis.selection_policy_fingerprint,
        candidate_comparisons=synthesis.candidate_comparisons,
    )

    # Manually override synthesis_fingerprint to trick validator
    object.__setattr__(tampered, "synthesis_fingerprint", synthesis.synthesis_fingerprint)

    with pytest.raises(CampaignSelectionIntegrityError, match="Synthesis fingerprint mismatch"):
        validate_campaign_selection_integrity(tampered, store=store)


# =============================================================================
# 5. Static AST Architecture Guard Tests
# =============================================================================

def test_ast_architectural_guards():
    synth_file = Path(__file__).resolve().parents[1] / "src" / "evaluation" / "campaign_synthesis.py"
    tree = ast.parse(synth_file.read_text(encoding="utf-8"))

    forbidden_terms = [
        "ProductionRuntimeAuthorization",
        "select_production_strategy",
        "Project2Publisher",
        "project2",
    ]

    code_str = synth_file.read_text(encoding="utf-8")
    for term in forbidden_terms:
        assert term not in code_str, f"Forbidden architecture term '{term}' found in campaign_synthesis.py!"

    # Verify no imports from project2 or production runtime
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "project2" not in alias.name.lower()
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert "project2" not in node.module.lower()
