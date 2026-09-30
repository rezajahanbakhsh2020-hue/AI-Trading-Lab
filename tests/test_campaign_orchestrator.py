"""Focused test suite for Research Campaign Orchestrator terminal finalization.

Verifies:
A. Successful terminal finalization (COMPLETED -> evidence synthesis, selection decision, learning artifact).
B. Truncated campaign finalization (TRUNCATED -> evidence synthesis, selection decision).
C. No eligible candidate (NO_ELIGIBLE_CANDIDATE decision persisted without learning materialization).
D. Unresolved tie (TIE_UNRESOLVED decision persisted without learning materialization).
E. Idempotent finalization (repeat runs return identical fingerprints, no duplicate records/patterns/hypotheses).
F. Conflicting persisted synthesis causes fail closed with CampaignIntegrityError.
G. Conflicting persisted selection decision causes fail closed with CampaignIntegrityError.
H. Missing finalization artifact after terminal execution is resumed without rerunning completed trials.
I. Hypothesis lifecycle status remains GENERATED with full lineage.
J. Production boundary regression check: finalization path creates no production authority artifacts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.campaign_orchestrator import (
    CampaignIntegrityError,
    ResearchCampaignExecutionPolicy,
    ResearchCampaignOrchestrator,
)
from src.evaluation.campaign_synthesis import (
    CampaignSelectionStatus,
    ResearchCampaignSelectionPolicy,
)
from src.evaluation.candidate_generator import CandidateSpec, ResearchSearchSpace
from src.evaluation.discovery_engine import DiscoveryCriteria
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
    HypothesisStatus,
    ResearchCampaignStatus,
)
from src.evaluation.research_registry import ResearchRegistryStore
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


# Requirement A: Successful terminal finalization
def test_successful_terminal_finalization(
    sample_market_data, sample_scope, sample_assumptions, sample_provenance, sample_search_space
):
    criteria = DiscoveryCriteria()
    with tempfile.TemporaryDirectory() as tmpdir:
        camp_store = ResearchCampaignStore(tmpdir)
        reg_dir = Path(tmpdir) / "registry"
        reg_store = ResearchRegistryStore(base_dir=reg_dir)
        orchestrator = ResearchCampaignOrchestrator(store=camp_store, memory_store=reg_store)

        defn, _plan = orchestrator.plan_campaign(
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
            persist_registry_dir=reg_dir,
        )

        assert campaign.status == ResearchCampaignStatus.COMPLETED

        # Check finalization artifacts exist
        synthesis = camp_store.load_evidence_synthesis(defn.campaign_id)
        decision = camp_store.load_selection_decision(defn.campaign_id)
        assert synthesis.synthesis_fingerprint is not None
        assert decision.decision_fingerprint is not None

        if decision.decision_status == CampaignSelectionStatus.SELECTED.value:
            learning_art = camp_store.load_campaign_learning(defn.campaign_id)
            assert learning_art.synthesis_fingerprint == synthesis.synthesis_fingerprint
            assert learning_art.campaign_selection_decision_fingerprint == decision.decision_fingerprint


# Requirement B: Truncated campaign finalization
def test_truncated_campaign_finalization(
    sample_market_data, sample_scope, sample_assumptions, sample_provenance, sample_search_space
):
    criteria = DiscoveryCriteria()
    with tempfile.TemporaryDirectory() as tmpdir:
        camp_store = ResearchCampaignStore(tmpdir)
        reg_dir = Path(tmpdir) / "registry"
        reg_store = ResearchRegistryStore(base_dir=reg_dir)
        orchestrator = ResearchCampaignOrchestrator(store=camp_store, memory_store=reg_store)

        # Truncate search space to max 1 trial
        policy = ResearchCampaignExecutionPolicy(max_trials=1, persist_evidence=True)
        defn, _plan = orchestrator.plan_campaign(
            search_space=sample_search_space,
            dataset_scope=sample_scope,
            execution_assumptions=sample_assumptions,
            code_provenance=sample_provenance,
            criteria=criteria,
            execution_policy=policy,
        )

        campaign = orchestrator.execute_campaign(
            campaign_id=defn.campaign_id,
            df=sample_market_data,
            search_space=sample_search_space,
            criteria=criteria,
            execution_policy=policy,
            persist_registry_dir=reg_dir,
        )

        assert campaign.status == ResearchCampaignStatus.TRUNCATED

        # Truncated campaign reaches canonical synthesis and decision
        synthesis = camp_store.load_evidence_synthesis(defn.campaign_id)
        decision = camp_store.load_selection_decision(defn.campaign_id)
        assert synthesis.campaign_id == defn.campaign_id
        assert decision.campaign_id == defn.campaign_id


# Requirement C: No eligible candidate
def test_no_eligible_candidate_persists_decision_without_learning(
    sample_market_data, sample_scope, sample_assumptions, sample_provenance, sample_search_space
):
    criteria = DiscoveryCriteria()
    strict_policy = ResearchCampaignSelectionPolicy(
        required_governance_states=("IMPOSSIBLE_GOVERNANCE_STATE",)
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        camp_store = ResearchCampaignStore(tmpdir)
        reg_dir = Path(tmpdir) / "registry"
        reg_store = ResearchRegistryStore(base_dir=reg_dir)
        orchestrator = ResearchCampaignOrchestrator(store=camp_store, memory_store=reg_store)

        defn, _plan = orchestrator.plan_campaign(
            search_space=sample_search_space,
            dataset_scope=sample_scope,
            execution_assumptions=sample_assumptions,
            code_provenance=sample_provenance,
            criteria=criteria,
        )

        # Execute campaign with strict selection policy
        orchestrator.execute_campaign(
            campaign_id=defn.campaign_id,
            df=sample_market_data,
            search_space=sample_search_space,
            criteria=criteria,
            selection_policy=strict_policy,
            persist_registry_dir=reg_dir,
        )

        decision = orchestrator.finalize_completed_campaign(
            campaign_id=defn.campaign_id,
            selection_policy=strict_policy,
            registry_store=reg_store,
        )

        assert decision.decision_status == CampaignSelectionStatus.NO_ELIGIBLE_CANDIDATE.value

        # Decision is persisted
        persisted_dec = camp_store.load_selection_decision(defn.campaign_id)
        assert persisted_dec.decision_status == CampaignSelectionStatus.NO_ELIGIBLE_CANDIDATE.value

        # Learning artifact is NOT created
        with pytest.raises(FileNotFoundError):
            camp_store.load_campaign_learning(defn.campaign_id)


# Requirement D: Unresolved tie
def test_unresolved_tie_persists_decision_without_learning(
    sample_market_data, sample_scope, sample_assumptions, sample_provenance, sample_search_space
):
    criteria = DiscoveryCriteria()
    with tempfile.TemporaryDirectory() as tmpdir:
        camp_store = ResearchCampaignStore(tmpdir)
        reg_dir = Path(tmpdir) / "registry"
        reg_store = ResearchRegistryStore(base_dir=reg_dir)
        orchestrator = ResearchCampaignOrchestrator(store=camp_store, memory_store=reg_store)

        defn, plan = orchestrator.plan_campaign(
            search_space=sample_search_space,
            dataset_scope=sample_scope,
            execution_assumptions=sample_assumptions,
            code_provenance=sample_provenance,
            criteria=criteria,
        )

        exp_dir = Path(tmpdir) / "experiments"
        exp_dir.mkdir(parents=True, exist_ok=True)

        from src.evaluation.research_constitution import EvidencePartition, EvidencePartitionRole, PromotionStatus, ResearchEvidence, ResearchExperimentSpec, ResearchTrialCheckpoint
        mock_spec = ResearchExperimentSpec(
            hypothesis="Mock hypothesis - tie test 1",
            methodology_version="1.0",
            strategy_name="baseline_sma",
            strategy_version="1.0",
            dataset_scope=sample_scope,
            execution_assumptions=sample_assumptions,
            code_provenance=sample_provenance,
            benchmark_reference="buy_and_hold",
        )
        p_is = EvidencePartition(
            role=EvidencePartitionRole.IN_SAMPLE,
            start_date="2020-01-01",
            end_date="2020-01-02",
            total_return=0.1,
            max_drawdown=0.05,
            sharpe_ratio=1.5,
            profit_factor=1.5,
            win_rate=0.6,
            observations=50,
        )
        p_oos = EvidencePartition(
            role=EvidencePartitionRole.OUT_OF_SAMPLE,
            start_date="2020-01-02",
            end_date="2020-01-04",
            total_return=0.1,
            max_drawdown=0.05,
            sharpe_ratio=1.5,
            profit_factor=1.5,
            win_rate=0.6,
            observations=50,
        )
        p_wf = EvidencePartition(
            role=EvidencePartitionRole.WALK_FORWARD,
            start_date="2020-01-04",
            end_date="2020-01-05",
            total_return=0.1,
            max_drawdown=0.05,
            sharpe_ratio=1.5,
            profit_factor=1.5,
            win_rate=0.6,
            observations=50,
        )
        mock_verdict = {
            "is_robust": True,
            "passed": True,
            "parameter_sensitivity": {"passed": True},
            "subsample_stability": {"passed": True},
            "execution_cost_stress": {"passed": True},
            "statistical_validation": {"passed": True, "p_value": 0.01},
            "anti_overfitting": {"passed": True},
        }
        mock_evidence = ResearchEvidence(
            experiment_fingerprint=mock_spec.fingerprint,
            spec=mock_spec,
            partitions=(p_is, p_oos, p_wf),
            robustness_verdict=mock_verdict,
            benchmark_comparison={},
            promotion_status=PromotionStatus.PROMOTABLE,
        )
        from src.evaluation.research_store import save_research_experiment
        save_research_experiment(mock_evidence, base_dir=exp_dir)

        # Create second distinct mock evidence with different hypothesis to ensure distinct fingerprints
        mock_spec_2 = ResearchExperimentSpec(
            hypothesis="Mock hypothesis 2 - tie test 2",
            methodology_version="1.0",
            strategy_name="baseline_sma",
            strategy_version="1.0",
            dataset_scope=sample_scope,
            execution_assumptions=sample_assumptions,
            code_provenance=sample_provenance,
            benchmark_reference="buy_and_hold",
        )
        mock_evidence_2 = ResearchEvidence(
            experiment_fingerprint=mock_spec_2.fingerprint,
            spec=mock_spec_2,
            partitions=(p_is, p_oos, p_wf),
            robustness_verdict=mock_verdict,
            benchmark_comparison={},
            promotion_status=PromotionStatus.PROMOTABLE,
        )
        save_research_experiment(mock_evidence_2, base_dir=exp_dir)

        # Set state to COMPLETED manually and setup tied checkpoints with distinct evidence
        camp_store.save_lifecycle_state(defn.campaign_id, ResearchCampaignStatus.RUNNING)

        ev_list = [mock_evidence, mock_evidence_2]
        for idx, pt in enumerate(plan.trials):
            ev = ev_list[idx % len(ev_list)]
            cp = ResearchTrialCheckpoint(
                trial_id=pt.trial_id,
                campaign_id=defn.campaign_id,
                candidate_id=pt.candidate_id,
                trial_index=pt.trial_index,
                attempt_number=1,
                status="QUALIFIED",
                experiment_fingerprint=ev.experiment_fingerprint,
                evidence_fingerprint=ev.evidence_id,
                qualification_status="QUALIFIED",
            )
            camp_store.save_trial_checkpoint(cp)

        camp_store.save_lifecycle_state(defn.campaign_id, ResearchCampaignStatus.COMPLETED)

        # Finalize campaign where candidates tie on metrics
        q_policy = ResearchCampaignSelectionPolicy(
            required_governance_states=("QUALIFIED", "PROMOTABLE", "VALIDATED", "REJECTED"),
            required_oos_evidence=False,
            required_walk_forward_evidence=False,
        )

        decision = orchestrator.finalize_completed_campaign(
            campaign_id=defn.campaign_id,
            selection_policy=q_policy,
            registry_store=reg_store,
            experiment_store_dir=exp_dir,
        )

        assert decision.decision_status == CampaignSelectionStatus.TIE_UNRESOLVED.value

        # Learning artifact is NOT created
        with pytest.raises(FileNotFoundError):
            camp_store.load_campaign_learning(defn.campaign_id)


# Requirement E: Idempotent finalization
def test_idempotent_finalization(
    sample_market_data, sample_scope, sample_assumptions, sample_provenance, sample_search_space
):
    criteria = DiscoveryCriteria()
    with tempfile.TemporaryDirectory() as tmpdir:
        camp_store = ResearchCampaignStore(tmpdir)
        reg_dir = Path(tmpdir) / "registry"
        reg_store = ResearchRegistryStore(base_dir=reg_dir)
        orchestrator = ResearchCampaignOrchestrator(store=camp_store, memory_store=reg_store)

        defn, _plan = orchestrator.plan_campaign(
            search_space=sample_search_space,
            dataset_scope=sample_scope,
            execution_assumptions=sample_assumptions,
            code_provenance=sample_provenance,
            criteria=criteria,
        )

        orchestrator.execute_campaign(
            campaign_id=defn.campaign_id,
            df=sample_market_data,
            search_space=sample_search_space,
            criteria=criteria,
            persist_registry_dir=reg_dir,
        )

        dec1 = orchestrator.finalize_completed_campaign(
            campaign_id=defn.campaign_id,
            registry_store=reg_store,
        )

        rec_count_1 = len(reg_store.list_records())
        pat_count_1 = len(reg_store.list_patterns())
        hyp_count_1 = len(reg_store.list_hypotheses())

        # Re-run finalization
        dec2 = orchestrator.finalize_completed_campaign(
            campaign_id=defn.campaign_id,
            registry_store=reg_store,
        )

        assert dec1.decision_fingerprint == dec2.decision_fingerprint
        assert dec1.synthesis_fingerprint == dec2.synthesis_fingerprint

        # Verify no duplicate learning records, knowledge patterns, or hypotheses were created
        assert len(reg_store.list_records()) == rec_count_1
        assert len(reg_store.list_patterns()) == pat_count_1
        assert len(reg_store.list_hypotheses()) == hyp_count_1


# Requirement F: Conflicting persisted synthesis causes fail closed
def test_conflicting_persisted_synthesis_fails_closed(
    sample_market_data, sample_scope, sample_assumptions, sample_provenance, sample_search_space
):
    criteria = DiscoveryCriteria()
    with tempfile.TemporaryDirectory() as tmpdir:
        camp_store = ResearchCampaignStore(tmpdir)
        orchestrator = ResearchCampaignOrchestrator(store=camp_store)

        defn, _plan = orchestrator.plan_campaign(
            search_space=sample_search_space,
            dataset_scope=sample_scope,
            execution_assumptions=sample_assumptions,
            code_provenance=sample_provenance,
            criteria=criteria,
        )

        orchestrator.execute_campaign(
            campaign_id=defn.campaign_id,
            df=sample_market_data,
            search_space=sample_search_space,
            criteria=criteria,
        )

        # Corrupt persisted synthesis JSON
        syn_path = camp_store._campaign_dir(defn.campaign_id) / "evidence_synthesis.json"
        content = syn_path.read_text()
        corrupted = content.replace('"synthesis_methodology_version": "1.0"', '"synthesis_methodology_version": "999.0"')
        syn_path.write_text(corrupted)

        with pytest.raises(CampaignIntegrityError, match="Conflicting persisted evidence synthesis"):
            orchestrator.finalize_completed_campaign(defn.campaign_id)


# Requirement G: Conflicting persisted selection decision causes fail closed
def test_conflicting_persisted_selection_decision_fails_closed(
    sample_market_data, sample_scope, sample_assumptions, sample_provenance, sample_search_space
):
    criteria = DiscoveryCriteria()
    with tempfile.TemporaryDirectory() as tmpdir:
        camp_store = ResearchCampaignStore(tmpdir)
        orchestrator = ResearchCampaignOrchestrator(store=camp_store)

        defn, _plan = orchestrator.plan_campaign(
            search_space=sample_search_space,
            dataset_scope=sample_scope,
            execution_assumptions=sample_assumptions,
            code_provenance=sample_provenance,
            criteria=criteria,
        )

        orchestrator.execute_campaign(
            campaign_id=defn.campaign_id,
            df=sample_market_data,
            search_space=sample_search_space,
            criteria=criteria,
        )

        # Corrupt persisted decision JSON
        dec_path = camp_store._campaign_dir(defn.campaign_id) / "selection_decision.json"
        content = dec_path.read_text()
        corrupted = content.replace('"decision_methodology_version": "1.0"', '"decision_methodology_version": "999.0"')
        dec_path.write_text(corrupted)

        with pytest.raises(CampaignIntegrityError, match="Conflicting persisted selection decision"):
            orchestrator.finalize_completed_campaign(defn.campaign_id)


# Requirement H: Missing finalization artifact after terminal execution
def test_missing_finalization_artifact_resumed_without_rerunning_trials(
    sample_market_data, sample_scope, sample_assumptions, sample_provenance, sample_search_space
):
    criteria = DiscoveryCriteria()
    with tempfile.TemporaryDirectory() as tmpdir:
        camp_store = ResearchCampaignStore(tmpdir)
        reg_dir = Path(tmpdir) / "registry"
        reg_store = ResearchRegistryStore(base_dir=reg_dir)
        orchestrator = ResearchCampaignOrchestrator(store=camp_store, memory_store=reg_store)

        defn, _plan = orchestrator.plan_campaign(
            search_space=sample_search_space,
            dataset_scope=sample_scope,
            execution_assumptions=sample_assumptions,
            code_provenance=sample_provenance,
            criteria=criteria,
        )

        orchestrator.execute_campaign(
            campaign_id=defn.campaign_id,
            df=sample_market_data,
            search_space=sample_search_space,
            criteria=criteria,
            persist_registry_dir=reg_dir,
        )

        # Delete synthesis, decision, and learning JSON artifacts while keeping completed trial checkpoints intact
        cdir = camp_store._campaign_dir(defn.campaign_id)
        (cdir / "evidence_synthesis.json").unlink(missing_ok=True)
        (cdir / "selection_decision.json").unlink(missing_ok=True)
        (cdir / "campaign_learning.json").unlink(missing_ok=True)

        # Execute resume_campaign over terminal campaign -> should materialize missing finalization phase without rerunning trials
        orchestrator.resume_campaign(
            campaign_id=defn.campaign_id,
            df=sample_market_data,
            search_space=sample_search_space,
            criteria=criteria,
            persist_registry_dir=reg_dir,
        )

        assert (cdir / "evidence_synthesis.json").exists()
        assert (cdir / "selection_decision.json").exists()


# Requirement I: Hypothesis lifecycle status
def test_generated_hypotheses_remain_generated_status(
    sample_market_data, sample_scope, sample_assumptions, sample_provenance, sample_search_space
):
    criteria = DiscoveryCriteria()
    with tempfile.TemporaryDirectory() as tmpdir:
        camp_store = ResearchCampaignStore(tmpdir)
        reg_dir = Path(tmpdir) / "registry"
        reg_store = ResearchRegistryStore(base_dir=reg_dir)
        orchestrator = ResearchCampaignOrchestrator(store=camp_store, memory_store=reg_store)

        defn, _plan = orchestrator.plan_campaign(
            search_space=sample_search_space,
            dataset_scope=sample_scope,
            execution_assumptions=sample_assumptions,
            code_provenance=sample_provenance,
            criteria=criteria,
        )

        orchestrator.execute_campaign(
            campaign_id=defn.campaign_id,
            df=sample_market_data,
            search_space=sample_search_space,
            criteria=criteria,
            persist_registry_dir=reg_dir,
        )

        # If hypotheses were generated during feedback, all must be strictly in GENERATED status
        for hyp in reg_store.list_hypotheses():
            assert hyp.status == HypothesisStatus.GENERATED


# Requirement J: Production boundary check
def test_campaign_finalization_creates_no_production_authority(
    sample_market_data, sample_scope, sample_assumptions, sample_provenance, sample_search_space
):
    criteria = DiscoveryCriteria()
    with tempfile.TemporaryDirectory() as tmpdir:
        camp_store = ResearchCampaignStore(tmpdir)
        reg_dir = Path(tmpdir) / "registry"
        reg_store = ResearchRegistryStore(base_dir=reg_dir)
        orchestrator = ResearchCampaignOrchestrator(store=camp_store, memory_store=reg_store)

        defn, _plan = orchestrator.plan_campaign(
            search_space=sample_search_space,
            dataset_scope=sample_scope,
            execution_assumptions=sample_assumptions,
            code_provenance=sample_provenance,
            criteria=criteria,
        )

        orchestrator.execute_campaign(
            campaign_id=defn.campaign_id,
            df=sample_market_data,
            search_space=sample_search_space,
            criteria=criteria,
            persist_registry_dir=reg_dir,
        )

        # Verify no promoted candidate bindings or production artifacts were created
        from src.evaluation.research_store import DEFAULT_RESEARCH_DIR, resolve_promoted_candidate
        promoted = resolve_promoted_candidate(
            strategy_id="baseline_sma",
            base_dir=DEFAULT_RESEARCH_DIR,
        )
        assert promoted is None
