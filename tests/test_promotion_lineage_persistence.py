"""Focused tests for promotion-lineage persistence boundary (PR #44).

Verifies fail-closed preservation of governance_decision_fingerprint and
campaign_selection_decision_fingerprint across candidate binding persistence
and reconstitution.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.evaluation.campaign_synthesis import (
    CampaignSelectionStatus,
    ResearchCampaignSelectionDecision,
)
from src.evaluation.live_production_decision import (
    PromotedCandidateArtifact,
)
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    EvidencePartition,
    EvidencePartitionRole,
    ExecutionAssumptions,
    PromotionStatus,
    RejectionReason,
    ResearchEvidence,
    ResearchExperimentSpec,
)
from src.evaluation.research_store import (
    PromotionEligibilityError,
    PromotionIntegrityError,
    load_candidate_binding,
    persist_promoted_candidate_binding,
    resolve_promoted_candidate,
    save_research_experiment,
)


def make_test_evidence(
    *,
    symbol: str = "XAUUSD",
    timeframe: str = "5m",
    strategy_name: str = "momentum",
    strategy_version: str = "1.0",
    status: PromotionStatus = PromotionStatus.PROMOTABLE,
) -> ResearchEvidence:
    spec = ResearchExperimentSpec(
        hypothesis="PR #44 test hypothesis",
        methodology_version="1.0",
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        dataset_scope=DatasetScope(
            dataset_id="ds_test",
            symbol=symbol,
            timeframe=timeframe,
            start_date="2025-01-01",
            end_date="2025-02-01",
        ),
        execution_assumptions=ExecutionAssumptions(
            transaction_cost=0.0001,
            slippage=0.0001,
            latency_ms=10.0,
        ),
        code_provenance=CodeProvenance(
            commit_sha="abcd1234efgh5678",
            repository_status="clean",
            author="test",
        ),
        benchmark_reference="buy_and_hold",
        parameters={"momentum_window": 10, "stop_loss_pct": 0.01, "take_profit_pct": 0.02},
    )

    part_is = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2025-01-01",
        end_date="2025-01-14",
        total_return=0.15,
        max_drawdown=0.03,
        sharpe_ratio=2.5,
        win_rate=0.65,
        profit_factor=2.0,
        observations=100,
        start_timestamp_utc="2025-01-01T00:00:00+00:00",
        end_timestamp_utc="2025-01-14T00:00:00+00:00",
    )

    part_oos = EvidencePartition(
        role=EvidencePartitionRole.OUT_OF_SAMPLE,
        start_date="2025-01-15",
        end_date="2025-02-01",
        total_return=0.10,
        max_drawdown=0.02,
        sharpe_ratio=2.0,
        win_rate=0.60,
        profit_factor=1.8,
        observations=100,
        start_timestamp_utc="2025-01-15T00:00:00+00:00",
        end_timestamp_utc="2025-02-01T00:00:00+00:00",
    )

    part_wf = EvidencePartition(
        role=EvidencePartitionRole.WALK_FORWARD,
        start_date="2025-01-01",
        end_date="2025-02-01",
        total_return=0.12,
        max_drawdown=0.025,
        sharpe_ratio=2.1,
        win_rate=0.62,
        profit_factor=1.9,
        observations=200,
        start_timestamp_utc="2025-01-01T00:00:00+00:00",
        end_timestamp_utc="2025-02-01T00:00:00+00:00",
    )

    return ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(part_is, part_oos, part_wf),
        robustness_verdict={
            "passed": True,
            "is_robust": True,
            "parameter_sensitivity": {"passed": True},
            "subsample_stability": {"passed": True},
            "execution_cost_stress": {"passed": True},
            "statistical_validation": {"passed": True},
            "anti_overfitting": {"passed": True},
        },
        promotion_status=status,
        created_at_utc="2025-01-01T00:00:00Z",
    )


class MockGovernanceDecision:
    def __init__(self, qualified: bool = True, decision_fingerprint: str = "gov_fp_12345"):
        self.qualified = qualified
        self.decision_fingerprint = decision_fingerprint
        self.qualification_notes = "mock qualification"
        self.rejection_reasons = () if qualified else (RejectionReason.POOR_OUT_OF_SAMPLE,)


def test_A_governance_fingerprint_round_trip(tmp_path: Path):
    """Test A: Governance fingerprint round trip."""
    evidence = make_test_evidence()
    save_research_experiment(evidence, base_dir=tmp_path)

    gov_dec = MockGovernanceDecision(qualified=True, decision_fingerprint="gov_fp_test_999")
    cand_id = "cand_test_A"

    binding_path = persist_promoted_candidate_binding(
        candidate_id=cand_id,
        evidence=evidence,
        governance_decision=gov_dec,
        base_dir=tmp_path,
    )

    binding_data = load_candidate_binding(cand_id, base_dir=tmp_path)
    assert binding_data["governance_decision_fingerprint"] == "gov_fp_test_999"

    artifact = resolve_promoted_candidate(candidate_id=cand_id, base_dir=tmp_path)
    assert artifact is not None
    assert artifact.governance_decision_fingerprint == "gov_fp_test_999"


def test_B_campaign_selection_fingerprint_round_trip(tmp_path: Path):
    """Test B: Campaign selection fingerprint round trip."""
    evidence = make_test_evidence()
    save_research_experiment(evidence, base_dir=tmp_path)

    cand_id = "cand_test_B"
    sel_dec = ResearchCampaignSelectionDecision(
        campaign_id="camp_test_001",
        synthesis_fingerprint="synth_fp_123",
        selection_policy_fingerprint="policy_fp_123",
        selected_candidate_ids=(cand_id,),
        eligible_candidate_ids=(cand_id,),
        rejected_candidate_ids=(),
        blocked_candidate_ids=(),
        candidate_comparison_fingerprints=("comp_1",),
        selection_governance_fingerprints=(),
        qualification_fingerprints=("qual_1",),
        robustness_fingerprints=("rob_1",),
        evidence_fingerprints=(evidence.evidence_id,),
        decision_status=CampaignSelectionStatus.SELECTED.value,
        decision_reason="Selected top candidate",
        deterministic_ordering=(cand_id,),
    )

    persist_promoted_candidate_binding(
        candidate_id=cand_id,
        evidence=evidence,
        campaign_selection_decision=sel_dec,
        base_dir=tmp_path,
    )

    binding_data = load_candidate_binding(cand_id, base_dir=tmp_path)
    assert binding_data["campaign_selection_decision_fingerprint"] == sel_dec.decision_fingerprint

    artifact = resolve_promoted_candidate(candidate_id=cand_id, base_dir=tmp_path)
    assert artifact is not None
    assert artifact.campaign_selection_decision_fingerprint == sel_dec.decision_fingerprint


def test_C_wrong_candidate_rejected(tmp_path: Path):
    """Test C: Wrong candidate rejected when using SELECTED decision for non-selected candidate ID."""
    evidence = make_test_evidence()
    save_research_experiment(evidence, base_dir=tmp_path)

    cand_id = "cand_test_C_wrong"
    selected_id = "cand_test_C_winner"

    sel_dec = ResearchCampaignSelectionDecision(
        campaign_id="camp_test_002",
        synthesis_fingerprint="synth_fp_123",
        selection_policy_fingerprint="policy_fp_123",
        selected_candidate_ids=(selected_id,),
        eligible_candidate_ids=(selected_id, cand_id),
        rejected_candidate_ids=(),
        blocked_candidate_ids=(),
        candidate_comparison_fingerprints=("comp_1",),
        selection_governance_fingerprints=(),
        qualification_fingerprints=("qual_1",),
        robustness_fingerprints=("rob_1",),
        evidence_fingerprints=(evidence.evidence_id,),
        decision_status=CampaignSelectionStatus.SELECTED.value,
        decision_reason="Selected winner",
        deterministic_ordering=(selected_id, cand_id),
    )

    with pytest.raises((PromotionEligibilityError, PromotionIntegrityError)) as exc_info:
        persist_promoted_candidate_binding(
            candidate_id=cand_id,
            evidence=evidence,
            campaign_selection_decision=sel_dec,
            base_dir=tmp_path,
        )

    assert "selected_candidate_ids" in str(exc_info.value) or "not contained" in str(exc_info.value)

    # Verify no binding written
    binding_file = tmp_path / "by_candidate" / cand_id / "candidate.json"
    assert not binding_file.exists()


def test_D_non_selected_decision_rejected(tmp_path: Path):
    """Test D: Non-selected campaign decisions (NO_ELIGIBLE_CANDIDATE, TIE_UNRESOLVED) rejected."""
    evidence = make_test_evidence()
    save_research_experiment(evidence, base_dir=tmp_path)

    cand_id = "cand_test_D"

    no_eligible_dec = ResearchCampaignSelectionDecision(
        campaign_id="camp_test_003",
        synthesis_fingerprint="synth_fp_123",
        selection_policy_fingerprint="policy_fp_123",
        selected_candidate_ids=(),
        eligible_candidate_ids=(),
        rejected_candidate_ids=(cand_id,),
        blocked_candidate_ids=(),
        candidate_comparison_fingerprints=("comp_1",),
        selection_governance_fingerprints=(),
        qualification_fingerprints=(),
        robustness_fingerprints=(),
        evidence_fingerprints=(),
        decision_status=CampaignSelectionStatus.NO_ELIGIBLE_CANDIDATE.value,
        decision_reason="No eligible candidate",
        deterministic_ordering=(),
    )

    with pytest.raises(PromotionEligibilityError):
        persist_promoted_candidate_binding(
            candidate_id=cand_id,
            evidence=evidence,
            campaign_selection_decision=no_eligible_dec,
            base_dir=tmp_path,
        )

    tie_dec = ResearchCampaignSelectionDecision(
        campaign_id="camp_test_004",
        synthesis_fingerprint="synth_fp_123",
        selection_policy_fingerprint="policy_fp_123",
        selected_candidate_ids=(),
        eligible_candidate_ids=(cand_id, "cand_other"),
        rejected_candidate_ids=(),
        blocked_candidate_ids=(),
        candidate_comparison_fingerprints=("comp_1",),
        selection_governance_fingerprints=(),
        qualification_fingerprints=(),
        robustness_fingerprints=(),
        evidence_fingerprints=(),
        decision_status=CampaignSelectionStatus.TIE_UNRESOLVED.value,
        decision_reason="Unresolved tie",
        deterministic_ordering=(cand_id, "cand_other"),
    )

    with pytest.raises(PromotionEligibilityError):
        persist_promoted_candidate_binding(
            candidate_id=cand_id,
            evidence=evidence,
            campaign_selection_decision=tie_dec,
            base_dir=tmp_path,
        )


def test_E_conflicting_persisted_lineage(tmp_path: Path):
    """Test E: Conflicting persisted lineage raises exception and leaves original file unchanged."""
    evidence = make_test_evidence()
    save_research_experiment(evidence, base_dir=tmp_path)

    cand_id = "cand_test_E"
    gov_dec_1 = MockGovernanceDecision(qualified=True, decision_fingerprint="gov_fp_AAAA")
    gov_dec_2 = MockGovernanceDecision(qualified=True, decision_fingerprint="gov_fp_BBBB")

    binding_path = persist_promoted_candidate_binding(
        candidate_id=cand_id,
        evidence=evidence,
        governance_decision=gov_dec_1,
        base_dir=tmp_path,
    )

    original_content = binding_path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        persist_promoted_candidate_binding(
            candidate_id=cand_id,
            evidence=evidence,
            governance_decision=gov_dec_2,
            base_dir=tmp_path,
        )

    assert binding_path.read_text(encoding="utf-8") == original_content

    # Repeat for campaign-selection decision
    sel_dec_1 = ResearchCampaignSelectionDecision(
        campaign_id="camp_E",
        synthesis_fingerprint="synth_fp",
        selection_policy_fingerprint="pol_fp",
        selected_candidate_ids=("cand_E2",),
        eligible_candidate_ids=("cand_E2",),
        rejected_candidate_ids=(),
        blocked_candidate_ids=(),
        candidate_comparison_fingerprints=(),
        selection_governance_fingerprints=(),
        qualification_fingerprints=(),
        robustness_fingerprints=(),
        evidence_fingerprints=(),
        decision_status=CampaignSelectionStatus.SELECTED.value,
        decision_reason="Sel 1",
        deterministic_ordering=("cand_E2",),
    )

    sel_dec_2 = ResearchCampaignSelectionDecision(
        campaign_id="camp_E_other",
        synthesis_fingerprint="synth_fp_diff",
        selection_policy_fingerprint="pol_fp",
        selected_candidate_ids=("cand_E2",),
        eligible_candidate_ids=("cand_E2",),
        rejected_candidate_ids=(),
        blocked_candidate_ids=(),
        candidate_comparison_fingerprints=(),
        selection_governance_fingerprints=(),
        qualification_fingerprints=(),
        robustness_fingerprints=(),
        evidence_fingerprints=(),
        decision_status=CampaignSelectionStatus.SELECTED.value,
        decision_reason="Sel 2",
        deterministic_ordering=("cand_E2",),
    )

    binding_path_2 = persist_promoted_candidate_binding(
        candidate_id="cand_E2",
        evidence=evidence,
        campaign_selection_decision=sel_dec_1,
        base_dir=tmp_path,
    )

    original_content_2 = binding_path_2.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        persist_promoted_candidate_binding(
            candidate_id="cand_E2",
            evidence=evidence,
            campaign_selection_decision=sel_dec_2,
            base_dir=tmp_path,
        )

    assert binding_path_2.read_text(encoding="utf-8") == original_content_2


def test_F_artifact_fingerprint_changes_with_lineage():
    """Test F: PromotedCandidateArtifact.artifact_fingerprint changes when lineage fingerprints differ."""
    evidence = make_test_evidence()

    art1 = PromotedCandidateArtifact.from_persisted_research(
        candidate_id="cand_F",
        evidence=evidence,
        symbol="XAUUSD",
        timeframe="5m",
        governance_decision_fingerprint="gov_fp_1",
        campaign_selection_decision_fingerprint="cs_fp_1",
    )

    art2 = PromotedCandidateArtifact.from_persisted_research(
        candidate_id="cand_F",
        evidence=evidence,
        symbol="XAUUSD",
        timeframe="5m",
        governance_decision_fingerprint="gov_fp_2",
        campaign_selection_decision_fingerprint="cs_fp_1",
    )

    art3 = PromotedCandidateArtifact.from_persisted_research(
        candidate_id="cand_F",
        evidence=evidence,
        symbol="XAUUSD",
        timeframe="5m",
        governance_decision_fingerprint="gov_fp_1",
        campaign_selection_decision_fingerprint="cs_fp_2",
    )

    assert art1.artifact_fingerprint != art2.artifact_fingerprint
    assert art1.artifact_fingerprint != art3.artifact_fingerprint
    assert art2.artifact_fingerprint != art3.artifact_fingerprint


def test_G_idempotency(tmp_path: Path):
    """Test G: Idempotency when persisting identical binding twice."""
    evidence = make_test_evidence()
    save_research_experiment(evidence, base_dir=tmp_path)

    cand_id = "cand_test_G"
    gov_dec = MockGovernanceDecision(qualified=True, decision_fingerprint="gov_idempotent")
    sel_dec = ResearchCampaignSelectionDecision(
        campaign_id="camp_G",
        synthesis_fingerprint="synth_fp",
        selection_policy_fingerprint="pol_fp",
        selected_candidate_ids=(cand_id,),
        eligible_candidate_ids=(cand_id,),
        rejected_candidate_ids=(),
        blocked_candidate_ids=(),
        candidate_comparison_fingerprints=(),
        selection_governance_fingerprints=(),
        qualification_fingerprints=(),
        robustness_fingerprints=(),
        evidence_fingerprints=(),
        decision_status=CampaignSelectionStatus.SELECTED.value,
        decision_reason="Idempotent selection",
        deterministic_ordering=(cand_id,),
    )

    path1 = persist_promoted_candidate_binding(
        candidate_id=cand_id,
        evidence=evidence,
        governance_decision=gov_dec,
        campaign_selection_decision=sel_dec,
        base_dir=tmp_path,
    )

    path2 = persist_promoted_candidate_binding(
        candidate_id=cand_id,
        evidence=evidence,
        governance_decision=gov_dec,
        campaign_selection_decision=sel_dec,
        base_dir=tmp_path,
    )

    assert path1 == path2
    art1 = resolve_promoted_candidate(candidate_id=cand_id, base_dir=tmp_path)
    art2 = resolve_promoted_candidate(candidate_id=cand_id, base_dir=tmp_path)
    assert art1 is not None and art2 is not None
    assert art1.artifact_fingerprint == art2.artifact_fingerprint


def test_H_no_production_authority(tmp_path: Path):
    """Test H: Persistence layer does not create production authority or live execution state."""
    evidence = make_test_evidence()
    save_research_experiment(evidence, base_dir=tmp_path)

    cand_id = "cand_test_H"
    binding_path = persist_promoted_candidate_binding(
        candidate_id=cand_id,
        evidence=evidence,
        base_dir=tmp_path,
    )

    assert binding_path.exists()

    # Confirm reconstituted artifact is strictly research promotion artifact
    artifact = resolve_promoted_candidate(candidate_id=cand_id, base_dir=tmp_path)
    assert isinstance(artifact, PromotedCandidateArtifact)

    # Ensure no live execution, runtime authorization, or signal/publication objects exist
    assert not hasattr(artifact, "live_runtime")
    assert not hasattr(artifact, "runtime_authorization")
    assert not hasattr(artifact, "production_decision")
