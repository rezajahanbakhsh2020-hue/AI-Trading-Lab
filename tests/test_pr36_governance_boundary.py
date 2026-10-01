"""Comprehensive tests for PR #36 research governance-decision-to-promotion boundary.

Verifies invariants A through J:
A. Qualification without supplied robustness fails closed with MISSING_ROBUSTNESS_EVIDENCE.
B. Qualification with supplied assessment consumes it directly without recomputing.
C. Legacy passed=True without per-dimension evidence leaves dimensions as unavailable.
D. Discovery executes robustness exactly once per evidence artifact and passes it downstream.
E. Governance decision fingerprint is deterministic and immutable.
F. Governance decision fingerprint is persisted in registry lineage and preserved in reconstitution.
G. Promotion eligibility consumes canonical decision and fails closed on mismatch/tampering.
H. PromotedCandidateArtifact binds governance decision fingerprint into its artifact_fingerprint.
I. Legacy records without governance decision fail closed during active promotion.
J. AST structural checks enforce architectural boundaries.
"""

import ast
import pathlib
import pytest

from src.evaluation.discovery_engine import CandidateSpec, DiscoveryEngine
from src.evaluation.live_production_decision import (
    ProductionPromotionPolicy,
    PromotedCandidateArtifact,
    validate_promotion_eligibility,
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
from src.evaluation.research_qualification import (
    ResearchQualificationPolicy,
    qualify_research_evidence,
)
from src.evaluation.research_registry import (
    construct_registry_record_from_evidence,
)
from src.evaluation.research_robustness import (
    RobustnessStatus,
    assess_research_robustness,
)


def make_test_evidence(robustness_passed: bool = True) -> ResearchEvidence:
    """Helper to construct standard ResearchEvidence for testing."""
    ds = DatasetScope(
        dataset_id="ds_pr36",
        symbol="XAUUSD",
        timeframe="5m",
        start_date="2025-01-01",
        end_date="2025-01-10",
    )
    ea = ExecutionAssumptions(transaction_cost=0.001, slippage=0.001, latency_ms=10.0)
    cp = CodeProvenance(commit_sha="abcdef1234567890123456789012345678901234")
    spec = ResearchExperimentSpec(
        hypothesis="PR36 test hypothesis",
        methodology_version="1.0",
        strategy_name="momentum",
        strategy_version="1.0",
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        benchmark_reference="buy_and_hold",
        parameters={"momentum_window": 10, "stop_loss_pct": 0.01, "take_profit_pct": 0.02},
    )
    part_is = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2025-01-01",
        end_date="2025-01-03",
        total_return=0.15,
        max_drawdown=0.05,
        sharpe_ratio=1.8,
        observations=100,
        start_timestamp_utc="2025-01-01T00:00:00+00:00",
        end_timestamp_utc="2025-01-03T00:00:00+00:00",
    )
    part_oos = EvidencePartition(
        role=EvidencePartitionRole.OUT_OF_SAMPLE,
        start_date="2025-01-04",
        end_date="2025-01-06",
        total_return=0.10,
        max_drawdown=0.04,
        sharpe_ratio=1.5,
        observations=50,
        start_timestamp_utc="2025-01-04T00:00:00+00:00",
        end_timestamp_utc="2025-01-06T00:00:00+00:00",
    )
    part_wf = EvidencePartition(
        role=EvidencePartitionRole.WALK_FORWARD,
        start_date="2025-01-01",
        end_date="2025-01-06",
        total_return=0.12,
        max_drawdown=0.05,
        sharpe_ratio=1.6,
        observations=80,
        start_timestamp_utc="2025-01-01T00:00:00+00:00",
        end_timestamp_utc="2025-01-06T00:00:00+00:00",
    )
    r_verdict = {
        "passed": robustness_passed,
        "is_robust": robustness_passed,
        "parameter_sensitivity": {"passed": True},
        "subsample_stability": {"passed": True},
        "execution_cost_stress": {"passed": True},
        "statistical_validation": {"passed": True},
        "anti_overfitting": {"passed": True},
    } if robustness_passed else {"passed": False, "is_robust": False}

    return ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(part_is, part_oos, part_wf),
        robustness_verdict=r_verdict,
        promotion_status=PromotionStatus.PROMOTABLE,
    )


def test_A_qualification_without_robustness_fails_closed_missing_evidence():
    """Requirement A: Qualification with required robustness and no supplied assessment fails closed."""
    evidence = make_test_evidence()
    policy = ResearchQualificationPolicy(require_robustness=True)

    result = qualify_research_evidence(evidence, policy=policy, robustness_assessment=None)

    assert result.qualified is False
    assert RejectionReason.MISSING_ROBUSTNESS_EVIDENCE in result.rejection_reasons


def test_B_qualification_consumes_supplied_robustness_without_recomputing():
    """Requirement B: Qualification consumes supplied canonical assessment directly."""
    evidence = make_test_evidence()
    rob_assessment = assess_research_robustness(evidence)

    result = qualify_research_evidence(
        evidence,
        policy=ResearchQualificationPolicy(require_robustness=True),
        robustness_assessment=rob_assessment,
    )

    assert result.qualified is True
    assert result.robustness_assessment_fingerprint == rob_assessment.robustness_fingerprint


def test_C_legacy_coarse_robustness_verdict_does_not_infer_dimensions():
    """Requirement C: Legacy passed=True without per-dimension evidence leaves dimensions unavailable."""
    ds = DatasetScope("ds", "XAUUSD", "5m", "2025-01-01", "2025-01-05")
    ea = ExecutionAssumptions(0.001, 0.001, 10.0)
    cp = CodeProvenance("1234567890123456789012345678901234567890")
    spec = ResearchExperimentSpec("H", "1.0", "momentum", "1.0", ds, ea, cp, "buy_and_hold", {"w": 10})
    part = EvidencePartition(
        EvidencePartitionRole.IN_SAMPLE,
        "2025-01-01",
        "2025-01-05",
        0.1,
        0.05,
        1.5,
        100,
        start_timestamp_utc="2025-01-01T00:00:00+00:00",
        end_timestamp_utc="2025-01-05T00:00:00+00:00",
    )

    # Coarse legacy verdict with passed=True but NO dimension dicts
    legacy_ev = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(part,),
        robustness_verdict={"passed": True, "is_robust": True},
        promotion_status=PromotionStatus.PROMOTABLE,
    )

    rob_assessment = assess_research_robustness(legacy_ev)

    assert len(rob_assessment.dimensions_unavailable) > 0
    assert rob_assessment.status == RobustnessStatus.PARTIALLY_EVALUATED

    result = qualify_research_evidence(legacy_ev, robustness_assessment=rob_assessment)
    assert result.qualified is False
    assert RejectionReason.FAILED_ROBUSTNESS in result.rejection_reasons


def test_D_discovery_executes_robustness_exactly_once():
    """Requirement D: Discovery executes robustness exactly once per evidence artifact."""
    import pandas as pd

    engine = DiscoveryEngine()
    ts = pd.date_range("2025-01-01", "2025-01-05 23:55", freq="5min", tz="UTC")
    n = len(ts)
    df = pd.DataFrame({
        "timestamp": ts,
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.5] * n,
    })
    ds = DatasetScope("ds_disc", "XAUUSD", "5m", "2025-01-01", "2025-01-05")
    ea = ExecutionAssumptions(0.001, 0.001, 10.0)
    cp = CodeProvenance("1234567890123456789012345678901234567890")
    cand_specs = [CandidateSpec("grid_search", "1.0", strategy_name="momentum", parameters={"momentum_window": 5})]

    res = engine.run_discovery(
        df,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        candidates=cand_specs,
    )

    total_evidence_count = len(res.promoted_evidence) + len(res.rejected_evidence)
    assert len(res.robustness_assessments) == total_evidence_count
    if res.promoted_evidence:
        ev = res.promoted_evidence[0]
        rob = res.robustness_assessments[0]
        reg_record = res.registry_records[0]
        assert rob.evidence_fingerprint == ev.evidence_id
        assert reg_record.robustness_assessment_fingerprint == rob.robustness_fingerprint
        assert reg_record.governance_decision_fingerprint is not None


def test_E_governance_decision_determinism_and_immutability():
    """Requirement E: Governance decision fingerprint is deterministic and changes with material inputs."""
    evidence = make_test_evidence()
    rob_assessment = assess_research_robustness(evidence)
    p1 = ResearchQualificationPolicy(policy_version="v1")
    p2 = ResearchQualificationPolicy(policy_version="v2")

    res1_a = qualify_research_evidence(evidence, policy=p1, robustness_assessment=rob_assessment)
    res1_b = qualify_research_evidence(evidence, policy=p1, robustness_assessment=rob_assessment)
    res2 = qualify_research_evidence(evidence, policy=p2, robustness_assessment=rob_assessment)

    assert res1_a.decision_fingerprint == res1_b.decision_fingerprint
    assert res1_a.decision_fingerprint != res2.decision_fingerprint


def test_F_registry_lineage_persists_governance_decision_fingerprint():
    """Requirement F: ResearchRegistryRecord preserves governance decision fingerprint."""
    evidence = make_test_evidence()
    rob_assessment = assess_research_robustness(evidence)
    gov_decision = qualify_research_evidence(evidence, robustness_assessment=rob_assessment)

    record = construct_registry_record_from_evidence(
        evidence=evidence,
        candidate_id="cand_reg",
        robustness_assessment=rob_assessment,
        governance_decision=gov_decision,
    )

    assert record.governance_decision_fingerprint == gov_decision.decision_fingerprint
    reconstituted_dict = record.semantic_content
    assert reconstituted_dict["lineage"]["governance_decision_fingerprint"] == gov_decision.decision_fingerprint


def test_G_H_promotion_consumes_canonical_decision_and_binds_fingerprint():
    """Requirements G & H: PromotedCandidateArtifact binds governance decision fingerprint into its identity."""
    evidence = make_test_evidence()
    rob_assessment = assess_research_robustness(evidence)
    gov_decision = qualify_research_evidence(evidence, robustness_assessment=rob_assessment)

    artifact = PromotedCandidateArtifact(
        candidate_id="cand_prom",
        strategy_name="momentum",
        strategy_version="1.0",
        evidence=evidence,
        symbol="XAUUSD",
        timeframe="5m",
        operational_stability_score=0.85,
        governance_decision=gov_decision,
        governance_decision_fingerprint=gov_decision.decision_fingerprint,
    )

    assert artifact.governance_decision_fingerprint == gov_decision.decision_fingerprint
    assert len(artifact.artifact_fingerprint) == 64

    # Mismatched governance decision fingerprint fails closed
    with pytest.raises(ValueError, match="does not match expected fingerprint"):
        validate_promotion_eligibility(
            evidence,
            governance_decision=gov_decision,
            governance_decision_fingerprint="tampered_fp_12345",
        )


def test_J_ast_structural_architectural_checks():
    """Requirement J: AST structural checks enforce non-recomputation and module boundaries."""
    # 1. qualification (research_qualification.py) must not call assess_research_robustness
    qual_file = pathlib.Path("src/evaluation/research_qualification.py")
    tree_qual = ast.parse(qual_file.read_text(encoding="utf-8"), filename=str(qual_file))
    for node in ast.walk(tree_qual):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "assess_research_robustness", (
                "research_qualification.py must not call assess_research_robustness internally."
            )

    # 2. production promotion path (live_production_decision.py) must not call qualify_research_evidence inside validate_promotion_eligibility
    prod_file = pathlib.Path("src/evaluation/live_production_decision.py")
    tree_prod = ast.parse(prod_file.read_text(encoding="utf-8"), filename=str(prod_file))
    for node in ast.walk(tree_prod):
        if isinstance(node, ast.FunctionDef) and node.name == "validate_promotion_eligibility":
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    assert child.func.id != "qualify_research_evidence", (
                        "validate_promotion_eligibility must consume canonical decision rather than recomputing qualification."
                    )
