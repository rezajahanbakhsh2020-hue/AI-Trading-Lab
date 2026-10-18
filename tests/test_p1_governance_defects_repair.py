"""Comprehensive regression test suite for Beta Ω - P1 Governance Integrity Repair.

Covers:
- Defect A: Statistical Sample Sufficiency Enforcement
- Defect B: Nested Robustness Dimension Failure and Schema Validation
- Defect C: Governance Fingerprint Tamper-Resistance and Resolution Security
- Canonical Governance Path Unification
- Seven-Timeframe Independence and Operational Stability Invariants
"""

import json
from dataclasses import replace
from pathlib import Path
import pytest

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
from src.evaluation.research_robustness import (
    RobustnessStatus,
    assess_research_robustness,
)
from src.evaluation.research_store import (
    PromotionEligibilityError,
    PromotionIntegrityError,
    resolve_promoted_candidate,
    save_research_candidate,
)


def make_valid_test_evidence(
    total_observations: int = 230,
    robustness_passed: bool = True,
    created_at_utc: str = "2025-01-01T00:00:00+00:00",
) -> ResearchEvidence:
    """Construct a genuinely valid ResearchEvidence positive control artifact."""
    ds = DatasetScope(
        dataset_id="ds_repair_test",
        symbol="XAUUSD",
        timeframe="5m",
        start_date="2025-01-01",
        end_date="2025-01-10",
    )
    ea = ExecutionAssumptions(transaction_cost=0.001, slippage=0.001, latency_ms=10.0)
    cp = CodeProvenance(commit_sha="1234567890123456789012345678901234567890")
    spec = ResearchExperimentSpec(
        hypothesis="Governance repair test hypothesis",
        methodology_version="1.0",
        strategy_name="momentum",
        strategy_version="1.0",
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        benchmark_reference="buy_and_hold",
        parameters={"momentum_window": 10, "stop_loss_pct": 0.01, "take_profit_pct": 0.02},
    )

    obs_is = int(total_observations * 0.5)
    obs_oos = int(total_observations * 0.2)
    obs_wf = total_observations - obs_is - obs_oos

    part_is = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2025-01-01",
        end_date="2025-01-03",
        total_return=0.15,
        max_drawdown=0.05,
        sharpe_ratio=1.8,
        observations=obs_is,
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
        observations=obs_oos,
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
        observations=obs_wf,
        start_timestamp_utc="2025-01-01T00:00:00+00:00",
        end_timestamp_utc="2025-01-06T00:00:00+00:00",
    )

    r_verdict = {
        "passed": robustness_passed,
        "is_robust": robustness_passed,
        "parameter_sensitivity": {"passed": True, "pass_rate": 1.0},
        "subsample_stability": {"passed": True, "pass_rate": 1.0},
        "execution_cost_stress": {"passed": True, "pass_rate": 1.0},
        "statistical_validation": {"passed": True, "is_valid": True, "t_statistic": 2.5, "p_value": 0.01},
        "anti_overfitting": {"passed": True},
    } if robustness_passed else {"passed": False, "is_robust": False}

    return ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(part_is, part_oos, part_wf),
        robustness_verdict=r_verdict,
        promotion_status=PromotionStatus.PROMOTABLE,
        created_at_utc=created_at_utc,
    )


# --- DEFECT A REGRESSIONS ---

def test_defect_a_low_observations_fails_qualification_and_promotion_eligibility():
    """Defect A: Total observations = 3 (below default 30 minimum) must fail qualification and eligibility."""
    evidence = make_valid_test_evidence(total_observations=3)

    assert sum(p.observations for p in evidence.partitions) == 3

    robustness = assess_research_robustness(evidence)
    qualification = qualify_research_evidence(
        evidence,
        robustness_assessment=robustness,
    )

    assert qualification.qualified is False
    assert RejectionReason.INSUFFICIENT_STATISTICAL_SAMPLE in qualification.rejection_reasons

    with pytest.raises((PromotionEligibilityError, ValueError)):
        validate_promotion_eligibility(evidence)


def test_defect_a_observation_counts_matrix():
    """Defect A: Test below-minimum (29), exactly-at-minimum (30), and above-minimum (31) observations."""
    ev_29 = make_valid_test_evidence(total_observations=29)
    ev_30 = make_valid_test_evidence(total_observations=30)
    ev_31 = make_valid_test_evidence(total_observations=31)

    # 29 observations -> Reject
    qual_29 = qualify_research_evidence(ev_29, robustness_assessment=assess_research_robustness(ev_29))
    assert qual_29.qualified is False
    assert RejectionReason.INSUFFICIENT_STATISTICAL_SAMPLE in qual_29.rejection_reasons
    with pytest.raises((PromotionEligibilityError, ValueError)):
        validate_promotion_eligibility(ev_29)

    # 30 observations -> Pass
    qual_30 = qualify_research_evidence(ev_30, robustness_assessment=assess_research_robustness(ev_30))
    assert qual_30.qualified is True
    assert validate_promotion_eligibility(ev_30) is True

    # 31 observations -> Pass
    qual_31 = qualify_research_evidence(ev_31, robustness_assessment=assess_research_robustness(ev_31))
    assert qual_31.qualified is True
    assert validate_promotion_eligibility(ev_31) is True


def test_defect_a_policy_override_supported():
    """Defect A: Custom policy threshold (e.g. min_statistical_observations=10) applies consistently."""
    evidence = make_valid_test_evidence(total_observations=15)

    custom_qual_policy = ResearchQualificationPolicy(min_statistical_observations=10)
    custom_prod_policy = ProductionPromotionPolicy()

    # Default policy (30) fails
    qual_default = qualify_research_evidence(evidence, robustness_assessment=assess_research_robustness(evidence))
    assert qual_default.qualified is False

    # Custom qualification policy (10) passes
    qual_custom = qualify_research_evidence(
        evidence,
        policy=custom_qual_policy,
        robustness_assessment=assess_research_robustness(evidence),
    )
    assert qual_custom.qualified is True

    # Eligibility with custom qualification policy passes
    assert validate_promotion_eligibility(evidence, policy=custom_qual_policy) is True


# --- DEFECT B REGRESSIONS ---

def test_defect_b_failed_nested_dimension_overrides_top_level_pass():
    """Defect B: Explicit parameter_sensitivity={"passed": False} overrides top-level passed=True."""
    evidence = make_valid_test_evidence()
    verdict = dict(evidence.robustness_verdict)
    verdict["passed"] = True
    verdict["is_robust"] = True
    verdict["parameter_sensitivity"] = {"passed": False, "pass_rate": 0.2}

    bad_evidence = replace(evidence, robustness_verdict=verdict)

    robustness = assess_research_robustness(bad_evidence)
    qualification = qualify_research_evidence(
        bad_evidence,
        robustness_assessment=robustness,
    )

    assert robustness.is_robust is False
    assert qualification.qualified is False

    with pytest.raises((PromotionEligibilityError, ValueError)):
        validate_promotion_eligibility(bad_evidence)


def test_defect_b_missing_dimension_fails_robustness_and_qualification():
    """Defect B: Missing required dimension leads to PARTIALLY_EVALUATED and fails qualification."""
    evidence = make_valid_test_evidence()
    verdict = dict(evidence.robustness_verdict)
    del verdict["parameter_sensitivity"]

    bad_evidence = replace(evidence, robustness_verdict=verdict)

    robustness = assess_research_robustness(bad_evidence)
    qualification = qualify_research_evidence(
        bad_evidence,
        robustness_assessment=robustness,
    )

    assert robustness.status == RobustnessStatus.PARTIALLY_EVALUATED
    assert robustness.is_robust is False
    assert qualification.qualified is False
    assert RejectionReason.FAILED_ROBUSTNESS in qualification.rejection_reasons

    with pytest.raises((PromotionEligibilityError, ValueError)):
        validate_promotion_eligibility(bad_evidence)


def test_defect_b_non_finite_numeric_in_dimension_fails():
    """Defect B: Non-finite numeric metric (e.g. t_statistic=NaN) in dimension payload fails robustness."""
    evidence = make_valid_test_evidence()
    verdict = dict(evidence.robustness_verdict)
    verdict["statistical_validation"] = {
        "passed": True,
        "is_valid": True,
        "t_statistic": float("nan"),
    }

    bad_evidence = replace(evidence, robustness_verdict=verdict)

    robustness = assess_research_robustness(bad_evidence)
    assert robustness.is_robust is False

    qualification = qualify_research_evidence(
        bad_evidence,
        robustness_assessment=robustness,
    )
    assert qualification.qualified is False

    with pytest.raises((PromotionEligibilityError, ValueError)):
        validate_promotion_eligibility(bad_evidence)


def test_defect_b_malformed_dimension_payload_fails():
    """Defect B: Malformed string or empty dict payload for a dimension fails robustness."""
    evidence = make_valid_test_evidence()

    # Case 1: string payload
    verdict1 = dict(evidence.robustness_verdict)
    verdict1["parameter_sensitivity"] = "invalid_payload_string"
    bad_ev1 = replace(evidence, robustness_verdict=verdict1)
    rob1 = assess_research_robustness(bad_ev1)
    assert rob1.is_robust is False

    # Case 2: empty dict payload
    verdict2 = dict(evidence.robustness_verdict)
    verdict2["parameter_sensitivity"] = {}
    bad_ev2 = replace(evidence, robustness_verdict=verdict2)
    rob2 = assess_research_robustness(bad_ev2)
    assert rob2.is_robust is False


# --- DEFECT C REGRESSIONS ---

def test_defect_c_tampered_governance_fingerprint_in_binding_rejected(tmp_path: Path):
    """Defect C: Resolver and direct eligibility validation reject tampered governance fingerprint."""
    evidence = make_valid_test_evidence()
    rob = assess_research_robustness(evidence)
    qualification = qualify_research_evidence(evidence, robustness_assessment=rob)
    candidate_id = "cand_defect_c"

    save_research_candidate(
        candidate_id=candidate_id,
        evidence=evidence,
        operational_stability_score=0.85,
        governance_decision=qualification,
        base_dir=tmp_path,
    )

    binding_path = tmp_path / "by_candidate" / candidate_id / "candidate.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))

    # Mutate governance decision fingerprint
    binding["governance_decision_fingerprint"] = "tampered_governance_fingerprint"
    binding_path.write_text(json.dumps(binding, indent=2), encoding="utf-8")

    # 1. Resolver must reject
    with pytest.raises(PromotionEligibilityError):
        resolve_promoted_candidate(
            candidate_id=candidate_id,
            base_dir=tmp_path,
        )

    # 2. Direct validation must reject
    with pytest.raises(PromotionEligibilityError):
        validate_promotion_eligibility(
            evidence,
            governance_decision_fingerprint="tampered_governance_fingerprint",
        )


def test_defect_c_unmodified_binding_positive_control(tmp_path: Path):
    """Defect C Positive Control: Legitimate untampered candidate resolves cleanly."""
    evidence = make_valid_test_evidence()
    rob = assess_research_robustness(evidence)
    qualification = qualify_research_evidence(evidence, robustness_assessment=rob)
    candidate_id = "cand_pos_control"

    save_research_candidate(
        candidate_id=candidate_id,
        evidence=evidence,
        operational_stability_score=0.85,
        governance_decision=qualification,
        base_dir=tmp_path,
    )

    resolved = resolve_promoted_candidate(
        candidate_id=candidate_id,
        base_dir=tmp_path,
    )

    assert resolved is not None
    assert resolved.candidate_id == candidate_id
    assert resolved.governance_decision_fingerprint == qualification.decision_fingerprint


def test_defect_c_missing_fingerprint_in_binding_rejected(tmp_path: Path):
    """Defect C: Missing or empty governance_decision_fingerprint in binding fails resolution."""
    evidence = make_valid_test_evidence()
    rob = assess_research_robustness(evidence)
    qualification = qualify_research_evidence(evidence, robustness_assessment=rob)
    candidate_id = "cand_missing_fp"

    save_research_candidate(
        candidate_id=candidate_id,
        evidence=evidence,
        operational_stability_score=0.85,
        governance_decision=qualification,
        base_dir=tmp_path,
    )

    binding_path = tmp_path / "by_candidate" / candidate_id / "candidate.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))

    del binding["governance_decision_fingerprint"]
    binding_path.write_text(json.dumps(binding, indent=2), encoding="utf-8")

    with pytest.raises(PromotionEligibilityError):
        resolve_promoted_candidate(
            candidate_id=candidate_id,
            base_dir=tmp_path,
        )


# --- SEVEN-TIMEFRAME INDEPENDENCE & OPERATIONAL STABILITY INVARIANTS ---

def test_seven_timeframe_independence():
    """Requirement 7: Canonical seven timeframes remain strictly independent."""
    canonical_frames = ("1m", "5m", "15m", "30m", "1H", "4H", "1D")
    assert len(canonical_frames) == 7

    evidence = make_valid_test_evidence()
    artifact = PromotedCandidateArtifact(
        candidate_id="cand_5m",
        strategy_name="momentum",
        strategy_version="1.0",
        evidence=evidence,
        symbol="XAUUSD",
        timeframe="5m",
        operational_stability_score=0.85,
    )

    # 5m candidate passed for 1D requested timeframe raises mismatch
    with pytest.raises(ValueError, match="timeframe"):
        from src.evaluation.live_production_decision import validate_production_scope
        validate_production_scope(artifact, current_timeframe="1D")


def test_invalid_operational_stability_score_rejected():
    """Requirement 7: operational_stability_score must be finite, numeric, and non-boolean."""
    evidence = make_valid_test_evidence()

    for invalid_score in (float("nan"), float("inf"), True, False, "0.85", None):
        with pytest.raises((ValueError, TypeError)):
            PromotedCandidateArtifact(
                candidate_id="cand_invalid_score",
                strategy_name="momentum",
                strategy_version="1.0",
                evidence=evidence,
                symbol="XAUUSD",
                timeframe="5m",
                operational_stability_score=invalid_score,
            )


# --- EXISTING LIVE EVIDENCE REMAINS BLOCKED ---

def test_existing_invalid_live_evidence_remains_blocked():
    """Requirement 6: Existing invalid 5m and 1D evidence remains BLOCKED."""
    from src.evaluation.live_execution_runtime import verify_timeframe_production_readiness

    res_5m = verify_timeframe_production_readiness(
        symbol="XAUUSD",
        timeframe="5m",
        candidate_id="cand_moving_average_5m",
    )
    assert res_5m["status"] == "BLOCKED"

    res_1D = verify_timeframe_production_readiness(
        symbol="XAUUSD",
        timeframe="1D",
        candidate_id="cand_moving_average_1d",
    )
    assert res_1D["status"] == "BLOCKED"
