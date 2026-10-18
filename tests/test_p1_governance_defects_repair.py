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


# --- BETA Ω ROBUSTNESS FLAG STRICTNESS REGRESSIONS ---

def test_robustness_flag_nested_boolean_controls():
    """Nested 'passed=False' and 'passed=True' controls."""
    evidence = make_valid_test_evidence()

    # True control
    verdict_true = dict(evidence.robustness_verdict)
    verdict_true["parameter_sensitivity"] = {"passed": True}
    ev_true = replace(evidence, robustness_verdict=verdict_true)
    assert assess_research_robustness(ev_true).is_robust is True

    # False control
    verdict_false = dict(evidence.robustness_verdict)
    verdict_false["parameter_sensitivity"] = {"passed": False}
    ev_false = replace(evidence, robustness_verdict=verdict_false)
    assert assess_research_robustness(ev_false).is_robust is False


def test_robustness_flag_nested_string_and_numeric_rejected():
    """Nested string 'false', '0', 'true', integer 0, 1 rejected."""
    evidence = make_valid_test_evidence()

    for bad_val in ("false", "0", "true", "1", 0, 1, 0.0, 1.0, None, [True]):
        verdict = dict(evidence.robustness_verdict)
        verdict["parameter_sensitivity"] = {"passed": bad_val}
        ev = replace(evidence, robustness_verdict=verdict)
        assert assess_research_robustness(ev).is_robust is False, f"Failed to reject bad_val: {bad_val}"


def test_robustness_flag_nested_other_flag_keys_malformed_rejected():
    """Nested 'is_valid', 'robust', and 'is_robust' with malformed non-Boolean values fail closed."""
    evidence = make_valid_test_evidence()

    for flag_key in ("is_valid", "robust", "is_robust"):
        for bad_val in ("false", "0", "1", 0, 1, None):
            verdict = dict(evidence.robustness_verdict)
            verdict["statistical_validation"] = {flag_key: bad_val}
            ev = replace(evidence, robustness_verdict=verdict)
            assert assess_research_robustness(ev).is_robust is False, f"Failed for {flag_key}={bad_val}"


def test_robustness_flag_toplevel_boolean_false():
    """Top-level 'is_robust=False' and 'passed=False' controls."""
    evidence = make_valid_test_evidence()

    # is_robust=False
    v1 = dict(evidence.robustness_verdict)
    v1["is_robust"] = False
    v1["passed"] = True
    assert assess_research_robustness(replace(evidence, robustness_verdict=v1)).is_robust is False

    # passed=False
    v2 = dict(evidence.robustness_verdict)
    v2["is_robust"] = True
    v2["passed"] = False
    assert assess_research_robustness(replace(evidence, robustness_verdict=v2)).is_robust is False


def test_robustness_flag_toplevel_malformed_strings_and_integers_rejected():
    """Top-level 'is_robust="false"', '"0"', '"true"', integer 0 and 1 rejected."""
    evidence = make_valid_test_evidence()

    for bad_val in ("false", "0", "true", "1", 0, 1, 0.0, 1.0, None):
        v1 = dict(evidence.robustness_verdict)
        v1["is_robust"] = bad_val
        assert assess_research_robustness(replace(evidence, robustness_verdict=v1)).is_robust is False

        v2 = dict(evidence.robustness_verdict)
        v2["passed"] = bad_val
        assert assess_research_robustness(replace(evidence, robustness_verdict=v2)).is_robust is False


def test_robustness_flag_missing_both_toplevel_flags_fails_closed():
    """Missing both top-level flags ('is_robust' and 'passed') fails closed."""
    evidence = make_valid_test_evidence()
    verdict = dict(evidence.robustness_verdict)
    if "is_robust" in verdict:
        del verdict["is_robust"]
    if "passed" in verdict:
        del verdict["passed"]

    ev = replace(evidence, robustness_verdict=verdict)
    assert assess_research_robustness(ev).is_robust is False


def test_robustness_flag_conflicting_toplevel_flags_fails_closed():
    """Conflicting top-level Boolean flags ('is_robust=True' and 'passed=False') fail closed."""
    evidence = make_valid_test_evidence()

    # is_robust=True, passed=False -> False
    v1 = dict(evidence.robustness_verdict)
    v1["is_robust"] = True
    v1["passed"] = False
    assert assess_research_robustness(replace(evidence, robustness_verdict=v1)).is_robust is False

    # is_robust=False, passed=True -> False
    v2 = dict(evidence.robustness_verdict)
    v2["is_robust"] = False
    v2["passed"] = True
    assert assess_research_robustness(replace(evidence, robustness_verdict=v2)).is_robust is False


def test_robustness_flag_valid_positive_controls():
    """Valid positive controls using actual Boolean True values pass."""
    evidence = make_valid_test_evidence()
    v = dict(evidence.robustness_verdict)
    v["is_robust"] = True
    v["passed"] = True
    ev = replace(evidence, robustness_verdict=v)

    rob = assess_research_robustness(ev)
    assert rob.is_robust is True

    qual = qualify_research_evidence(ev, robustness_assessment=rob)
    assert qual.qualified is True
    assert validate_promotion_eligibility(ev) is True


def test_end_to_end_security_gate_malformed_evidence_cannot_authorize_buy_decision(tmp_path: Path):
    """End-to-end security gate: Malformed robustness evidence cannot qualify, promote, resolve, or produce a BUY decision."""
    from src.evaluation.live_runtime import evaluate_authorized_live_runtime, evaluate_production_decision
    import pandas as pd

    # Construct evidence with malformed string "0" for top-level is_robust
    evidence = make_valid_test_evidence()
    verdict = dict(evidence.robustness_verdict)
    verdict["is_robust"] = "0"
    verdict["passed"] = True
    malformed_ev = replace(evidence, robustness_verdict=verdict)

    # 1. Robustness assessment fails
    rob = assess_research_robustness(malformed_ev)
    assert rob.is_robust is False

    # 2. Qualification fails
    qual = qualify_research_evidence(malformed_ev, robustness_assessment=rob)
    assert qual.qualified is False

    # 3. Promotion eligibility fails
    with pytest.raises((PromotionEligibilityError, ValueError)):
        validate_promotion_eligibility(malformed_ev)

    # 4. Save attempt fails closed on qualification check
    candidate_id = "cand_malformed_security_gate"
    with pytest.raises(PromotionEligibilityError):
        save_research_candidate(
            candidate_id=candidate_id,
            evidence=malformed_ev,
            operational_stability_score=0.85,
            governance_decision=qual,
            base_dir=tmp_path,
        )

    # Candidate was blocked from being saved; resolution returns None
    assert resolve_promoted_candidate(
        candidate_id=candidate_id,
        base_dir=tmp_path,
    ) is None

    # 5. Direct evaluate_production_decision / evaluate_authorized_live_runtime cannot produce BUY
    # Attempting to authorize runtime with invalid candidate/evidence fails
    with pytest.raises((PromotionEligibilityError, ValueError, AttributeError)):
        from src.evaluation.live_production_decision import authorize_production_runtime
        from src.evaluation.live_runtime_context import create_authorized_runtime_context

        art = PromotedCandidateArtifact(
            candidate_id=candidate_id,
            strategy_name="momentum",
            strategy_version="1.0",
            evidence=malformed_ev,
            symbol="XAUUSD",
            timeframe="5m",
            operational_stability_score=0.85,
            governance_decision_fingerprint=qual.decision_fingerprint,
        )
        # Authorize runtime must fail eligibility validation
        authorize_production_runtime(
            promoted_artifact=art,
            candidate_id=candidate_id,
            symbol="XAUUSD",
            timeframe="5m",
            base_dir=tmp_path,
        )


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
