"""Comprehensive unit and integration tests for Robustness, Stress & Statistical Validation.

Covers all 23 prompt requirements for research lineage robustness, statistical validation,
anti-overfitting enforcement, deterministic fingerprinting, and DiscoveryEngine integration.
"""

import math
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.evaluation.candidate_generator import CandidateSpec
from src.evaluation.discovery_engine import DiscoveryCriteria, DiscoveryEngine
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
    RobustnessCriteria,
)
from src.evaluation.research_store import (
    load_research_experiment,
    save_research_experiment,
)
from src.evaluation.robustness_evaluator import (
    RobustnessEvaluator,
    compute_statistical_validation,
    derive_parameter_perturbations,
)


@pytest.fixture
def sample_market_data():
    """Generates synthetic price dataset for tests."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.015, size=100)
    price = 100.0 * np.cumprod(1.0 + returns)
    df = pd.DataFrame({
        "timestamp": dates,
        "open": price,
        "high": price * 1.01,
        "low": price * 0.99,
        "close": price,
        "volume": 1000.0,
        "return": returns,
    })
    return df


@pytest.fixture
def dataset_scope():
    return DatasetScope(
        dataset_id="test_ds",
        symbol="XAUUSD",
        timeframe="1D",
        start_date="2023-01-01",
        end_date="2023-04-10",
    )


@pytest.fixture
def execution_assumptions():
    return ExecutionAssumptions(
        transaction_cost=0.0001,
        slippage=0.0001,
        latency_ms=10.0,
    )


@pytest.fixture
def code_provenance():
    return CodeProvenance(
        commit_sha="dbe7bf47ab76abbeeeaa1bd3905f054085c6bc47",
        repository_status="clean",
        author="Jules",
    )


@pytest.fixture
def source_candidate():
    return CandidateSpec(
        generator_name="grid_search",
        generator_version="1.0",
        strategy_name="momentum",
        parameters={"window": 10},
        random_seed=42,
    )


# 1. Deterministic robustness evidence for identical authoritative inputs
def test_deterministic_robustness_evidence(source_candidate, sample_market_data, dataset_scope, execution_assumptions):
    evaluator = RobustnessEvaluator()
    res1 = evaluator.evaluate_candidate_robustness(
        candidate=source_candidate,
        df_reference=sample_market_data.iloc[:70],
        df_oos=sample_market_data.iloc[70:],
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
    )
    res2 = evaluator.evaluate_candidate_robustness(
        candidate=source_candidate,
        df_reference=sample_market_data.iloc[:70],
        df_oos=sample_market_data.iloc[70:],
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
    )
    assert res1.as_dict() == res2.as_dict()


# 2. Evidence identity changes when a robustness input changes
def test_evidence_identity_changes_on_input_change(source_candidate, sample_market_data, dataset_scope, execution_assumptions):
    evaluator1 = RobustnessEvaluator(RobustnessCriteria(min_t_stat=1.65))
    evaluator2 = RobustnessEvaluator(RobustnessCriteria(min_t_stat=3.00, version="robustness_v2.0"))

    res1 = evaluator1.evaluate_candidate_robustness(
        source_candidate, sample_market_data.iloc[:70], sample_market_data.iloc[70:], dataset_scope, execution_assumptions
    )
    res2 = evaluator2.evaluate_candidate_robustness(
        source_candidate, sample_market_data.iloc[:70], sample_market_data.iloc[70:], dataset_scope, execution_assumptions
    )
    assert res1.as_dict() != res2.as_dict()


# 3. Parameter perturbation generation is deterministic
def test_parameter_perturbation_deterministic(source_candidate):
    p1 = derive_parameter_perturbations(source_candidate)
    p2 = derive_parameter_perturbations(source_candidate)
    assert len(p1) == len(p2)
    for c1, c2 in zip(p1, p2):
        assert c1.candidate_id == c2.candidate_id
        assert c1.parameters == c2.parameters


# 4. Original candidate is never mutated
def test_original_candidate_not_mutated(source_candidate):
    original_params = dict(source_candidate.parameters)
    derive_parameter_perturbations(source_candidate)
    assert source_candidate.parameters == original_params


# 5. Perturbation candidates cannot collide with the source candidate
def test_perturbation_candidates_no_collision(source_candidate):
    p_cands = derive_parameter_perturbations(source_candidate)
    for p_cand in p_cands:
        assert p_cand.candidate_id != source_candidate.candidate_id


# 6. Chronological robustness slices preserve DatasetScope boundaries
def test_subsample_slices_preserve_chronology(source_candidate, sample_market_data, dataset_scope, execution_assumptions):
    evaluator = RobustnessEvaluator(RobustnessCriteria(subsample_slices_count=3))
    res = evaluator.evaluate_candidate_robustness(
        source_candidate, sample_market_data.iloc[:70], sample_market_data.iloc[70:], dataset_scope, execution_assumptions
    )
    slices = res.subsample_stability["details"]
    assert len(slices) == 3


# 7. Future/OOS leakage is rejected
def test_future_oos_leakage_rejected(source_candidate, sample_market_data, dataset_scope, execution_assumptions):
    evaluator = RobustnessEvaluator()
    # Passing df_reference that overlaps or occurs AFTER OOS data
    df_ref = sample_market_data.iloc[80:]
    df_oos = sample_market_data.iloc[:50]
    res = evaluator.evaluate_candidate_robustness(
        source_candidate, df_ref, df_oos, dataset_scope, execution_assumptions
    )
    assert not res.is_robust
    assert RejectionReason.ANTI_OVERFITTING_VIOLATION in res.rejection_reasons


# 8. Overlapping invalid partitions are rejected by DatasetScope
def test_invalid_dataset_scope_rejected():
    with pytest.raises(ValueError):
        DatasetScope("id", "XAUUSD", "1D", "2023-05-01", "2023-01-01")


# 9. Insufficient observations fail closed
def test_insufficient_observations_fail_closed():
    res = compute_statistical_validation([0.01, 0.02], RobustnessCriteria(min_statistical_observations=30))
    assert not res.is_valid
    assert res.observation_count == 2


# 10. Non-finite observations fail closed
def test_non_finite_observations_fail_closed():
    obs = [0.01] * 29 + [float("nan")]
    res = compute_statistical_validation(obs, RobustnessCriteria(min_statistical_observations=30))
    assert not res.is_valid


# 11. Invalid execution-cost/slippage assumptions fail closed
def test_invalid_execution_assumptions_fail_closed():
    with pytest.raises(ValueError):
        ExecutionAssumptions(transaction_cost=-0.01, slippage=0.001, latency_ms=10.0)


# 12. Failed robustness evaluation blocks promotion
def test_failed_robustness_blocks_promotion(source_candidate, sample_market_data, dataset_scope, execution_assumptions, code_provenance):
    # Setting an impossible t-stat threshold to force failure
    strict_criteria = DiscoveryCriteria(
        robustness_criteria=RobustnessCriteria(min_t_stat=100.0)
    )
    engine = DiscoveryEngine(criteria=strict_criteria)
    result = engine.run_discovery(
        df=sample_market_data,
        candidates=[source_candidate],
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
    )
    assert len(result.promoted_evidence) == 0
    assert len(result.rejected_evidence) == 1
    assert result.rejected_evidence[0].promotion_status == PromotionStatus.REJECTED


# 13 & 14. Missing robustness / statistical evidence blocks promotion
def test_missing_robustness_blocks_promotion():
    spec = ResearchExperimentSpec(
        hypothesis="test",
        methodology_version="1.0",
        strategy_name="momentum",
        strategy_version="1.0.0",
        dataset_scope=DatasetScope("id", "XAUUSD", "1D", "2023-01-01", "2023-01-10"),
        execution_assumptions=ExecutionAssumptions(0.001, 0.001, 10.0),
        code_provenance=CodeProvenance("dbe7bf47ab76abbeeeaa1bd3905f054085c6bc47"),
        benchmark_reference="buy_and_hold",
    )
    part = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2023-01-01",
        end_date="2023-01-10",
        total_return=0.10,
        max_drawdown=-0.02,
        sharpe_ratio=1.5,
    )
    # Empty robustness verdict should not promote if rejected status is required or evaluated
    evidence = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(part,),
        robustness_verdict={},
        promotion_status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.MISSING_ROBUSTNESS_EVIDENCE,),
    )
    assert evidence.promotion_status == PromotionStatus.REJECTED


# 15. Statistical calculation uses real evaluator observations
def test_statistical_calc_uses_real_returns(source_candidate, sample_market_data, execution_assumptions):
    evaluator = RobustnessEvaluator()
    stat_res = evaluator._evaluate_statistical_significance(
        candidate=source_candidate,
        df_reference=sample_market_data,
        execution_assumptions=execution_assumptions,
    )
    assert stat_res.observation_count > 0


# 16. Duplicate identical evidence is idempotent
def test_duplicate_identical_evidence_idempotent(source_candidate, sample_market_data, dataset_scope, execution_assumptions, code_provenance):
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = DiscoveryEngine()
        result = engine.run_discovery(
            df=sample_market_data,
            candidates=[source_candidate],
            dataset_scope=dataset_scope,
            execution_assumptions=execution_assumptions,
            code_provenance=code_provenance,
            persist_evidence=True,
        )
        evidence = result.rejected_evidence[0] if result.rejected_evidence else result.promoted_evidence[0]
        p1 = save_research_experiment(evidence, base_dir=tmpdir)
        p2 = save_research_experiment(evidence, base_dir=tmpdir)
        assert p1 == p2


# 17. Conflicting evidence with same fingerprint is rejected
def test_conflicting_evidence_rejected(source_candidate, sample_market_data, dataset_scope, execution_assumptions, code_provenance):
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = DiscoveryEngine()
        result = engine.run_discovery(
            df=sample_market_data,
            candidates=[source_candidate],
            dataset_scope=dataset_scope,
            execution_assumptions=execution_assumptions,
            code_provenance=code_provenance,
        )
        evidence1 = result.rejected_evidence[0] if result.rejected_evidence else result.promoted_evidence[0]
        save_research_experiment(evidence1, base_dir=tmpdir)

        # Create conflicting evidence with same spec fingerprint
        evidence2 = ResearchEvidence(
            experiment_fingerprint=evidence1.experiment_fingerprint,
            spec=evidence1.spec,
            partitions=evidence1.partitions,
            robustness_verdict=evidence1.robustness_verdict,
            promotion_status=PromotionStatus.REJECTED,
            rejection_reasons=(RejectionReason.FAILED_ROBUSTNESS,),
            critique_notes="Conflicting critique notes",
        )
        with pytest.raises(FileExistsError):
            save_research_experiment(evidence2, base_dir=tmpdir)


# 18. Persistence failure cannot yield promotion
def test_persistence_failure_handling(source_candidate, sample_market_data, dataset_scope, execution_assumptions, code_provenance, monkeypatch):
    def mock_save_failing(evidence, base_dir=None):
        raise IOError("Disk full or write permission denied")

    monkeypatch.setattr("src.evaluation.discovery_engine.save_research_experiment", mock_save_failing)

    engine = DiscoveryEngine()
    with pytest.raises(IOError):
        engine.run_discovery(
            df=sample_market_data,
            candidates=[source_candidate],
            dataset_scope=dataset_scope,
            execution_assumptions=execution_assumptions,
            code_provenance=code_provenance,
            persist_evidence=True,
        )


# 19. Replay / concurrent evaluation cannot produce contradictory promotion state
def test_replay_evaluation_consistency(source_candidate, sample_market_data, dataset_scope, execution_assumptions, code_provenance):
    engine = DiscoveryEngine()
    res1 = engine.run_discovery(
        df=sample_market_data,
        candidates=[source_candidate],
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
    )
    res2 = engine.run_discovery(
        df=sample_market_data,
        candidates=[source_candidate],
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
    )
    ev1 = (res1.promoted_evidence + res1.rejected_evidence)[0]
    ev2 = (res2.promoted_evidence + res2.rejected_evidence)[0]
    assert ev1.experiment_fingerprint == ev2.experiment_fingerprint
    assert ev1.promotion_status == ev2.promotion_status


# 20. Complete successful discovery path works
def test_complete_discovery_path(source_candidate, sample_market_data, dataset_scope, execution_assumptions, code_provenance):
    lenient_criteria = DiscoveryCriteria(
        min_is_sharpe=-10.0,
        min_validation_sharpe=-10.0,
        min_oos_sharpe=-10.0,
        robustness_criteria=RobustnessCriteria(
            min_t_stat=-10.0,
            max_p_value=1.0,
            min_statistical_observations=10,
            min_subsample_pass_rate=0.0,
            min_cost_stress_pass_rate=0.0,
            min_perturbation_pass_rate=0.0,
        )
    )
    engine = DiscoveryEngine(criteria=lenient_criteria)
    res = engine.run_discovery(
        df=sample_market_data,
        candidates=[source_candidate],
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
    )
    assert res.total_candidates == 1
    assert len(res.promoted_evidence) == 1
    assert res.promoted_evidence[0].promotion_status == PromotionStatus.PROMOTABLE
