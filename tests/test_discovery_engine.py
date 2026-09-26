"""Tests for Research Discovery Engine, Selection Governance, and Candidate Generation in Project 1."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.candidate_generator import (
    CandidateGenerator,
    CandidateGeneratorSpec,
    CandidateSpec,
    ResearchSearchSpace,
)
from src.evaluation.discovery_engine import (
    DiscoveryCriteria,
    DiscoveryEngine,
    DiscoveryRunResult,
    ResearchSearchPolicy,
    ResearchTrialRecord,
)
from src.evaluation.selection_governance import (
    ResearchSelectionAssessment,
    SelectionGovernanceStatus,
)
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    EvidencePartitionRole,
    ExecutionAssumptions,
    PromotionStatus,
    RejectionReason,
    ResearchEvidence,
)
from src.evaluation.research_store import load_research_experiment


@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    """Generate deterministic time-series market data for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(100) * 1.5)
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000,
        }
    )
    return df


@pytest.fixture
def dataset_scope() -> DatasetScope:
    return DatasetScope(
        dataset_id="ds_xauusd_daily_2023",
        symbol="XAUUSD",
        timeframe="1D",
        start_date="2023-01-01",
        end_date="2023-04-10",
    )


@pytest.fixture
def execution_assumptions() -> ExecutionAssumptions:
    return ExecutionAssumptions(
        transaction_cost=0.0005,
        slippage=0.0002,
        latency_ms=50.0,
    )


@pytest.fixture
def code_provenance() -> CodeProvenance:
    return CodeProvenance(
        commit_sha="a1b2c3d4e5f67890123456789012345678901234",
        repository_status="clean",
        author="Jules",
    )


def test_candidate_generator_deterministic_output():
    grid = {
        "fast_window": [5, 10],
        "slow_window": [20, 50],
    }
    spec = CandidateGeneratorSpec(
        generator_name="grid_search",
        generator_version="1.0",
        strategy_name="baseline",
        parameter_grid=grid,
        random_seed=123,
    )
    generator = CandidateGenerator(spec)
    candidates1 = generator.generate_candidates()
    candidates2 = generator.generate_candidates()

    assert len(candidates1) == 4
    assert [c.candidate_id for c in candidates1] == [c.candidate_id for c in candidates2]
    assert candidates1[0].strategy_name == "baseline"
    assert candidates1[0].parameters == {"fast_window": 5, "slow_window": 20}


def test_candidate_spec_validation():
    with pytest.raises(ValueError, match="generator_name must be a non-empty string"):
        CandidateSpec(
            generator_name="",
            generator_version="1.0",
            strategy_name="baseline",
            parameters={},
        )

    with pytest.raises(ValueError, match="strategy_name must be a non-empty string"):
        CandidateSpec(
            generator_name="grid",
            generator_version="1.0",
            strategy_name="",
            parameters={},
        )


def test_discovery_engine_full_vertical_slice(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance
):
    cand_spec = CandidateGeneratorSpec(
        generator_name="grid_search",
        generator_version="1.0",
        strategy_name="baseline",
        parameter_grid={"fast_window": [3], "slow_window": [8]},
    )
    candidates = CandidateGenerator(cand_spec).generate_candidates()

    criteria = DiscoveryCriteria(
        min_observations_is=10,
        min_observations_oos=5,
        min_is_sharpe=-2.0,
        min_validation_sharpe=-2.0,
        min_oos_sharpe=-2.0,
        max_oos_sharpe_degradation=10.0,
        min_walk_forward_positive_ratio=0.0,
    )
    engine = DiscoveryEngine(criteria=criteria)

    with tempfile.TemporaryDirectory() as tmp_dir:
        res = engine.run_discovery(
            df=sample_market_data,
            candidates=candidates,
            dataset_scope=dataset_scope,
            execution_assumptions=execution_assumptions,
            code_provenance=code_provenance,
            wf_train_size=30,
            wf_test_size=15,
            persist_evidence=False,
        )

        assert isinstance(res, DiscoveryRunResult)
        assert res.candidates_evaluated == 1
        assert len(res.promoted_evidence) + len(res.rejected_evidence) == 1
        assert len(res.selection_assessments) == 1
        assert isinstance(res.selection_assessments[0], ResearchSelectionAssessment)

        evidence = (
            res.promoted_evidence[0]
            if res.promoted_evidence
            else res.rejected_evidence[0]
        )
        assert isinstance(evidence, ResearchEvidence)
        assert len(evidence.partitions) == 4  # IS, Validation, OOS, Walk-Forward
        roles = [p.role for p in evidence.partitions]
        assert EvidencePartitionRole.IN_SAMPLE in roles
        assert EvidencePartitionRole.VALIDATION in roles
        assert EvidencePartitionRole.OUT_OF_SAMPLE in roles
        assert EvidencePartitionRole.WALK_FORWARD in roles


def test_discovery_engine_evidence_persistence(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance
):
    cand_spec = CandidateGeneratorSpec(
        generator_name="grid_search",
        generator_version="1.0",
        strategy_name="baseline",
        parameter_grid={"fast_window": [3], "slow_window": [8]},
    )
    candidates = CandidateGenerator(cand_spec).generate_candidates()

    engine = DiscoveryEngine(
        criteria=DiscoveryCriteria(
            min_observations_is=10,
            min_observations_oos=5,
            min_is_sharpe=-5.0,
            min_validation_sharpe=-5.0,
            min_oos_sharpe=-5.0,
            max_oos_sharpe_degradation=100.0,
            min_walk_forward_positive_ratio=0.0,
        )
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        import src.evaluation.discovery_engine as de_mod

        original_save = de_mod.save_research_experiment

        def custom_save(ev):
            return original_save(ev, base_dir=tmp_path)

        de_mod.save_research_experiment = custom_save

        try:
            res = engine.run_discovery(
                df=sample_market_data,
                candidates=candidates,
                dataset_scope=dataset_scope,
                execution_assumptions=execution_assumptions,
                code_provenance=code_provenance,
                wf_train_size=30,
                wf_test_size=15,
                persist_evidence=True,
            )

            all_ev = res.promoted_evidence + res.rejected_evidence
            ev = all_ev[0]

            saved_file = tmp_path / ev.experiment_fingerprint / "evidence.json"
            assert saved_file.exists()

            loaded_ev = load_research_experiment(saved_file)
            assert loaded_ev.evidence_id == ev.evidence_id
            assert loaded_ev.experiment_fingerprint == ev.experiment_fingerprint
        finally:
            de_mod.save_research_experiment = original_save


def test_discovery_rejection_insufficient_data(
    sample_market_data, execution_assumptions, code_provenance
):
    short_df = sample_market_data.head(15)
    short_dataset_scope = DatasetScope(
        dataset_id="ds_short",
        symbol="XAUUSD",
        timeframe="1D",
        start_date="2023-01-01",
        end_date="2023-01-15",
    )

    cand_spec = CandidateGeneratorSpec(
        generator_name="grid_search",
        generator_version="1.0",
        strategy_name="baseline",
        parameter_grid={"fast_window": [3], "slow_window": [8]},
    )
    candidates = CandidateGenerator(cand_spec).generate_candidates()

    engine = DiscoveryEngine(
        criteria=DiscoveryCriteria(min_observations_is=50, min_observations_oos=20)
    )

    res = engine.run_discovery(
        df=short_df,
        candidates=candidates,
        dataset_scope=short_dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
    )

    assert len(res.promoted_evidence) == 0
    assert len(res.rejected_evidence) == 1
    rejected = res.rejected_evidence[0]
    assert rejected.promotion_status == PromotionStatus.REJECTED
    assert RejectionReason.INSUFFICIENT_DATA in rejected.rejection_reasons


def test_discovery_rejection_duplicate_candidate(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance
):
    cand = CandidateSpec(
        generator_name="grid",
        generator_version="1.0",
        strategy_name="baseline",
        parameters={"fast_window": 3, "slow_window": 8},
    )
    candidates = (cand, cand)

    engine = DiscoveryEngine(
        criteria=DiscoveryCriteria(
            min_observations_is=10,
            min_observations_oos=5,
            min_is_sharpe=-10.0,
            min_validation_sharpe=-10.0,
            min_oos_sharpe=-10.0,
            max_oos_sharpe_degradation=100.0,
            min_walk_forward_positive_ratio=0.0,
        )
    )

    res = engine.run_discovery(
        df=sample_market_data,
        candidates=candidates,
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        wf_train_size=30,
        wf_test_size=15,
    )

    assert len(res.rejected_evidence) >= 1
    dup_rejected = [
        ev for ev in res.rejected_evidence if RejectionReason.DUPLICATE_CANDIDATE in ev.rejection_reasons
    ]
    assert len(dup_rejected) == 1


def test_discovery_dataset_scope_out_of_bounds(
    sample_market_data, execution_assumptions, code_provenance
):
    invalid_scope = DatasetScope(
        dataset_id="ds_xauusd",
        symbol="XAUUSD",
        timeframe="1D",
        start_date="2020-01-01",
        end_date="2023-04-10",
    )

    cand = CandidateSpec(
        generator_name="grid",
        generator_version="1.0",
        strategy_name="baseline",
        parameters={},
    )

    engine = DiscoveryEngine()

    with pytest.raises(ValueError, match="extend beyond actual data boundaries"):
        engine.run_discovery(
            df=sample_market_data,
            candidates=(cand,),
            dataset_scope=invalid_scope,
            execution_assumptions=execution_assumptions,
            code_provenance=code_provenance,
        )


def test_research_search_space_determinism():
    c1 = CandidateSpec("grid", "1.0", "baseline", {"fast_window": 10})
    c2 = CandidateSpec("grid", "1.0", "baseline", {"fast_window": 5})

    # Passing in reverse order
    space1 = ResearchSearchSpace(candidate_definitions=(c1, c2), search_id="search_test")
    space2 = ResearchSearchSpace(candidate_definitions=(c2, c1), search_id="search_test")

    # Candidates must be deterministically ordered by candidate_id
    assert [c.candidate_id for c in space1.candidate_definitions] == [c.candidate_id for c in space2.candidate_definitions]
    assert space1.search_fingerprint == space2.search_fingerprint


def test_trial_ledger_and_budget_truncation(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance
):
    cands = tuple(
        CandidateSpec("grid", "1.0", "baseline", {"fast_window": i}) for i in range(5)
    )
    space = ResearchSearchSpace(candidate_definitions=cands, search_id="search_budget")
    policy = ResearchSearchPolicy(max_trials=2)

    engine = DiscoveryEngine(
        criteria=DiscoveryCriteria(
            min_observations_is=10,
            min_observations_oos=5,
            min_is_sharpe=-10.0,
            min_validation_sharpe=-10.0,
            min_oos_sharpe=-10.0,
            max_oos_sharpe_degradation=100.0,
            min_walk_forward_positive_ratio=0.0,
        )
    )

    res = engine.run_discovery(
        df=sample_market_data,
        candidates=space,
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        search_policy=policy,
        wf_train_size=30,
        wf_test_size=15,
    )

    assert res.search_truncated is True
    assert res.candidates_evaluated == 2
    assert res.trial_count == 2
    assert len(res.trial_ledger) == 2
    assert res.trial_ledger[0].search_id == "search_budget"
    assert res.trial_ledger[0].trial_index == 0
    assert len(res.selection_assessments) == 2
    assert res.selection_assessments[0].status == SelectionGovernanceStatus.SELECTION_CONTEXT_RECORDED
    assert res.selection_assessments[0].trial_count == 2


def test_failure_isolation(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance, monkeypatch
):
    c1 = CandidateSpec("grid", "1.0", "baseline", {"fast_window": 3, "slow_window": 8})
    c2 = CandidateSpec("grid", "1.0", "baseline", {"fast_window": 5, "slow_window": 10})

    engine = DiscoveryEngine(
        criteria=DiscoveryCriteria(
            min_observations_is=10,
            min_observations_oos=5,
            min_is_sharpe=-10.0,
            min_validation_sharpe=-10.0,
            min_oos_sharpe=-10.0,
            max_oos_sharpe_degradation=100.0,
            min_walk_forward_positive_ratio=0.0,
        )
    )

    orig_eval = engine._evaluate_candidate

    def mock_eval(*args, **kwargs):
        cand = kwargs.get("cand")
        if cand.parameters.get("fast_window") == 3:
            raise RuntimeError("Simulated execution failure")
        return orig_eval(*args, **kwargs)

    monkeypatch.setattr(engine, "_evaluate_candidate", mock_eval)

    res = engine.run_discovery(
        df=sample_market_data,
        candidates=(c1, c2),
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        search_policy=ResearchSearchPolicy(max_trials=10, fail_fast=False),
        wf_train_size=30,
        wf_test_size=15,
    )

    assert res.failed_trial_count == 1
    assert res.successful_trial_count == 1
    failed_record = [t for t in res.trial_ledger if t.status == "FAILED"][0]
    assert "Simulated execution failure" in failed_record.error_message


def test_reproducibility_repeated_runs(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance
):
    cand_spec = CandidateGeneratorSpec(
        generator_name="grid_search",
        generator_version="1.0",
        strategy_name="baseline",
        parameter_grid={"fast_window": [3, 5], "slow_window": [8, 12]},
    )
    candidates = CandidateGenerator(cand_spec).generate_candidates()
    space = ResearchSearchSpace(candidate_definitions=candidates, search_id="search_repro")

    engine = DiscoveryEngine(
        criteria=DiscoveryCriteria(
            min_observations_is=10,
            min_observations_oos=5,
            min_is_sharpe=-10.0,
            min_validation_sharpe=-10.0,
            min_oos_sharpe=-10.0,
            max_oos_sharpe_degradation=100.0,
            min_walk_forward_positive_ratio=0.0,
        )
    )

    res1 = engine.run_discovery(
        df=sample_market_data,
        candidates=space,
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        wf_train_size=30,
        wf_test_size=15,
    )

    res2 = engine.run_discovery(
        df=sample_market_data,
        candidates=space,
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        wf_train_size=30,
        wf_test_size=15,
    )

    assert res1.search_space_fingerprint == res2.search_space_fingerprint
    assert [t.candidate_id for t in res1.trial_ledger] == [t.candidate_id for t in res2.trial_ledger]
    assert [ev.evidence_id for ev in res1.promoted_evidence] == [ev.evidence_id for ev in res2.promoted_evidence]
