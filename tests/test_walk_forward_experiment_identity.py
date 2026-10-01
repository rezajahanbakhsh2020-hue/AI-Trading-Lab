"""Comprehensive tests for Walk-Forward execution protocol identity and reproducibility.

Verifies requirements A through K:
A. Same effective WF configuration => same ResearchExperimentSpec fingerprint.
B. Different wf_train_size => different experiment fingerprint.
C. Different wf_test_size => different experiment fingerprint.
D. ResearchHypothesis and ResearchExperimentSpec preserve the same WF protocol and lineage.
E. DiscoveryEngine does not mark candidates as duplicate solely because all fields match except WF configuration.
F. Default WF resolution preserves current behavior: train = int(n * 0.40), test = int(n * 0.15).
G. Explicit configuration matching the effective defaults produces identical identity relationship.
H. Persist -> load/reconstruct preserves WF protocol and experiment fingerprint.
I. Existing canonical runner tests satisfy: evidence.experiment_fingerprint == spec.fingerprint.
J. Existing Walk-Forward execution metrics/partition behavior remain unchanged.
K. Malformed persisted WF configuration fails closed.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.candidate_generator import CandidateSpec
from src.evaluation.discovery_engine import DiscoveryCriteria, DiscoveryEngine
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    EvidencePartitionRole,
    ExecutionAssumptions,
    PromotionStatus,
    RejectionReason,
    ResearchCandidate,
    ResearchEvidence,
    ResearchExperimentSpec,
    ResearchHypothesis,
    WalkForwardProtocol,
    resolve_walk_forward_protocol,
)
from src.evaluation.hypothesis_generator import accept_hypothesis_for_research
from src.evaluation.research_qualification import qualify_research_evidence
from src.evaluation.research_runner import run_research_experiment
from src.evaluation.research_store import (
    load_research_experiment,
    reconstruct_research_evidence,
    save_research_experiment,
)


def make_sample_df(n_rows: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=n_rows, freq="1D")
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n_rows) * 1.0)
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
    df["return"] = df["close"].pct_change().fillna(0.0)
    return df


def make_base_spec_kwargs() -> dict:
    return {
        "hypothesis": "Test WF identity spec",
        "methodology_version": "discovery_v1.0",
        "strategy_name": "baseline",
        "strategy_version": "1.0.0",
        "dataset_scope": DatasetScope(
            dataset_id="ds_test",
            symbol="XAUUSD",
            timeframe="1D",
            start_date="2025-01-01",
            end_date="2025-04-10",
        ),
        "execution_assumptions": ExecutionAssumptions(
            transaction_cost=0.001,
            slippage=0.001,
            latency_ms=10.0,
        ),
        "code_provenance": CodeProvenance(
            commit_sha="abcdef1234567890abcdef1234567890abcdef12",
        ),
        "benchmark_reference": "buy_and_hold",
        "parameters": {"fast_window": 3, "slow_window": 8},
        "random_seed": 42,
    }


def test_A_same_effective_wf_config_same_fingerprint():
    kwargs = make_base_spec_kwargs()
    wf1 = WalkForwardProtocol(train_size=40, test_size=15)
    wf2 = WalkForwardProtocol(train_size=40, test_size=15)

    spec1 = ResearchExperimentSpec(**kwargs, walk_forward_protocol=wf1)
    spec2 = ResearchExperimentSpec(**kwargs, walk_forward_protocol=wf2)

    assert spec1.fingerprint == spec2.fingerprint


def test_B_different_wf_train_size_different_fingerprint():
    kwargs = make_base_spec_kwargs()
    wf1 = WalkForwardProtocol(train_size=40, test_size=15)
    wf2 = WalkForwardProtocol(train_size=50, test_size=15)

    spec1 = ResearchExperimentSpec(**kwargs, walk_forward_protocol=wf1)
    spec2 = ResearchExperimentSpec(**kwargs, walk_forward_protocol=wf2)

    assert spec1.fingerprint != spec2.fingerprint


def test_C_different_wf_test_size_different_fingerprint():
    kwargs = make_base_spec_kwargs()
    wf1 = WalkForwardProtocol(train_size=40, test_size=15)
    wf2 = WalkForwardProtocol(train_size=40, test_size=20)

    spec1 = ResearchExperimentSpec(**kwargs, walk_forward_protocol=wf1)
    spec2 = ResearchExperimentSpec(**kwargs, walk_forward_protocol=wf2)

    assert spec1.fingerprint != spec2.fingerprint


def test_D_hypothesis_and_spec_preserve_wf_protocol_and_lineage():
    kwargs = make_base_spec_kwargs()
    wf = WalkForwardProtocol(train_size=35, test_size=12)

    hypothesis = ResearchHypothesis(
        statement=kwargs["hypothesis"],
        methodology_version=kwargs["methodology_version"],
        strategy_name=kwargs["strategy_name"],
        strategy_version=kwargs["strategy_version"],
        dataset_scope=kwargs["dataset_scope"],
        execution_assumptions=kwargs["execution_assumptions"],
        code_provenance=kwargs["code_provenance"],
        benchmark_reference=kwargs["benchmark_reference"],
        parameters=kwargs["parameters"],
        random_seed=kwargs["random_seed"],
        walk_forward_protocol=wf,
    )

    spec = hypothesis.to_experiment_spec()
    assert spec.walk_forward_protocol == wf
    assert spec.fingerprint == hypothesis.fingerprint

    reconstructed_hyp = ResearchHypothesis.from_experiment_spec(spec)
    assert reconstructed_hyp.walk_forward_protocol == wf
    assert reconstructed_hyp.fingerprint == spec.fingerprint


def test_E_discovery_engine_duplicate_handling_aware_of_wf_config():
    df = make_sample_df(100)
    kwargs = make_base_spec_kwargs()
    cand = CandidateSpec(
        generator_name="grid",
        generator_version="1.0",
        strategy_name="baseline",
        parameters={"fast_window": 3, "slow_window": 8},
    )

    criteria = DiscoveryCriteria(
        min_observations_is=10,
        min_observations_oos=5,
        min_is_sharpe=-10.0,
        min_validation_sharpe=-10.0,
        min_oos_sharpe=-10.0,
        max_oos_sharpe_degradation=100.0,
        min_walk_forward_positive_ratio=0.0,
    )
    engine = DiscoveryEngine(criteria=criteria)

    res1 = engine.run_discovery(
        df=df,
        candidates=(cand,),
        dataset_scope=kwargs["dataset_scope"],
        execution_assumptions=kwargs["execution_assumptions"],
        code_provenance=kwargs["code_provenance"],
        wf_train_size=30,
        wf_test_size=10,
    )

    res2 = engine.run_discovery(
        df=df,
        candidates=(cand,),
        dataset_scope=kwargs["dataset_scope"],
        execution_assumptions=kwargs["execution_assumptions"],
        code_provenance=kwargs["code_provenance"],
        wf_train_size=40,
        wf_test_size=15,
    )

    ev1 = (res1.promoted_evidence + res1.rejected_evidence)[0]
    ev2 = (res2.promoted_evidence + res2.rejected_evidence)[0]

    assert ev1.experiment_fingerprint != ev2.experiment_fingerprint
    assert RejectionReason.DUPLICATE_CANDIDATE not in ev2.rejection_reasons


def test_F_default_wf_resolution_preserves_behavior():
    n = 100
    wf = resolve_walk_forward_protocol(n)
    assert wf.train_size == int(100 * 0.40)  # 40
    assert wf.test_size == int(100 * 0.15)  # 15

    n2 = 250
    wf2 = resolve_walk_forward_protocol(n2)
    assert wf2.train_size == 100
    assert wf2.test_size == 37


def test_G_explicit_config_matching_defaults_identity():
    n = 100
    wf_default = resolve_walk_forward_protocol(n)
    wf_explicit = resolve_walk_forward_protocol(n, train_size=40, test_size=15)

    assert wf_default == wf_explicit

    kwargs = make_base_spec_kwargs()
    spec_default = ResearchExperimentSpec(**kwargs, walk_forward_protocol=wf_default)
    spec_explicit = ResearchExperimentSpec(**kwargs, walk_forward_protocol=wf_explicit)

    assert spec_default.fingerprint == spec_explicit.fingerprint


def test_H_persist_load_reconstruct_preserves_wf_protocol_and_fingerprint():
    df = make_sample_df(100)
    kwargs = make_base_spec_kwargs()
    wf = WalkForwardProtocol(train_size=35, test_size=12)
    spec = ResearchExperimentSpec(**kwargs, walk_forward_protocol=wf)
    hyp = accept_hypothesis_for_research(ResearchHypothesis.from_experiment_spec(spec))

    evidence = run_research_experiment(hyp, df=df)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        saved_path = save_research_experiment(evidence, base_dir=tmp_path)
        assert saved_path.exists()

        loaded_ev = load_research_experiment(saved_path, base_dir=tmp_path)
        assert loaded_ev.spec.walk_forward_protocol == wf
        assert loaded_ev.experiment_fingerprint == spec.fingerprint
        assert loaded_ev.evidence_id == evidence.evidence_id

        # Qualification re-check
        qual_res = qualify_research_evidence(loaded_ev)
        assert qual_res.evidence_fingerprint == spec.fingerprint


def test_I_existing_canonical_runner_satisfies_fingerprint_invariant():
    df = make_sample_df(100)
    kwargs = make_base_spec_kwargs()
    wf = WalkForwardProtocol(train_size=40, test_size=15)
    spec = ResearchExperimentSpec(**kwargs, walk_forward_protocol=wf)
    hyp = accept_hypothesis_for_research(ResearchHypothesis.from_experiment_spec(spec))

    evidence = run_research_experiment(hyp, df=df)

    assert evidence.experiment_fingerprint == spec.fingerprint
    assert evidence.spec.fingerprint == spec.fingerprint


def test_J_walk_forward_execution_metrics_and_partitions():
    df = make_sample_df(100)
    kwargs = make_base_spec_kwargs()
    wf = WalkForwardProtocol(train_size=40, test_size=15)
    spec = ResearchExperimentSpec(**kwargs, walk_forward_protocol=wf)
    hyp = accept_hypothesis_for_research(ResearchHypothesis.from_experiment_spec(spec))

    evidence = run_research_experiment(hyp, df=df)

    wf_partitions = [p for p in evidence.partitions if p.role == EvidencePartitionRole.WALK_FORWARD]
    assert len(wf_partitions) == 1
    wf_p = wf_partitions[0]
    assert wf_p.additional_metrics["train_size"] == 40.0
    assert wf_p.additional_metrics["test_size"] == 15.0


def test_K_malformed_persisted_wf_config_fails_closed():
    kwargs = make_base_spec_kwargs()
    wf = WalkForwardProtocol(train_size=40, test_size=15)
    spec = ResearchExperimentSpec(**kwargs, walk_forward_protocol=wf)
    hyp = accept_hypothesis_for_research(ResearchHypothesis.from_experiment_spec(spec))
    df = make_sample_df(100)
    evidence = run_research_experiment(hyp, df=df)

    d = evidence.as_dict()

    # Case 1: Non-dictionary walk_forward_protocol
    d_malformed_type = json.loads(json.dumps(d))
    d_malformed_type["spec"]["walk_forward_protocol"] = "invalid_string"
    with pytest.raises(ValueError, match="walk_forward_protocol"):
        reconstruct_research_evidence(d_malformed_type)

    # Case 2: Missing required field train_size
    d_missing_field = json.loads(json.dumps(d))
    d_missing_field["spec"]["walk_forward_protocol"] = {"test_size": 15}
    with pytest.raises(ValueError, match="missing 'train_size' or 'test_size'"):
        reconstruct_research_evidence(d_missing_field)

    # Case 3: Negative integer value
    d_negative_val = json.loads(json.dumps(d))
    d_negative_val["spec"]["walk_forward_protocol"] = {"train_size": -40, "test_size": 15}
    with pytest.raises(ValueError, match="train_size must be a positive integer"):
        reconstruct_research_evidence(d_negative_val)

    # Case 4: Modified values producing fingerprint mismatch
    d_tampered = json.loads(json.dumps(d))
    d_tampered["spec"]["walk_forward_protocol"] = {"train_size": 50, "test_size": 15}
    with pytest.raises(ValueError, match="does not match spec fingerprint"):
        reconstruct_research_evidence(d_tampered)


def test_walk_forward_protocol_type_and_value_validations():
    with pytest.raises(TypeError, match="train_size must be an integer"):
        WalkForwardProtocol(train_size="40", test_size=15)  # type: ignore

    with pytest.raises(TypeError, match="test_size must be an integer"):
        WalkForwardProtocol(train_size=40, test_size=True)  # type: ignore

    with pytest.raises(ValueError, match="train_size must be a positive integer"):
        WalkForwardProtocol(train_size=0, test_size=15)

    with pytest.raises(ValueError, match="test_size must be a positive integer"):
        WalkForwardProtocol(train_size=40, test_size=-5)


def test_runner_argument_conflict_with_spec_fails_closed():
    df = make_sample_df(100)
    kwargs = make_base_spec_kwargs()
    wf = WalkForwardProtocol(train_size=40, test_size=15)
    spec = ResearchExperimentSpec(**kwargs, walk_forward_protocol=wf)
    hyp = accept_hypothesis_for_research(ResearchHypothesis.from_experiment_spec(spec))

    with pytest.raises(ValueError, match="conflicts with spec.walk_forward_protocol"):
        run_research_experiment(hyp, df=df, wf_train_size=50)

    with pytest.raises(ValueError, match="conflicts with spec.walk_forward_protocol"):
        run_research_experiment(hyp, df=df, wf_test_size=20)
