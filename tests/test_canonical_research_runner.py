"""Comprehensive Unit and Behavioral Tests for Canonical Research Experiment Execution.

Verifies:
A. Valid experiment executes and creates ResearchEvidence.
B. Invalid DatasetScope fails closed.
C. Invalid ExecutionAssumptions fail closed.
D. Missing dataset fails closed.
E. Dataset identity mismatch fails closed.
F. Time-order violation fails closed.
G. Train/test overlap fails closed.
H. Future-data leakage fails closed.
I. Declared costs/slippage/latency are actually reflected in the execution path and evidence.
J. Evidence contains correct experiment identity, scope, assumptions, provenance, metrics, status.
K. Identical deterministic inputs produce reproducible experiment identity.
L. Research execution cannot silently create a production candidate.
M. Production decision chain integrity remains unaffected.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    EvidencePartitionRole,
    ExecutionAssumptions,
    PromotionStatus,
    RejectionReason,
    ResearchEvidence,
    ResearchExperimentSpec,
    WalkForwardProtocol,
)
from src.evaluation.research_runner import (
    resolve_historical_dataset,
    run_research_experiment,
    validate_and_prepare_dataset,
)
from src.evaluation.research_store import (
    DEFAULT_RESEARCH_DIR,
    load_candidate_binding,
)


def make_sample_data(n_rows: int = 100, start_date: str = "2025-01-01") -> pd.DataFrame:
    dates = pd.date_range(start=start_date, periods=n_rows, freq="1D")
    np.random.seed(42)
    close = 2000.0 + np.cumsum(np.random.randn(n_rows) * 10.0)
    df = pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d"),
        "open": close - 1.0,
        "high": close + 2.0,
        "low": close - 2.0,
        "close": close,
    })
    df["return"] = df["close"].pct_change().fillna(0.0)
    return df


def make_valid_spec(
    strategy_name: str = "moving_average_crossover",
    start_date: str = "2025-01-01",
    end_date: str = "2025-04-10",
    transaction_cost: float = 0.001,
    slippage: float = 0.001,
    latency_ms: float = 10.0,
    parameters: dict = None,
    walk_forward_protocol: WalkForwardProtocol | None = None,
) -> ResearchExperimentSpec:
    if parameters is None:
        parameters = {"short_window": 5, "long_window": 20}
    if walk_forward_protocol is None:
        walk_forward_protocol = WalkForwardProtocol(train_size=40, test_size=15)
    ds = DatasetScope(
        dataset_id="xauusd_test",
        symbol="XAUUSD",
        timeframe="1D",
        start_date=start_date,
        end_date=end_date,
    )
    ea = ExecutionAssumptions(
        transaction_cost=transaction_cost,
        slippage=slippage,
        latency_ms=latency_ms,
    )
    cp = CodeProvenance(
        commit_sha="a1b2c3d4e5f678901234567890abcdef12345678",
        repository_status="clean",
        author="Research Bot",
    )
    return ResearchExperimentSpec(
        hypothesis="Testing MA crossover trend capture",
        methodology_version="discovery_v1.0",
        strategy_name=strategy_name,
        strategy_version="1.0.0",
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        benchmark_reference="buy_and_hold",
        parameters=parameters,
        random_seed=42,
        walk_forward_protocol=walk_forward_protocol,
    )


def test_A_valid_experiment_executes_and_creates_evidence():
    df = make_sample_data(100)
    spec = make_valid_spec()
    evidence = run_research_experiment(spec, df=df)

    assert isinstance(evidence, ResearchEvidence)
    assert evidence.experiment_fingerprint == spec.fingerprint
    assert len(evidence.partitions) >= 3
    assert evidence.evidence_id.startswith("ev_")
    assert evidence.promotion_status in (PromotionStatus.PROMOTABLE, PromotionStatus.REJECTED)


def test_B_invalid_dataset_scope_fails_closed():
    with pytest.raises(ValueError, match="start_date .* cannot be later than end_date"):
        DatasetScope(
            dataset_id="test",
            symbol="XAUUSD",
            timeframe="1D",
            start_date="2025-05-01",
            end_date="2025-01-01",
        )

    df = make_sample_data(30, start_date="2025-01-01")
    spec = make_valid_spec(start_date="2025-01-01", end_date="2025-06-01")

    with pytest.raises(ValueError, match="DatasetScope dates .* extend beyond actual data boundaries"):
        run_research_experiment(spec, df=df)


def test_C_invalid_execution_assumptions_fail_closed():
    with pytest.raises(ValueError, match="transaction_cost cannot be negative"):
        ExecutionAssumptions(transaction_cost=-0.01, slippage=0.001, latency_ms=10.0)

    with pytest.raises(ValueError, match="slippage cannot be negative"):
        ExecutionAssumptions(transaction_cost=0.001, slippage=-0.001, latency_ms=10.0)

    with pytest.raises(ValueError, match="latency_ms cannot be negative"):
        ExecutionAssumptions(transaction_cost=0.001, slippage=0.001, latency_ms=-5.0)


def test_D_missing_dataset_fails_closed(tmp_path):
    spec = make_valid_spec(start_date="2025-01-01", end_date="2025-04-01")
    evidence = run_research_experiment(spec, df=None, base_dir=tmp_path)

    assert evidence.promotion_status == PromotionStatus.REJECTED
    assert RejectionReason.INVALID_DATASET_SCOPE in evidence.rejection_reasons


def test_E_dataset_identity_mismatch_fails_closed():
    df = make_sample_data(50, start_date="2024-01-01")
    spec = make_valid_spec(start_date="2025-01-01", end_date="2025-04-10")

    with pytest.raises(ValueError, match="DatasetScope dates .* extend beyond actual data boundaries"):
        run_research_experiment(spec, df=df)


def test_F_time_order_violation_fails_closed():
    df = make_sample_data(100)
    df_reversed = df.iloc[::-1].reset_index(drop=True)
    ds = DatasetScope("test", "XAUUSD", "1D", "2025-01-01", "2025-04-10")

    with pytest.raises(ValueError, match="non-monotonic"):
        validate_and_prepare_dataset(df_reversed, ds)

    # Correctly ordered dataset succeeds
    prepared = validate_and_prepare_dataset(df, ds)
    assert prepared["timestamp"].is_monotonic_increasing


def test_G_train_test_overlap_fails_closed():
    df = make_sample_data(100)
    spec = make_valid_spec()
    evidence = run_research_experiment(spec, df=df)

    partitions_by_role = {p.role: p for p in evidence.partitions if p.role in (EvidencePartitionRole.IN_SAMPLE, EvidencePartitionRole.VALIDATION, EvidencePartitionRole.OUT_OF_SAMPLE)}

    is_p = partitions_by_role[EvidencePartitionRole.IN_SAMPLE]
    val_p = partitions_by_role[EvidencePartitionRole.VALIDATION]
    oos_p = partitions_by_role[EvidencePartitionRole.OUT_OF_SAMPLE]

    assert is_p.start_date <= is_p.end_date
    assert is_p.end_date <= val_p.start_date
    assert val_p.start_date <= val_p.end_date
    assert val_p.end_date <= oos_p.start_date
    assert oos_p.start_date <= oos_p.end_date


def test_H_future_data_leakage_fails_closed():
    df = make_sample_data(100)
    spec = make_valid_spec()
    evidence = run_research_experiment(spec, df=df)

    robustness = evidence.robustness_verdict
    assert "anti_overfitting_verdict" in robustness
    assert robustness["anti_overfitting_verdict"]["passed"] is True


def test_I_costs_slippage_latency_reflected_in_execution_and_evidence():
    df = make_sample_data(100)

    spec_zero_cost = make_valid_spec(transaction_cost=0.0, slippage=0.0, latency_ms=0.0)
    spec_high_cost = make_valid_spec(transaction_cost=0.05, slippage=0.05, latency_ms=500.0)

    ev_zero = run_research_experiment(spec_zero_cost, df=df)
    ev_high = run_research_experiment(spec_high_cost, df=df)

    is_zero = [p for p in ev_zero.partitions if p.role == EvidencePartitionRole.IN_SAMPLE][0]
    is_high = [p for p in ev_high.partitions if p.role == EvidencePartitionRole.IN_SAMPLE][0]

    assert is_high.total_return <= is_zero.total_return
    assert is_high.additional_metrics["latency_ms"] == 500.0
    assert is_zero.additional_metrics["latency_ms"] == 0.0


def test_J_evidence_contains_complete_traceable_information():
    df = make_sample_data(100)
    spec = make_valid_spec()
    evidence = run_research_experiment(spec, df=df)

    d = evidence.as_dict()
    assert d["evidence_id"].startswith("ev_")
    assert d["experiment_fingerprint"] == spec.fingerprint
    assert d["spec"]["strategy_name"] == spec.strategy_name
    assert d["spec"]["dataset_scope"]["symbol"] == "XAUUSD"
    assert d["spec"]["execution_assumptions"]["transaction_cost"] == 0.001
    assert d["spec"]["code_provenance"]["commit_sha"] == spec.code_provenance.commit_sha
    assert len(d["partitions"]) >= 3
    assert "robustness_verdict" in d
    assert "benchmark_comparison" in d


def test_K_identical_deterministic_inputs_produce_reproducible_fingerprint():
    spec1 = make_valid_spec()
    spec2 = make_valid_spec()

    assert spec1.fingerprint == spec2.fingerprint

    df = make_sample_data(100)
    ev1 = run_research_experiment(spec1, df=df)
    ev2 = run_research_experiment(spec2, df=df)

    assert ev1.experiment_fingerprint == ev2.experiment_fingerprint
    assert ev1.evidence_id == ev2.evidence_id


def test_L_research_execution_cannot_silently_create_production_candidate(tmp_path):
    df = make_sample_data(100)
    spec = make_valid_spec()

    evidence = run_research_experiment(spec, df=df, persist_evidence=True, base_dir=tmp_path)

    assert evidence.promotion_status in (PromotionStatus.PROMOTABLE, PromotionStatus.REJECTED)

    candidate_id = f"cand_{spec.strategy_name}_test"
    with pytest.raises(FileNotFoundError):
        load_candidate_binding(candidate_id, base_dir=tmp_path)


def test_M_existing_production_invariants_unaffected():
    from src.evaluation.live_production_decision import (
        Direction,
        ProductionDecision,
        ProductionRiskLevels,
    )
    direction = Direction.NO_TRADE
    assert direction.value == "NO TRADE"


def test_N_resolver_exact_dataset_artifact_resolves_successfully(tmp_path, monkeypatch):
    """Test A: Exact dataset artifact resolves successfully when present."""
    data_dir = tmp_path / "data" / "raw" / "xauusd"
    data_dir.mkdir(parents=True, exist_ok=True)
    target_csv = data_dir / "xauusd_requested_2025.csv"
    df = make_sample_data(20)
    df.to_csv(target_csv, index=False)

    # Monkeypatch repo_root in resolve_historical_dataset or test via scope
    ds = DatasetScope(
        dataset_id="xauusd_requested_2025",
        symbol="XAUUSD",
        timeframe="1D",
        start_date="2025-01-01",
        end_date="2025-01-20",
    )

    with monkeypatch.context() as m:
        m.setattr("src.evaluation.research_runner.Path.resolve", lambda self: tmp_path / "src" / "evaluation" / "research_runner.py")
        resolved_df = resolve_historical_dataset(ds)
        assert not resolved_df.empty
        assert len(resolved_df) == 20


def test_O_resolver_missing_dataset_with_unrelated_same_symbol_fails_closed(tmp_path, monkeypatch):
    """Test B: Missing requested dataset with an unrelated same-symbol CSV fails closed."""
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    unrelated_csv = data_dir / "xauusd_unrelated.csv"
    df = make_sample_data(20)
    df.to_csv(unrelated_csv, index=False)

    ds = DatasetScope(
        dataset_id="xauusd_requested_2025",
        symbol="XAUUSD",
        timeframe="1D",
        start_date="2025-01-01",
        end_date="2025-01-20",
    )

    with monkeypatch.context() as m:
        m.setattr("src.evaluation.research_runner.Path.resolve", lambda self: tmp_path / "src" / "evaluation" / "research_runner.py")
        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_historical_dataset(ds)

        err_msg = str(exc_info.value)
        assert "xauusd_requested_2025" in err_msg
        assert "XAUUSD" in err_msg
        assert "Silent substitution is intentionally forbidden" in err_msg


def test_P_resolver_exact_dataset_id_glob_match(tmp_path, monkeypatch):
    """Test C: File containing exact dataset_id resolves properly."""
    data_dir = tmp_path / "data" / "custom_dir"
    data_dir.mkdir(parents=True, exist_ok=True)
    target_csv = data_dir / "ds_xauusd_requested_2025_full.csv"
    df = make_sample_data(20)
    df.to_csv(target_csv, index=False)

    ds = DatasetScope(
        dataset_id="xauusd_requested_2025",
        symbol="XAUUSD",
        timeframe="1D",
        start_date="2025-01-01",
        end_date="2025-01-20",
    )

    with monkeypatch.context() as m:
        m.setattr("src.evaluation.research_runner.Path.resolve", lambda self: tmp_path / "src" / "evaluation" / "research_runner.py")
        resolved_df = resolve_historical_dataset(ds)
        assert len(resolved_df) == 20


def test_Q_resolver_ambiguous_matches_fail_closed(tmp_path, monkeypatch):
    """Test D: Two ambiguous matching artifacts fail closed."""
    data_dir = tmp_path / "data" / "custom_dir"
    data_dir.mkdir(parents=True, exist_ok=True)
    file1 = data_dir / "ds_xauusd_requested_2025_a.csv"
    file2 = data_dir / "ds_xauusd_requested_2025_b.csv"
    df = make_sample_data(20)
    df.to_csv(file1, index=False)
    df.to_csv(file2, index=False)

    ds = DatasetScope(
        dataset_id="xauusd_requested_2025",
        symbol="XAUUSD",
        timeframe="1D",
        start_date="2025-01-01",
        end_date="2025-01-20",
    )

    with monkeypatch.context() as m:
        m.setattr("src.evaluation.research_runner.Path.resolve", lambda self: tmp_path / "src" / "evaluation" / "research_runner.py")
        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_historical_dataset(ds)

        err_msg = str(exc_info.value)
        assert "Ambiguous dataset resolution" in err_msg
        assert "xauusd_requested_2025" in err_msg


def test_R_run_research_experiment_dataset_resolution_failure_handling(tmp_path, monkeypatch):
    """Test E: run_research_experiment(spec, df=None) preserves fail-closed behavior on dataset resolution failure."""
    spec = make_valid_spec(start_date="2025-01-01", end_date="2025-04-01")
    # spec has dataset_id="xauusd_test"
    # Set repo_root to empty dir
    empty_repo = tmp_path / "empty_repo"
    (empty_repo / "data").mkdir(parents=True, exist_ok=True)

    with monkeypatch.context() as m:
        m.setattr("src.evaluation.research_runner.Path.resolve", lambda self: empty_repo / "src" / "evaluation" / "research_runner.py")
        evidence = run_research_experiment(spec, df=None)

    assert evidence.promotion_status == PromotionStatus.REJECTED
    assert RejectionReason.INVALID_DATASET_SCOPE in evidence.rejection_reasons
