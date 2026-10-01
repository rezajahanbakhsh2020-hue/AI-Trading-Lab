"""Tests proving exactly one candidate resolution and exactly one runtime authorization per execution cycle,
fresh/stale convergence, context lineage preservation, and structural AST invariants."""

from __future__ import annotations

import ast
import datetime
from pathlib import Path

import pandas as pd
import pytest

import src.evaluation.live_execution_runtime as ler_module
import src.evaluation.live_runtime as lr_module
from src.evaluation.live_execution_runtime import (
    LiveExecutionRuntime,
    ProductionRuntimeConfig,
)
from src.evaluation.live_production_decision import (
    ProductionAuthorizationReceipt,
    PromotedCandidateArtifact,
)
from src.evaluation.live_runtime import build_live_runtime
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    EvidencePartition,
    EvidencePartitionRole,
    ExecutionAssumptions,
    PromotionStatus,
    ResearchEvidence,
    ResearchExperimentSpec,
)


def _make_dummy_candidate(
    candidate_id: str = "cand_momentum_5m",
    symbol: str = "XAUUSD",
    timeframe: str = "5m",
) -> PromotedCandidateArtifact:
    spec = ResearchExperimentSpec(
        hypothesis="Test hypothesis for single authorization",
        methodology_version="1.0",
        strategy_name="momentum",
        strategy_version="1.0",
        dataset_scope=DatasetScope(
            dataset_id="ds_xauusd_5m",
            symbol=symbol,
            timeframe=timeframe,
            start_date="2024-01-01",
            end_date="2024-06-01",
        ),
        execution_assumptions=ExecutionAssumptions(
            transaction_cost=0.0001,
            slippage=0.0001,
            latency_ms=100.0,
        ),
        code_provenance=CodeProvenance(commit_sha="a1b2c3d4e5f6"),
        benchmark_reference="buy_and_hold",
        parameters={"momentum_window": 10, "stop_loss_pct": 0.01, "take_profit_pct": 0.02},
    )

    part = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2024-01-01",
        end_date="2024-06-01",
        total_return=0.1,
        max_drawdown=0.05,
        sharpe_ratio=1.5,
        observations=100,
    )

    evidence = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(part,),
        promotion_status=PromotionStatus.PROMOTABLE,
    )

    return PromotedCandidateArtifact.from_persisted_research(
        candidate_id=candidate_id,
        evidence=evidence,
        symbol=symbol,
        timeframe=timeframe,
        governance_decision_fingerprint="gov_fp_1234567890",
        campaign_selection_decision_fingerprint="cs_fp_1234567890",
        operational_stability_score=0.85,
    )


def _make_market_data(now_dt: datetime.datetime, stale: bool = False) -> pd.DataFrame:
    delta = datetime.timedelta(minutes=10 if stale else 1)
    candle_ts = (now_dt - delta).isoformat()
    return pd.DataFrame([
        {
            "openTime": candle_ts,
            "timestamp": pd.to_datetime(candle_ts),
            "open": 2000.0,
            "high": 2010.0,
            "low": 1995.0,
            "close": 2005.0,
        }
    ])


def test_fresh_execution_calls_resolution_and_authorization_exactly_once(monkeypatch, tmp_path):
    cand = _make_dummy_candidate()
    ref_now = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    market_df = _make_market_data(ref_now, stale=False)

    resolve_count = 0
    auth_count = 0
    created_context = None

    orig_auth = ler_module.authorize_production_runtime
    orig_ctx_factory = ler_module.create_authorized_runtime_context

    def mock_resolve(config):
        nonlocal resolve_count
        resolve_count += 1
        return cand

    def mock_auth(promoted_candidate, *, symbol, timeframe, now=None, authorization_policy_version="runtime_auth_v1.0"):
        nonlocal auth_count
        auth_count += 1
        return orig_auth(promoted_candidate, symbol=symbol, timeframe=timeframe, now=now, authorization_policy_version=authorization_policy_version)

    def mock_create_ctx(candidate, authorization, authorization_receipt):
        nonlocal created_context
        ctx = orig_ctx_factory(candidate, authorization, authorization_receipt)
        created_context = ctx
        assert ctx.authorization_receipt is authorization_receipt
        return ctx

    monkeypatch.setattr(ler_module, "resolve_authoritative_promoted_candidate", mock_resolve)
    monkeypatch.setattr(ler_module, "authorize_production_runtime", mock_auth)
    monkeypatch.setattr(lr_module, "authorize_production_runtime", mock_auth)
    monkeypatch.setattr(ler_module, "create_authorized_runtime_context", mock_create_ctx)
    monkeypatch.setattr(ler_module, "load_live_market_data", lambda **kw: market_df)

    p_config = ProductionRuntimeConfig(
        symbol="XAUUSD",
        timeframe="5m",
        candidate_id=cand.candidate_id,
        strategy_id=cand.strategy_name,
        strategy_version=cand.strategy_version,
    )

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "decision_history.json",
        snapshot_path=tmp_path / "latest_execution.json",
        production_config=p_config,
    )

    res = runtime.run_once(publish=False, persist=True, reference_now=ref_now)

    assert not res.get("blocked")
    assert resolve_count == 1, f"Expected resolve_authoritative_promoted_candidate == 1, got {resolve_count}"
    assert auth_count == 1, f"Expected authorize_production_runtime == 1, got {auth_count}"
    assert created_context is not None
    assert "context_fingerprint" in res
    assert res["context_fingerprint"] == created_context.context_fingerprint


def test_stale_execution_calls_resolution_and_authorization_exactly_once(monkeypatch, tmp_path):
    cand = _make_dummy_candidate()
    ref_now = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    stale_df = _make_market_data(ref_now, stale=True)

    resolve_count = 0
    auth_count = 0
    created_context = None

    orig_auth = ler_module.authorize_production_runtime
    orig_ctx_factory = ler_module.create_authorized_runtime_context

    def mock_resolve(config):
        nonlocal resolve_count
        resolve_count += 1
        return cand

    def mock_auth(promoted_candidate, *, symbol, timeframe, now=None, authorization_policy_version="runtime_auth_v1.0"):
        nonlocal auth_count
        auth_count += 1
        return orig_auth(promoted_candidate, symbol=symbol, timeframe=timeframe, now=now, authorization_policy_version=authorization_policy_version)

    def mock_create_ctx(candidate, authorization, authorization_receipt):
        nonlocal created_context
        ctx = orig_ctx_factory(candidate, authorization, authorization_receipt)
        created_context = ctx
        assert ctx.authorization_receipt is authorization_receipt
        return ctx

    monkeypatch.setattr(ler_module, "resolve_authoritative_promoted_candidate", mock_resolve)
    monkeypatch.setattr(ler_module, "authorize_production_runtime", mock_auth)
    monkeypatch.setattr(lr_module, "authorize_production_runtime", mock_auth)
    monkeypatch.setattr(ler_module, "create_authorized_runtime_context", mock_create_ctx)
    monkeypatch.setattr(ler_module, "load_live_market_data", lambda **kw: stale_df)

    p_config = ProductionRuntimeConfig(
        symbol="XAUUSD",
        timeframe="5m",
        candidate_id=cand.candidate_id,
        strategy_id=cand.strategy_name,
        strategy_version=cand.strategy_version,
    )

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "decision_history.json",
        snapshot_path=tmp_path / "latest_execution.json",
        production_config=p_config,
    )

    res = runtime.run_once(publish=False, persist=True, reference_now=ref_now)

    assert not res.get("blocked")
    assert res["decision"] == "NO TRADE"
    assert resolve_count == 1, f"Expected resolve_authoritative_promoted_candidate == 1, got {resolve_count}"
    assert auth_count == 1, f"Expected authorize_production_runtime == 1, got {auth_count}"
    assert created_context is not None
    assert "context_fingerprint" in res
    assert res["context_fingerprint"] == created_context.context_fingerprint


def test_build_live_runtime_bypasses_resolution_and_authorization_when_authorized_context_supplied(monkeypatch):
    cand = _make_dummy_candidate()
    ref_now = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    market_df = _make_market_data(ref_now, stale=False)

    auth = ler_module.authorize_production_runtime(cand, symbol="XAUUSD", timeframe="5m", now=ref_now)
    receipt = ProductionAuthorizationReceipt.from_authorization(auth)
    ctx = ler_module.create_authorized_runtime_context(cand, auth, receipt)

    assert ctx.authorization_receipt is receipt

    monkeypatch.setattr(lr_module, "resolve_promoted_candidate", pytest.fail)
    monkeypatch.setattr(lr_module, "authorize_production_runtime", pytest.fail)

    res = build_live_runtime(
        market_df,
        stable_strategy="momentum",
        stability_score=0.85,
        symbol="XAUUSD",
        interval="5m",
        persist=False,
        authorized_context=ctx,
    )

    assert res.canonical_decision is not None
    assert res.decision["context_fingerprint"] == ctx.context_fingerprint


def test_ast_structural_no_double_authorization_helpers():
    """AST test asserting build_live_runtime handles authorized_context without secondary authorization helpers."""
    target_file = Path("src/evaluation/live_runtime.py")
    tree = ast.parse(target_file.read_text(encoding="utf-8"))

    build_live_runtime_func = None
    for item in tree.body:
        if isinstance(item, ast.FunctionDef) and item.name == "build_live_runtime":
            build_live_runtime_func = item
            break

    assert build_live_runtime_func is not None, "build_live_runtime function not found in live_runtime.py"

    # Count calls to authorize_production_runtime and resolve_promoted_candidate in build_live_runtime
    auth_calls = 0
    res_calls = 0

    for node in ast.walk(build_live_runtime_func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "authorize_production_runtime":
                auth_calls += 1
            elif node.func.id == "resolve_promoted_candidate":
                res_calls += 1

    assert auth_calls <= 1, f"Found {auth_calls} authorize_production_runtime calls in build_live_runtime, expected <= 1"
    assert res_calls <= 1, f"Found {res_calls} resolve_promoted_candidate calls in build_live_runtime, expected <= 1"
