from __future__ import annotations

import json
import pandas as pd
import pytest

import run_live_end_to_end_proof as proof
from src.evaluation.live_execution_runtime import ProductionRuntimeConfig


def _sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "openTime": pd.to_datetime(
                [
                    "2026-09-10 10:00:00+00:00",
                    "2026-09-10 10:05:00+00:00",
                ]
            ),
            "timestamp": pd.to_datetime(
                [
                    "2026-09-10 10:00:00+00:00",
                    "2026-09-10 10:05:00+00:00",
                ]
            ),
            "open": [4400.0, 4401.0],
            "high": [4402.0, 4403.0],
            "low": [4399.0, 4400.0],
            "close": [4401.0, 4402.0],
        }
    )


def _mock_runtime_res(publish_res: dict | None = None) -> dict:
    res = {
        "blocked": False,
        "symbol": "XAUUSD",
        "interval": "5m",
        "decision": "BUY",
        "strategy": "momentum",
        "stability_score": 0.517268,
        "candidate_id": "cand_proof_001",
        "evidence_id": "ev_proof_002",
        "runtime_authorization_fingerprint": "auth_fp_123456",
        "promoted_artifact_fingerprint": "prom_fp_234567",
        "governance_decision_fingerprint": "gov_fp_345678",
        "campaign_selection_decision_fingerprint": "camp_fp_456789",
        "context_fingerprint": "ctx_fp_567890",
        "evaluation_fingerprint": "eval_fp_678901",
        "record": {
            "symbol": "XAUUSD",
            "interval": "5m",
            "signal": 1,
            "signal_label": "BUY",
            "trend": "UP",
            "strategy": "momentum",
            "stability_score": 0.517268,
            "decision_id": "dec_proof_789",
            "canonical_live_decision_fingerprint": "cld_fp_890123",
        },
        "publication": {
            "publication_id": "pub_proof_111",
            "symbol": "XAUUSD",
            "timeframe": "5m",
            "decision": "BUY",
            "strategy_id": "momentum",
            "operational_stability_score": 0.517268,
            "entry": 4402.0,
            "stop_loss": 4358.0,
            "tp1": 4446.0,
            "tp2": 4490.0,
            "tp3": 4534.0,
        },
    }
    if publish_res is not None:
        res["publish_result"] = publish_res
    return res


def test_live_end_to_end_proof_publication_disabled(
    monkeypatch,
    tmp_path,
):
    data = _sample_data()

    monkeypatch.setattr(
        proof,
        "fetch_xauusd_ohlc",
        lambda interval, limit: data.copy(),
    )

    monkeypatch.setattr(
        proof,
        "fetch_xauusd_quote",
        lambda: {
            "symbol": "XAUUSD",
            "mid": 4402.0,
            "marketState": "OPEN",
            "stale": False,
            "quoteAgeSeconds": 0.0,
        },
    )

    recorded_publish = []

    def mock_run_once(self_runtime, publish=False, persist=True, market_data=None, market_data_loader=None):
        recorded_publish.append(publish)
        res = _mock_runtime_res()
        res["market_data"] = data.copy()
        return res

    monkeypatch.setattr(
        "src.evaluation.live_execution_runtime.LiveExecutionRuntime.run_once",
        mock_run_once,
    )

    class FakeFigure:
        def write_html(self, path, include_plotlyjs, full_html):
            path_obj = tmp_path / "proof.html"
            path_obj.write_text("<html>LIVE PROOF</html>", encoding="utf-8")

    monkeypatch.setattr(proof, "_build_chart", lambda data, overlay: FakeFigure())
    monkeypatch.setattr(proof, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(proof, "OUTPUT_JSON", tmp_path / "proof.json")
    monkeypatch.setattr(proof, "OUTPUT_HTML", tmp_path / "proof.html")

    result = proof.run_live_end_to_end_proof(publish=False)

    assert recorded_publish == [False]
    assert result["publication_requested"] is False
    assert result["published"] is False
    assert result["publication_status"] == "SKIPPED_DISABLED"
    assert result["end_to_end_passed"] is False  # P1-only proof does NOT claim P2 delivery success!

    # Verify output files
    assert proof.OUTPUT_HTML.is_file()
    assert proof.OUTPUT_JSON.is_file()

    json_data = json.loads(proof.OUTPUT_JSON.read_text(encoding="utf-8"))
    assert json_data["publication_requested"] is False
    assert json_data["published"] is False
    assert json_data["publication_status"] == "SKIPPED_DISABLED"
    assert json_data["end_to_end_passed"] is False

    # Lineage assertions
    assert json_data["candidate_id"] == "cand_proof_001"
    assert json_data["evidence_id"] == "ev_proof_002"
    assert json_data["decision_id"] == "dec_proof_789"
    assert json_data["runtime_authorization_fingerprint"] == "auth_fp_123456"
    assert json_data["canonical_live_decision_fingerprint"] == "cld_fp_890123"

    # Regression assertion: legacy load_production_selection path is NOT imported or used
    assert not hasattr(proof, "load_production_selection")
    assert not hasattr(proof, "production_live_bridge")


def test_live_end_to_end_proof_publication_enabled_success(
    monkeypatch,
    tmp_path,
):
    data = _sample_data()

    monkeypatch.setattr(proof, "fetch_xauusd_ohlc", lambda interval, limit: data.copy())
    monkeypatch.setattr(
        proof,
        "fetch_xauusd_quote",
        lambda: {"symbol": "XAUUSD", "mid": 4402.0, "marketState": "OPEN", "stale": False, "quoteAgeSeconds": 0.0},
    )

    recorded_publish = []

    def mock_run_once(self_runtime, publish=False, persist=True, market_data=None, market_data_loader=None):
        recorded_publish.append(publish)
        res = _mock_runtime_res(
            publish_res={
                "status": "PUBLISHED",
                "published": True,
                "http_code": 200,
                "event_id": "pub_proof_111",
                "publication_id": "pub_proof_111",
                "remote_event_id": "pub_proof_111",
                "delivery_status": "DELIVERED",
                "delivery_receipt_fingerprint": "del_receipt_fp_999",
            }
        )
        res["market_data"] = data.copy()
        return res

    monkeypatch.setattr("src.evaluation.live_execution_runtime.LiveExecutionRuntime.run_once", mock_run_once)

    class FakeFigure:
        def write_html(self, path, include_plotlyjs, full_html):
            path_obj = tmp_path / "proof.html"
            path_obj.write_text("<html>LIVE PROOF</html>", encoding="utf-8")

    monkeypatch.setattr(proof, "_build_chart", lambda data, overlay: FakeFigure())
    monkeypatch.setattr(proof, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(proof, "OUTPUT_JSON", tmp_path / "proof.json")
    monkeypatch.setattr(proof, "OUTPUT_HTML", tmp_path / "proof.html")

    result = proof.run_live_end_to_end_proof(publish=True)

    assert recorded_publish == [True]
    assert result["publication_requested"] is True
    assert result["published"] is True
    assert result["publication_status"] == "PUBLISHED"
    assert result["publication_id"] == "pub_proof_111"
    assert result["event_id"] == "pub_proof_111"
    assert result["delivery_receipt_fingerprint"] == "del_receipt_fp_999"
    assert result["end_to_end_passed"] is True

    json_data = json.loads(proof.OUTPUT_JSON.read_text(encoding="utf-8"))
    assert json_data["publication_requested"] is True
    assert json_data["published"] is True
    assert json_data["publication_status"] == "PUBLISHED"
    assert json_data["delivery_receipt_fingerprint"] == "del_receipt_fp_999"
    assert json_data["end_to_end_passed"] is True


@pytest.mark.parametrize(
    "status,reason",
    [
        ("FAILED", "Connection refused"),
        ("REJECTED", "Gateway explicitly rejected signal"),
        ("AUTH_FAILED", "Invalid API key"),
        ("INVALID_RESPONSE", "Response missing acknowledgement identity"),
    ],
)
def test_live_end_to_end_proof_publication_enabled_gateway_failures(
    monkeypatch,
    tmp_path,
    status,
    reason,
):
    data = _sample_data()

    monkeypatch.setattr(proof, "fetch_xauusd_ohlc", lambda interval, limit: data.copy())
    monkeypatch.setattr(
        proof,
        "fetch_xauusd_quote",
        lambda: {"symbol": "XAUUSD", "mid": 4402.0, "marketState": "OPEN", "stale": False, "quoteAgeSeconds": 0.0},
    )

    def mock_run_once(self_runtime, publish=False, persist=True, market_data=None, market_data_loader=None):
        res = _mock_runtime_res(
            publish_res={
                "status": status,
                "published": False,
                "reason": reason,
                "error": reason,
            }
        )
        res["market_data"] = data.copy()
        return res

    monkeypatch.setattr("src.evaluation.live_execution_runtime.LiveExecutionRuntime.run_once", mock_run_once)

    class FakeFigure:
        def write_html(self, path, include_plotlyjs, full_html):
            path_obj = tmp_path / "proof.html"
            path_obj.write_text("<html>LIVE PROOF</html>", encoding="utf-8")

    monkeypatch.setattr(proof, "_build_chart", lambda data, overlay: FakeFigure())
    monkeypatch.setattr(proof, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(proof, "OUTPUT_JSON", tmp_path / "proof.json")
    monkeypatch.setattr(proof, "OUTPUT_HTML", tmp_path / "proof.html")

    with pytest.raises(RuntimeError, match=f"Live end-to-end publication failed: publication_status='{status}'"):
        proof.run_live_end_to_end_proof(publish=True)

    # Ensure JSON output was written with end_to_end_passed=False and error details preserved
    assert proof.OUTPUT_JSON.is_file()
    json_data = json.loads(proof.OUTPUT_JSON.read_text(encoding="utf-8"))
    assert json_data["publication_requested"] is True
    assert json_data["published"] is False
    assert json_data["publication_status"] == status
    assert json_data["end_to_end_passed"] is False
    assert json_data["error"] == reason


def test_live_end_to_end_proof_fails_closed_when_blocked(
    monkeypatch,
    tmp_path,
):
    data = _sample_data()

    monkeypatch.setattr(
        proof,
        "fetch_xauusd_ohlc",
        lambda interval, limit: data.copy(),
    )

    monkeypatch.setattr(
        proof,
        "fetch_xauusd_quote",
        lambda: {
            "symbol": "XAUUSD",
            "mid": 4402.0,
            "marketState": "OPEN",
            "stale": False,
            "quoteAgeSeconds": 0.0,
        },
    )

    monkeypatch.setattr(
        "src.evaluation.live_execution_runtime.LiveExecutionRuntime.run_once",
        lambda self_runtime, publish=False, persist=True, market_data=None, market_data_loader=None: {
            "blocked": True,
            "reason": "PromotionUnavailable",
            "detail": "No candidate available for strategy",
        },
    )

    with pytest.raises(RuntimeError, match="Live execution runtime blocked: PromotionUnavailable"):
        proof.run_live_end_to_end_proof(publish=True)


def test_live_end_to_end_proof_lineage_passthrough(
    monkeypatch,
    tmp_path,
):
    data = _sample_data()

    monkeypatch.setattr(proof, "fetch_xauusd_ohlc", lambda interval, limit: data.copy())
    monkeypatch.setattr(
        proof,
        "fetch_xauusd_quote",
        lambda: {"symbol": "XAUUSD", "mid": 4402.0, "marketState": "OPEN", "stale": False, "quoteAgeSeconds": 0.0},
    )

    custom_runtime_res = _mock_runtime_res(
        publish_res={
            "status": "PUBLISHED",
            "published": True,
            "publication_id": "custom_pub_id",
            "delivery_receipt_fingerprint": "custom_receipt_fp",
        }
    )
    custom_runtime_res["candidate_id"] = "custom_cand_id"
    custom_runtime_res["evidence_id"] = "custom_ev_id"
    custom_runtime_res["runtime_authorization_fingerprint"] = "custom_auth_fp"
    custom_runtime_res["record"]["decision_id"] = "custom_dec_id"
    custom_runtime_res["record"]["canonical_live_decision_fingerprint"] = "custom_cld_fp"
    custom_runtime_res["market_data"] = data.copy()

    monkeypatch.setattr(
        "src.evaluation.live_execution_runtime.LiveExecutionRuntime.run_once",
        lambda self_runtime, publish=False, persist=True, market_data=None, market_data_loader=None: custom_runtime_res,
    )

    class FakeFigure:
        def write_html(self, path, include_plotlyjs, full_html):
            path_obj = tmp_path / "proof.html"
            path_obj.write_text("<html>LIVE PROOF</html>", encoding="utf-8")

    monkeypatch.setattr(proof, "_build_chart", lambda data, overlay: FakeFigure())
    monkeypatch.setattr(proof, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(proof, "OUTPUT_JSON", tmp_path / "proof.json")
    monkeypatch.setattr(proof, "OUTPUT_HTML", tmp_path / "proof.html")

    result = proof.run_live_end_to_end_proof(publish=True)

    assert result["candidate_id"] == "custom_cand_id"
    assert result["evidence_id"] == "custom_ev_id"
    assert result["decision_id"] == "custom_dec_id"
    assert result["runtime_authorization_fingerprint"] == "custom_auth_fp"
    assert result["canonical_live_decision_fingerprint"] == "custom_cld_fp"
    assert result["publication_id"] == "custom_pub_id"
    assert result["delivery_receipt_fingerprint"] == "custom_receipt_fp"


# Helper fixture setup for real candidate persistence tests
def _setup_persisted_candidate(
    tmp_path,
    cand_id="cand_real_001",
    strategy_name="momentum",
    symbol="XAUUSD",
    timeframe="5m",
    score=0.75,
):
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
    from src.evaluation.research_qualification import (
        ResearchQualificationPolicy,
        qualify_research_evidence,
    )
    from src.evaluation.research_robustness import assess_research_robustness
    from src.evaluation.research_store import save_research_candidate

    research_dir = tmp_path / "research_experiments"
    walk_forward_dir = tmp_path / "walk_forward"
    research_dir.mkdir(parents=True, exist_ok=True)
    walk_forward_dir.mkdir(parents=True, exist_ok=True)

    # Save walk forward experiment run dir in walk_forward_dir for select_production_strategy
    run_dir = walk_forward_dir / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame(
        [
            {
                "strategy": strategy_name,
                "rank": 1,
                "total_return": 0.20,
                "max_drawdown": 0.05,
                "sharpe_ratio": 2.0,
                "sortino_ratio": 2.5,
                "calmar_ratio": 4.0,
                "positive_window_rate": 0.80,
            }
        ]
    )
    report_df.to_csv(run_dir / "final_report.csv", index=False)

    eligible_df = pd.DataFrame([{"strategy": strategy_name}])
    eligible_df.to_csv(run_dir / "eligible_strategies.csv", index=False)

    meta = {
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "best_strategy": strategy_name,
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    spec = ResearchExperimentSpec(
        hypothesis="Real candidate integration hypothesis",
        methodology_version="1.0",
        strategy_name=strategy_name,
        strategy_version="1.0.0",
        dataset_scope=DatasetScope(
            dataset_id="ds_real",
            symbol=symbol,
            timeframe=timeframe,
            start_date="2026-01-01",
            end_date="2026-06-01",
        ),
        execution_assumptions=ExecutionAssumptions(
            transaction_cost=0.0001,
            slippage=0.0001,
            latency_ms=10.0,
        ),
        code_provenance=CodeProvenance(
            commit_sha="a" * 40,
            repository_status="clean",
            author="tester",
        ),
        benchmark_reference="benchmark_v1",
        parameters={"rsi_period": 14, "trend_sma_period": 50},
    )

    part_is = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2026-01-01",
        end_date="2026-03-01",
        total_return=0.15,
        max_drawdown=0.05,
        sharpe_ratio=1.8,
        observations=50,
        start_timestamp_utc="2026-01-01T00:00:00+00:00",
        end_timestamp_utc="2026-03-01T00:00:00+00:00",
    )
    part_oos = EvidencePartition(
        role=EvidencePartitionRole.OUT_OF_SAMPLE,
        start_date="2026-03-02",
        end_date="2026-06-01",
        total_return=0.12,
        max_drawdown=0.04,
        sharpe_ratio=1.6,
        observations=30,
        start_timestamp_utc="2026-03-02T00:00:00+00:00",
        end_timestamp_utc="2026-06-01T00:00:00+00:00",
    )
    part_wf = EvidencePartition(
        role=EvidencePartitionRole.WALK_FORWARD,
        start_date="2026-01-01",
        end_date="2026-06-01",
        total_return=0.10,
        max_drawdown=0.05,
        sharpe_ratio=1.5,
        observations=30,
        start_timestamp_utc="2026-01-01T00:00:00+00:00",
        end_timestamp_utc="2026-06-01T00:00:00+00:00",
    )

    evidence = ResearchEvidence(
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
        benchmark_comparison={"outperformed": True},
        promotion_status=PromotionStatus.PROMOTABLE,
        created_at_utc="2026-01-01T00:00:00+00:00",
    )

    rob_assessment = assess_research_robustness(evidence)
    gov_dec = qualify_research_evidence(
        evidence,
        policy=ResearchQualificationPolicy(max_evidence_age_days=999999),
        robustness_assessment=rob_assessment,
    )

    save_research_candidate(
        candidate_id=cand_id,
        evidence=evidence,
        operational_stability_score=score,
        base_dir=research_dir,
        governance_decision=gov_dec,
    )

    return research_dir, walk_forward_dir


def test_live_end_to_end_proof_with_real_persisted_candidate(monkeypatch, tmp_path):
    """Regression test: Proof resolves real persisted candidate from research store without mocking run_once."""
    research_dir, walk_forward_dir = _setup_persisted_candidate(
        tmp_path, cand_id="cand_real_100", strategy_name="momentum", symbol="XAUUSD", timeframe="5m", score=0.82
    )

    data = _sample_data()
    monkeypatch.setattr(proof, "fetch_xauusd_ohlc", lambda interval, limit: data.copy())
    monkeypatch.setattr(
        proof,
        "fetch_xauusd_quote",
        lambda: {"symbol": "XAUUSD", "mid": 4402.0, "marketState": "OPEN", "stale": False, "quoteAgeSeconds": 0.0},
    )

    class FakeFigure:
        def write_html(self, path, include_plotlyjs, full_html):
            path_obj = tmp_path / "proof.html"
            path_obj.write_text("<html>LIVE PROOF REAL CANDIDATE</html>", encoding="utf-8")

    monkeypatch.setattr(proof, "_build_chart", lambda data, overlay: FakeFigure())
    monkeypatch.setattr(proof, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(proof, "OUTPUT_JSON", tmp_path / "proof.json")
    monkeypatch.setattr(proof, "OUTPUT_HTML", tmp_path / "proof.html")

    # Mock live market data fetching inside runtime to use sample data
    monkeypatch.setattr(
        "src.evaluation.live_execution_runtime.load_live_market_data",
        lambda symbol, interval, limit: data.copy(),
    )

    # Isolated store path to prevent decision store replay collisions
    orig_init = proof.LiveExecutionRuntime.__init__

    def mock_init(self_runtime, *args, **kwargs):
        kwargs["store_path"] = tmp_path / "live_decision_history.json"
        kwargs["snapshot_path"] = tmp_path / "latest_execution.json"
        orig_init(self_runtime, *args, **kwargs)

    monkeypatch.setattr(proof.LiveExecutionRuntime, "__init__", mock_init)

    result = proof.run_live_end_to_end_proof(
        publish=False,
        research_dir=research_dir,
        walk_forward_dir=walk_forward_dir,
    )

    assert result["candidate_id"] == "cand_real_100"
    assert result["stable_strategy"] == "momentum"
    assert result["stability_score"] == 0.82
    assert result["evidence_id"] is not None
    assert result["runtime_authorization_fingerprint"] is not None
    assert result["promoted_artifact_fingerprint"] is not None
    assert result["governance_decision_fingerprint"] is not None

    json_data = json.loads(proof.OUTPUT_JSON.read_text(encoding="utf-8"))
    assert json_data["candidate_id"] == "cand_real_100"
    assert json_data["stable_strategy"] == "momentum"
    assert json_data["stability_score"] == 0.82


def test_live_end_to_end_proof_fails_closed_when_no_candidate(monkeypatch, tmp_path):
    """Regression test: Proof fails closed when no candidate matches the selected strategy."""
    research_dir = tmp_path / "research_experiments"
    walk_forward_dir = tmp_path / "walk_forward"
    research_dir.mkdir(parents=True, exist_ok=True)
    walk_forward_dir.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame([{"strategy": "momentum", "stability_score": 0.80}])
    report_df.to_csv(walk_forward_dir / "stability_report.csv", index=False)

    data = _sample_data()
    monkeypatch.setattr(proof, "fetch_xauusd_ohlc", lambda interval, limit: data.copy())
    monkeypatch.setattr(
        proof,
        "fetch_xauusd_quote",
        lambda: {"symbol": "XAUUSD", "mid": 4402.0, "marketState": "OPEN", "stale": False, "quoteAgeSeconds": 0.0},
    )

    monkeypatch.setattr(proof, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(proof, "OUTPUT_JSON", tmp_path / "proof.json")
    monkeypatch.setattr(proof, "OUTPUT_HTML", tmp_path / "proof.html")

    with pytest.raises(RuntimeError, match="Live execution runtime blocked: PromotionUnavailable"):
        proof.run_live_end_to_end_proof(
            publish=False,
            research_dir=research_dir,
            walk_forward_dir=walk_forward_dir,
        )


def test_proof_single_fetch_ohlc(monkeypatch, tmp_path):
    """Requirement 1: run_live_end_to_end_proof() fetches market OHLC exactly ONCE per cycle."""
    research_dir, walk_forward_dir = _setup_persisted_candidate(
        tmp_path, cand_id="cand_single_fetch", strategy_name="momentum", symbol="XAUUSD", timeframe="5m", score=0.80
    )

    data = _sample_data()
    fetch_count = 0

    def mock_fetch_ohlc(interval, limit):
        nonlocal fetch_count
        fetch_count += 1
        return data.copy()

    monkeypatch.setattr(proof, "fetch_xauusd_ohlc", mock_fetch_ohlc)
    monkeypatch.setattr(
        proof,
        "fetch_xauusd_quote",
        lambda: {"symbol": "XAUUSD", "mid": 4402.0, "marketState": "OPEN", "stale": False, "quoteAgeSeconds": 0.0},
    )

    class FakeFigure:
        def write_html(self, path, include_plotlyjs, full_html):
            (tmp_path / "proof.html").write_text("<html>PROOF</html>", encoding="utf-8")

    monkeypatch.setattr(proof, "_build_chart", lambda data, overlay: FakeFigure())
    monkeypatch.setattr(proof, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(proof, "OUTPUT_JSON", tmp_path / "proof.json")
    monkeypatch.setattr(proof, "OUTPUT_HTML", tmp_path / "proof.html")

    # Ensure runtime load_live_market_data would fail if called
    monkeypatch.setattr(
        "src.evaluation.live_execution_runtime.load_live_market_data",
        lambda *args, **kwargs: pytest.fail("Runtime load_live_market_data should NOT be called when market_data is supplied"),
    )

    orig_init = proof.LiveExecutionRuntime.__init__

    def mock_init(self_runtime, *args, **kwargs):
        kwargs["store_path"] = tmp_path / "live_decision_history.json"
        kwargs["snapshot_path"] = tmp_path / "latest_execution.json"
        orig_init(self_runtime, *args, **kwargs)

    monkeypatch.setattr(proof.LiveExecutionRuntime, "__init__", mock_init)

    proof.run_live_end_to_end_proof(
        publish=False,
        research_dir=research_dir,
        walk_forward_dir=walk_forward_dir,
    )

    assert fetch_count == 1


def test_proof_same_content_lineage_and_no_second_fetch(monkeypatch, tmp_path):
    """Requirement 2: Prove that snapshot used for chart is identical to evaluated snapshot and no 2nd fetch occurs."""
    research_dir, walk_forward_dir = _setup_persisted_candidate(
        tmp_path, cand_id="cand_lineage", strategy_name="momentum", symbol="XAUUSD", timeframe="5m", score=0.80
    )

    data_v1 = _sample_data()
    data_v2 = _sample_data().copy()
    data_v2["close"] = [9999.0, 9999.0]

    fetch_calls = 0

    def mock_fetch_ohlc(interval, limit):
        nonlocal fetch_calls
        fetch_calls += 1
        if fetch_calls == 1:
            return data_v1.copy()
        return data_v2.copy()

    monkeypatch.setattr(proof, "fetch_xauusd_ohlc", mock_fetch_ohlc)
    monkeypatch.setattr(
        proof,
        "fetch_xauusd_quote",
        lambda: {"symbol": "XAUUSD", "mid": 4402.0, "marketState": "OPEN", "stale": False, "quoteAgeSeconds": 0.0},
    )

    chart_received_data = []

    def mock_build_chart(data, overlay):
        chart_received_data.append(data.copy())

        class FakeFigure:
            def write_html(self, path, include_plotlyjs, full_html):
                (tmp_path / "proof.html").write_text("<html>PROOF</html>", encoding="utf-8")

        return FakeFigure()

    monkeypatch.setattr(proof, "_build_chart", mock_build_chart)
    monkeypatch.setattr(proof, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(proof, "OUTPUT_JSON", tmp_path / "proof.json")
    monkeypatch.setattr(proof, "OUTPUT_HTML", tmp_path / "proof.html")

    # Ensure runtime load_live_market_data would fail if called
    monkeypatch.setattr(
        "src.evaluation.live_execution_runtime.load_live_market_data",
        lambda *args, **kwargs: pytest.fail("Runtime should not fetch market data"),
    )

    orig_init = proof.LiveExecutionRuntime.__init__

    def mock_init(self_runtime, *args, **kwargs):
        kwargs["store_path"] = tmp_path / "live_decision_history.json"
        kwargs["snapshot_path"] = tmp_path / "latest_execution.json"
        orig_init(self_runtime, *args, **kwargs)

    monkeypatch.setattr(proof.LiveExecutionRuntime, "__init__", mock_init)

    res = proof.run_live_end_to_end_proof(
        publish=False,
        research_dir=research_dir,
        walk_forward_dir=walk_forward_dir,
    )

    assert fetch_calls == 1
    assert len(chart_received_data) == 1
    assert chart_received_data[0]["close"].iloc[0] == 4401.0
    assert chart_received_data[0]["close"].iloc[0] != 9999.0
    assert res["candle_count"] == len(data_v1)


def test_proof_output_consistency(monkeypatch, tmp_path):
    """Requirement 6: candle_count and evaluation_fingerprint match canonical snapshot."""
    research_dir, walk_forward_dir = _setup_persisted_candidate(
        tmp_path, cand_id="cand_output_consistency", strategy_name="momentum", symbol="XAUUSD", timeframe="5m", score=0.80
    )

    data = _sample_data()
    monkeypatch.setattr(proof, "fetch_xauusd_ohlc", lambda interval, limit: data.copy())
    monkeypatch.setattr(
        proof,
        "fetch_xauusd_quote",
        lambda: {"symbol": "XAUUSD", "mid": 4402.0, "marketState": "OPEN", "stale": False, "quoteAgeSeconds": 0.0},
    )

    class FakeFigure:
        def write_html(self, path, include_plotlyjs, full_html):
            (tmp_path / "proof.html").write_text("<html>PROOF</html>", encoding="utf-8")

    monkeypatch.setattr(proof, "_build_chart", lambda data, overlay: FakeFigure())
    monkeypatch.setattr(proof, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(proof, "OUTPUT_JSON", tmp_path / "proof.json")
    monkeypatch.setattr(proof, "OUTPUT_HTML", tmp_path / "proof.html")

    orig_init = proof.LiveExecutionRuntime.__init__

    def mock_init(self_runtime, *args, **kwargs):
        kwargs["store_path"] = tmp_path / "live_decision_history.json"
        kwargs["snapshot_path"] = tmp_path / "latest_execution.json"
        orig_init(self_runtime, *args, **kwargs)

    monkeypatch.setattr(proof.LiveExecutionRuntime, "__init__", mock_init)

    res = proof.run_live_end_to_end_proof(
        publish=False,
        research_dir=research_dir,
        walk_forward_dir=walk_forward_dir,
    )

    assert res["candle_count"] == len(data)
    assert res["evaluation_fingerprint"] is not None
    assert len(res["evaluation_fingerprint"]) == 64


def test_live_end_to_end_proof_fails_closed_on_ambiguous_candidates(monkeypatch, tmp_path):
    """Regression test: Proof fails closed when multiple candidates exist for the strategy."""
    research_dir, walk_forward_dir = _setup_persisted_candidate(
        tmp_path, cand_id="cand_ambig_1", strategy_name="momentum", symbol="XAUUSD", timeframe="5m", score=0.80
    )
    # Add second candidate for same strategy
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
    from src.evaluation.research_qualification import (
        ResearchQualificationPolicy,
        qualify_research_evidence,
    )
    from src.evaluation.research_robustness import assess_research_robustness
    from src.evaluation.research_store import save_research_candidate

    spec2 = ResearchExperimentSpec(
        hypothesis="Second candidate hypothesis",
        methodology_version="1.0",
        strategy_name="momentum",
        strategy_version="1.0.0",
        dataset_scope=DatasetScope(
            dataset_id="ds_real",
            symbol="XAUUSD",
            timeframe="5m",
            start_date="2026-01-01",
            end_date="2026-06-01",
        ),
        execution_assumptions=ExecutionAssumptions(transaction_cost=0.0001, slippage=0.0001, latency_ms=10.0),
        code_provenance=CodeProvenance(commit_sha="b" * 40, repository_status="clean", author="tester"),
        benchmark_reference="benchmark_v1",
        parameters={"rsi_period": 21, "trend_sma_period": 50},
    )
    part_is2 = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2026-01-01",
        end_date="2026-03-01",
        total_return=0.15,
        max_drawdown=0.05,
        sharpe_ratio=1.8,
        observations=50,
        start_timestamp_utc="2026-01-01T00:00:00+00:00",
        end_timestamp_utc="2026-03-01T00:00:00+00:00",
    )
    part_oos2 = EvidencePartition(
        role=EvidencePartitionRole.OUT_OF_SAMPLE,
        start_date="2026-03-02",
        end_date="2026-06-01",
        total_return=0.12,
        max_drawdown=0.04,
        sharpe_ratio=1.6,
        observations=30,
        start_timestamp_utc="2026-03-02T00:00:00+00:00",
        end_timestamp_utc="2026-06-01T00:00:00+00:00",
    )
    part_wf2 = EvidencePartition(
        role=EvidencePartitionRole.WALK_FORWARD,
        start_date="2026-01-01",
        end_date="2026-06-01",
        total_return=0.10,
        max_drawdown=0.05,
        sharpe_ratio=1.5,
        observations=30,
        start_timestamp_utc="2026-01-01T00:00:00+00:00",
        end_timestamp_utc="2026-06-01T00:00:00+00:00",
    )

    ev2 = ResearchEvidence(
        experiment_fingerprint=spec2.fingerprint,
        spec=spec2,
        partitions=(part_is2, part_oos2, part_wf2),
        robustness_verdict={
            "passed": True,
            "is_robust": True,
            "parameter_sensitivity": {"passed": True},
            "subsample_stability": {"passed": True},
            "execution_cost_stress": {"passed": True},
            "statistical_validation": {"passed": True},
            "anti_overfitting": {"passed": True},
        },
        benchmark_comparison={"outperformed": True},
        promotion_status=PromotionStatus.PROMOTABLE,
        created_at_utc="2026-01-01T00:00:00+00:00",
    )
    rob2 = assess_research_robustness(ev2)
    gov2 = qualify_research_evidence(
        ev2,
        policy=ResearchQualificationPolicy(max_evidence_age_days=999999),
        robustness_assessment=rob2,
    )
    save_research_candidate(
        candidate_id="cand_ambig_2",
        evidence=ev2,
        operational_stability_score=0.80,
        base_dir=research_dir,
        governance_decision=gov2,
    )

    data = _sample_data()
    monkeypatch.setattr(proof, "fetch_xauusd_ohlc", lambda interval, limit: data.copy())
    monkeypatch.setattr(
        proof,
        "fetch_xauusd_quote",
        lambda: {"symbol": "XAUUSD", "mid": 4402.0, "marketState": "OPEN", "stale": False, "quoteAgeSeconds": 0.0},
    )

    monkeypatch.setattr(proof, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(proof, "OUTPUT_JSON", tmp_path / "proof.json")
    monkeypatch.setattr(proof, "OUTPUT_HTML", tmp_path / "proof.html")

    with pytest.raises(RuntimeError, match="Live execution runtime blocked: PromotionIntegrityError"):
        proof.run_live_end_to_end_proof(
            publish=False,
            research_dir=research_dir,
            walk_forward_dir=walk_forward_dir,
        )


def test_live_end_to_end_proof_fails_closed_on_symbol_timeframe_mismatch(monkeypatch, tmp_path):
    """Regression test: Proof fails closed when candidate symbol/timeframe does not match runtime config."""
    # Create candidate with timeframe '15m' while runtime asks for '5m'
    research_dir, walk_forward_dir = _setup_persisted_candidate(
        tmp_path, cand_id="cand_mismatch", strategy_name="momentum", symbol="XAUUSD", timeframe="15m", score=0.80
    )

    data = _sample_data()
    monkeypatch.setattr(proof, "fetch_xauusd_ohlc", lambda interval, limit: data.copy())
    monkeypatch.setattr(
        proof,
        "fetch_xauusd_quote",
        lambda: {"symbol": "XAUUSD", "mid": 4402.0, "marketState": "OPEN", "stale": False, "quoteAgeSeconds": 0.0},
    )

    monkeypatch.setattr(proof, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(proof, "OUTPUT_JSON", tmp_path / "proof.json")
    monkeypatch.setattr(proof, "OUTPUT_HTML", tmp_path / "proof.html")

    with pytest.raises(RuntimeError, match="Live execution runtime blocked: PromotionUnavailable"):
        proof.run_live_end_to_end_proof(
            publish=False,
            research_dir=research_dir,
            walk_forward_dir=walk_forward_dir,
        )


def test_no_legacy_bypass_references_in_proof_or_runtime():
    """Regression test: Ensure load_production_selection, production_live_bridge, and select_production_strategy are not used in proof."""
    import run_live_end_to_end_proof as p
    import src.evaluation.live_execution_runtime as ler

    assert not hasattr(p, "load_production_selection")
    assert not hasattr(p, "production_live_bridge")
    assert not hasattr(p, "select_production_strategy")
    assert not hasattr(ler, "load_production_selection")
    assert not hasattr(ler, "production_live_bridge")


def test_stability_report_changes_do_not_affect_proof_resolution(monkeypatch, tmp_path):
    """Test that altering a Walk-Forward stability report does not alter proof candidate identity."""
    research_dir, walk_forward_dir = _setup_persisted_candidate(
        tmp_path, cand_id="cand_fixed_1", strategy_name="momentum", symbol="XAUUSD", timeframe="5m", score=0.82
    )

    # Put a different strategy in walk_forward_dir stability report
    report_df = pd.DataFrame([{"strategy": "fake_override_strategy", "stability_score": 0.99}])
    report_df.to_csv(walk_forward_dir / "stability_report.csv", index=False)

    data = _sample_data()
    monkeypatch.setattr(proof, "fetch_xauusd_ohlc", lambda interval, limit: data.copy())
    monkeypatch.setattr(
        proof,
        "fetch_xauusd_quote",
        lambda: {"symbol": "XAUUSD", "mid": 4402.0, "marketState": "OPEN", "stale": False, "quoteAgeSeconds": 0.0},
    )

    class FakeFigure:
        def write_html(self, path, include_plotlyjs, full_html):
            (tmp_path / "proof.html").write_text("<html>LIVE PROOF</html>", encoding="utf-8")

    monkeypatch.setattr(proof, "_build_chart", lambda data, overlay: FakeFigure())
    monkeypatch.setattr(proof, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(proof, "OUTPUT_JSON", tmp_path / "proof.json")
    monkeypatch.setattr(proof, "OUTPUT_HTML", tmp_path / "proof.html")

    monkeypatch.setattr(
        "src.evaluation.live_execution_runtime.load_live_market_data",
        lambda symbol, interval, limit: data.copy(),
    )

    orig_init = proof.LiveExecutionRuntime.__init__

    def mock_init(self_runtime, *args, **kwargs):
        kwargs["store_path"] = tmp_path / "live_decision_history.json"
        kwargs["snapshot_path"] = tmp_path / "latest_execution.json"
        orig_init(self_runtime, *args, **kwargs)

    monkeypatch.setattr(proof.LiveExecutionRuntime, "__init__", mock_init)

    result = proof.run_live_end_to_end_proof(
        publish=False,
        research_dir=research_dir,
        walk_forward_dir=walk_forward_dir,
    )

    # Resolved strategy MUST come from persisted candidate (momentum), NOT from fake_override_strategy
    assert result["candidate_id"] == "cand_fixed_1"
    assert result["stable_strategy"] == "momentum"
    assert result["stability_score"] == 0.82


def test_corrupted_candidate_binding_fails_closed(monkeypatch, tmp_path):
    """Test that corrupted candidate binding json triggers PromotionIntegrityError and fails closed."""
    research_dir, walk_forward_dir = _setup_persisted_candidate(
        tmp_path, cand_id="cand_corrupt", strategy_name="momentum", symbol="XAUUSD", timeframe="5m", score=0.82
    )

    # Corrupt the binding JSON
    from src.evaluation.research_store import _candidate_binding_path
    binding_path = _candidate_binding_path("cand_corrupt", research_dir)
    binding_path.write_text("{corrupted_json...", encoding="utf-8")

    data = _sample_data()
    monkeypatch.setattr(proof, "fetch_xauusd_ohlc", lambda interval, limit: data.copy())
    monkeypatch.setattr(
        proof,
        "fetch_xauusd_quote",
        lambda: {"symbol": "XAUUSD", "mid": 4402.0, "marketState": "OPEN", "stale": False, "quoteAgeSeconds": 0.0},
    )

    monkeypatch.setattr(proof, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(proof, "OUTPUT_JSON", tmp_path / "proof.json")
    monkeypatch.setattr(proof, "OUTPUT_HTML", tmp_path / "proof.html")

    with pytest.raises(RuntimeError, match="Live execution runtime blocked: PromotionIntegrityError"):
        proof.run_live_end_to_end_proof(
            publish=False,
            research_dir=research_dir,
            walk_forward_dir=walk_forward_dir,
        )
