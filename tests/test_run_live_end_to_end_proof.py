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

    def mock_run_once(self_runtime, publish=False, persist=True):
        recorded_publish.append(publish)
        return _mock_runtime_res()

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

    def mock_run_once(self_runtime, publish=False, persist=True):
        recorded_publish.append(publish)
        return _mock_runtime_res(
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

    def mock_run_once(self_runtime, publish=False, persist=True):
        return _mock_runtime_res(
            publish_res={
                "status": status,
                "published": False,
                "reason": reason,
                "error": reason,
            }
        )

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
        lambda self_runtime, publish=False, persist=True: {
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

    monkeypatch.setattr(
        "src.evaluation.live_execution_runtime.LiveExecutionRuntime.run_once",
        lambda self_runtime, publish=False, persist=True: custom_runtime_res,
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
