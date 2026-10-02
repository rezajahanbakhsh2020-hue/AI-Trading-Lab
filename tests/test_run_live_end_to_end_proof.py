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


def test_live_end_to_end_proof_uses_canonical_runtime(
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

    recorded_config = []

    def mock_run_once(self_runtime, publish=False, persist=True):
        recorded_config.append(self_runtime.production_config)
        return {
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

    monkeypatch.setattr(
        "src.evaluation.live_execution_runtime.LiveExecutionRuntime.run_once",
        mock_run_once,
    )

    class FakeFigure:
        def write_html(
            self,
            path,
            include_plotlyjs,
            full_html,
        ):
            assert include_plotlyjs is True
            assert full_html is True
            path_obj = tmp_path / "proof.html"
            path_obj.write_text(
                "<html>LIVE END-TO-END PROOF</html>",
                encoding="utf-8",
            )

    monkeypatch.setattr(
        proof,
        "_build_chart",
        lambda data, overlay: FakeFigure(),
    )

    monkeypatch.setattr(
        proof,
        "OUTPUT_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        proof,
        "OUTPUT_JSON",
        tmp_path / "proof.json",
    )

    monkeypatch.setattr(
        proof,
        "OUTPUT_HTML",
        tmp_path / "proof.html",
    )

    result = proof.run_live_end_to_end_proof()

    # Verify ProductionRuntimeConfig passed to runtime
    assert len(recorded_config) == 1
    assert isinstance(recorded_config[0], ProductionRuntimeConfig)
    assert recorded_config[0].symbol == "XAUUSD"
    assert recorded_config[0].timeframe == "5m"

    # Verify result fields & canonical runtime lineage
    assert result["symbol"] == "XAUUSD"
    assert result["interval"] == "5m"
    assert result["stable_strategy"] == "momentum"
    assert result["decision"] == "BUY"
    assert result["trend"] == "UP"
    assert result["entry"] == 4402.0
    assert result["stop_loss"] == 4358.0
    assert result["tp1"] == 4446.0
    assert result["tp2"] == 4490.0
    assert result["tp3"] == 4534.0
    assert result["quote_stale"] is False
    assert result["end_to_end_passed"] is True

    # Lineage assertions
    assert result["candidate_id"] == "cand_proof_001"
    assert result["evidence_id"] == "ev_proof_002"
    assert result["decision_id"] == "dec_proof_789"
    assert result["runtime_authorization_fingerprint"] == "auth_fp_123456"
    assert result["canonical_live_decision_fingerprint"] == "cld_fp_890123"
    assert result["promoted_artifact_fingerprint"] == "prom_fp_234567"
    assert result["governance_decision_fingerprint"] == "gov_fp_345678"

    # Verify output files
    assert proof.OUTPUT_HTML.is_file()
    assert proof.OUTPUT_JSON.is_file()

    # Verify JSON content matches
    json_data = json.loads(proof.OUTPUT_JSON.read_text(encoding="utf-8"))
    assert json_data["candidate_id"] == "cand_proof_001"
    assert json_data["evidence_id"] == "ev_proof_002"
    assert json_data["decision_id"] == "dec_proof_789"
    assert json_data["runtime_authorization_fingerprint"] == "auth_fp_123456"
    assert json_data["canonical_live_decision_fingerprint"] == "cld_fp_890123"

    # Regression assertion: legacy load_production_selection path is NOT imported or used
    assert not hasattr(proof, "load_production_selection")
    assert not hasattr(proof, "production_live_bridge")


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
        proof.run_live_end_to_end_proof()
