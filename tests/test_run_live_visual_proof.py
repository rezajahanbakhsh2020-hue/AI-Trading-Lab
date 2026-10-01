from pathlib import Path

import pandas as pd
import pytest

import run_live_visual_proof
from src.evaluation.live_production_decision import (
    PromotedCandidateArtifact,
    ProductionRuntimeAuthorizationError,
    PromotionStatus,
)
from tests.test_production_decision_integrity import make_promoted_evidence


def _make_mock_ohlc() -> pd.DataFrame:
    closes = [2000.0 + float(i) for i in range(50)]
    return pd.DataFrame(
        {
            "openTime": pd.date_range("2026-01-01", periods=len(closes), freq="5min"),
            "open": closes,
            "high": [c + 2.0 for c in closes],
            "low": [c - 2.0 for c in closes],
            "close": closes,
            "volume": [100.0] * len(closes),
            "tickVolume": [100.0] * len(closes),
            "isOpen": [False] * len(closes),
        }
    )


def _make_mock_quote() -> dict:
    return {
        "symbol": "XAUUSD",
        "mid": 2049.0,
        "bid": 2048.5,
        "ask": 2049.5,
        "marketState": "OPEN",
        "quoteAgeSeconds": 5.0,
        "stale": False,
    }


def test_live_visual_proof_creates_real_html(monkeypatch, tmp_path):
    ev = make_promoted_evidence(PromotionStatus.PROMOTABLE)
    promoted = PromotedCandidateArtifact(
        candidate_id="cand_vp_01",
        strategy_name="momentum",
        strategy_version="1.0",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        operational_stability_score=0.85,
        governance_decision_fingerprint="gov_fp_vp_01",
    )

    monkeypatch.setattr(
        "run_live_visual_proof.resolve_authoritative_promoted_candidate",
        lambda config: promoted,
    )
    monkeypatch.setattr(
        "run_live_visual_proof.fetch_xauusd_ohlc",
        lambda **kwargs: _make_mock_ohlc(),
    )
    monkeypatch.setattr(
        "run_live_visual_proof.fetch_xauusd_quote",
        lambda **kwargs: _make_mock_quote(),
    )
    monkeypatch.setattr(
        "run_live_visual_proof._load_live_production_selection",
        lambda: {
            "candidate_id": "cand_vp_01",
            "stable_strategy": "momentum",
            "stability_score": 0.85,
            "source_path": "results/production/latest.json",
        },
    )
    test_out = tmp_path / "live_proof_visual.html"
    monkeypatch.setattr(
        "run_live_visual_proof.OUTPUT_PATH",
        test_out,
    )
    monkeypatch.setattr(
        "src.evaluation.live_runtime.DEFAULT_STORE_PATH",
        tmp_path / "decision_history.json",
    )

    result = run_live_visual_proof.run_live_visual_proof()

    assert result["symbol"] == "XAUUSD"
    assert result["interval"] == "5m"

    assert result["decision"] in {
        "BUY",
        "NO TRADE",
    }

    assert result["signal"] in {0, 1}

    assert result["signal_label"] in {
        "BUY",
        "NO TRADE",
    }

    assert result["trend"] in {
        "UP",
        "DOWN",
        "INSUFFICIENT DATA",
        "NEUTRAL",
    }

    assert result["stable_strategy"] == "momentum"
    assert 0.0 <= result["stability_score"] <= 1.0

    assert result["candle_count"] > 0
    assert result["timestamp"]
    assert result["market_state"]
    assert result["authorization_fingerprint"] is not None
    assert result["governance_decision_fingerprint"] == "gov_fp_vp_01"

    assert isinstance(
        result["quote_stale"],
        bool,
    )

    assert isinstance(
        result["actionable"],
        bool,
    )

    assert result["production_source"] == "results/production/latest.json"

    assert result["human_text"]

    output_path = Path(
        result["output_path"]
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_live_visual_proof_html_contains_visual_elements():
    if not run_live_visual_proof.OUTPUT_PATH.exists():
        pytest.skip(
            "Live visual proof HTML does not exist yet."
        )

    html = run_live_visual_proof.OUTPUT_PATH.read_text(
        encoding="utf-8"
    )

    assert "<html" in html.lower()
    assert "plotly" in html.lower()
    assert "Fast MA" in html or "fast ma" in html.lower() or "close" in html.lower()


def test_live_visual_proof_uses_production_selection(monkeypatch, tmp_path):
    ev = make_promoted_evidence(PromotionStatus.PROMOTABLE)
    promoted = PromotedCandidateArtifact(
        candidate_id="cand_vp_02",
        strategy_name="momentum",
        strategy_version="1.0",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        operational_stability_score=0.85,
        governance_decision_fingerprint="gov_fp_vp_02",
    )

    monkeypatch.setattr(
        "run_live_visual_proof.resolve_authoritative_promoted_candidate",
        lambda config: promoted,
    )
    monkeypatch.setattr(
        "run_live_visual_proof.fetch_xauusd_ohlc",
        lambda **kwargs: _make_mock_ohlc(),
    )
    monkeypatch.setattr(
        "run_live_visual_proof.fetch_xauusd_quote",
        lambda **kwargs: _make_mock_quote(),
    )
    monkeypatch.setattr(
        "run_live_visual_proof._load_live_production_selection",
        lambda: {
            "candidate_id": "cand_vp_02",
            "stable_strategy": "momentum",
            "stability_score": 0.85,
            "source_path": "results/production/latest.json",
        },
    )
    test_out = tmp_path / "live_proof_visual.html"
    monkeypatch.setattr(
        "run_live_visual_proof.OUTPUT_PATH",
        test_out,
    )
    monkeypatch.setattr(
        "src.evaluation.live_runtime.DEFAULT_STORE_PATH",
        tmp_path / "decision_history.json",
    )

    result = run_live_visual_proof.run_live_visual_proof()

    assert result["stable_strategy"] == "momentum"
    assert result["production_source"] == "results/production/latest.json"
    assert result["stability_score"] == 0.85


def test_live_visual_proof_fails_when_production_selection_missing():
    with pytest.raises(FileNotFoundError):
        run_live_visual_proof.run_live_visual_proof()


def test_live_visual_proof_fails_when_unauthorized(monkeypatch):
    monkeypatch.setattr(
        "run_live_visual_proof._load_live_production_selection",
        lambda: {
            "candidate_id": "cand_unauth",
            "stable_strategy": "momentum",
            "stability_score": 0.85,
            "source_path": "results/production/latest.json",
        },
    )

    with pytest.raises(ProductionRuntimeAuthorizationError):
        run_live_visual_proof.run_live_visual_proof()
