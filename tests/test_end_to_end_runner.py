from __future__ import annotations

import pandas as pd

from src.evaluation.end_to_end_runner import run_end_to_end
from src.evaluation.mtf_intelligence import CanonicalTimeframe


import pytest


@pytest.fixture(autouse=True)
def mock_candidate_resolution(monkeypatch):
    from tests.test_production_decision_integrity import make_promoted_evidence
    from src.evaluation.research_qualification import qualify_research_evidence
    from src.evaluation.research_robustness import assess_research_robustness
    from src.evaluation.live_production_decision import PromotedCandidateArtifact

    def synth_resolve(candidate_id=None, strategy_id=None, symbol="XAUUSD", timeframe="1D", base_dir=None, **kwargs):
        strat = strategy_id or "momentum"
        tf = timeframe or "1D"
        cid = candidate_id or f"cand_{strat}_{tf}"
        ev = make_promoted_evidence(timeframe=tf, symbol=symbol, strategy_name=strat)
        rob = assess_research_robustness(ev)
        gov = qualify_research_evidence(ev, robustness_assessment=rob)
        params = dict(ev.spec.parameters)
        return PromotedCandidateArtifact(
            candidate_id=cid,
            strategy_name=strat,
            strategy_version="1.0",
            evidence=ev,
            symbol=symbol,
            timeframe=tf,
            operational_stability_score=0.7458282289664787,
            parameters=params,
            governance_decision=gov,
            governance_decision_fingerprint=gov.decision_fingerprint,
        )

    monkeypatch.setattr("src.evaluation.research_store.resolve_promoted_candidate", synth_resolve)
    monkeypatch.setattr("src.evaluation.live_runtime.resolve_promoted_candidate", synth_resolve)


def _rising_data(rows: int = 80) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="1D",
        tz="UTC",
    )

    close = pd.Series(
        [2000.0 + index * 2.0 for index in range(rows)],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 1.0,
            "high": close + 3.0,
            "low": close - 2.0,
            "close": close,
        }
    )


def test_end_to_end_runner_produces_complete_buy_result(tmp_path):
    data = _rising_data()
    result = run_end_to_end(
        data,
        stable_strategy="momentum",
        stability_score=None,
        reference_now=data["timestamp"].iloc[-1] + pd.Timedelta(days=1),
        store_path=tmp_path / "store.json",
    )

    assert result["end_to_end_ready"] is True
    assert result["decision"]["decision"] == "BUY"
    assert result["display"]["decision"] == "BUY"
    assert result["overlay"]["decision"] == "BUY"
    assert result["release_gate"]["release_ready"] is True


def test_end_to_end_runner_contains_trade_levels(tmp_path):
    data = _rising_data()
    result = run_end_to_end(
        data,
        stable_strategy="momentum",
        stability_score=None,
        reference_now=data["timestamp"].iloc[-1] + pd.Timedelta(days=1),
        store_path=tmp_path / "store.json",
    )

    levels = result["overlay"]["levels"]

    assert levels["entry"] is not None
    assert levels["stop_loss"] is not None
    assert levels["tp1"] is not None
    assert levels["tp2"] is not None
    assert levels["tp3"] is not None

    assert levels["stop_loss"] < levels["entry"]
    assert levels["entry"] < levels["tp1"]
    assert levels["tp1"] < levels["tp2"]
    assert levels["tp2"] < levels["tp3"]


def test_end_to_end_runner_rejects_low_stability_for_release(tmp_path):
    result = run_end_to_end(
        _rising_data(),
        stable_strategy="momentum",
        stability_score=None,
        reference_now=_rising_data()["timestamp"].iloc[-1] + pd.Timedelta(days=1),
        min_stability_score=0.90,
        store_path=tmp_path / "store.json",
    )

    assert result["end_to_end_ready"] is False
    assert result["release_gate"]["release_ready"] is False


def test_end_to_end_runner_preserves_symbol_and_interval(tmp_path):
    result = run_end_to_end(
        _rising_data(),
        stable_strategy="momentum",
        stability_score=None,
        symbol="XAUUSD",
        interval="1d",
        reference_now=_rising_data()["timestamp"].iloc[-1] + pd.Timedelta(days=1),
        store_path=tmp_path / "store.json",
    )

    assert result["decision"]["symbol"] == "XAUUSD"
    assert CanonicalTimeframe.from_str(result["decision"]["interval"]) == CanonicalTimeframe.ONE_DAY
    assert result["display"]["symbol"] == "XAUUSD"
    assert CanonicalTimeframe.from_str(result["display"]["interval"]) == CanonicalTimeframe.ONE_DAY
    assert result["overlay"]["symbol"] == "XAUUSD"
    assert CanonicalTimeframe.from_str(result["overlay"]["interval"]) == CanonicalTimeframe.ONE_DAY


def test_end_to_end_runner_exposes_all_pipeline_layers(tmp_path):
    result = run_end_to_end(
        _rising_data(),
        stable_strategy="momentum",
        stability_score=None,
        reference_now=_rising_data()["timestamp"].iloc[-1] + pd.Timedelta(days=1),
        store_path=tmp_path / "store.json",
    )

    assert set(result) == {
        "decision",
        "display",
        "overlay",
        "release_gate",
        "end_to_end_ready",
    }

    assert result["overlay"]["lines"]
    assert len(result["overlay"]["lines"]) == 5
