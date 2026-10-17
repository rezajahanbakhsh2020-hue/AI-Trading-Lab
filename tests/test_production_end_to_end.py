from __future__ import annotations

import json

import pandas as pd
import pytest

from src.evaluation.production_end_to_end import (
    load_production_market_data,
    run_production_end_to_end,
)


@pytest.fixture(autouse=True)
def mock_candidate_resolution(monkeypatch):
    from tests.test_production_decision_integrity import make_promoted_evidence
    from src.evaluation.research_qualification import qualify_research_evidence
    from src.evaluation.research_robustness import assess_research_robustness
    from src.evaluation.live_production_decision import PromotedCandidateArtifact

    def synth_resolve(candidate_id=None, strategy_id=None, symbol="XAUUSD", timeframe="1d", base_dir=None, **kwargs):
        if symbol and symbol.upper() != "XAUUSD":
            return None
        strat = strategy_id or "momentum"
        tf = timeframe or "1d"
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
        [
            2000.0 + index * 2.0
            for index in range(rows)
        ],
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


def test_production_end_to_end_uses_candidate_lineage(
    tmp_path,
):
    # Conflicting production.json file MUST NOT affect the decision or stability authority
    result_file = tmp_path / "production.json"
    result_file.write_text(
        json.dumps(
            {
                "stable_strategy": "bogus_strategy",
                "stability_score": 0.10,
            }
        ),
        encoding="utf-8",
    )

    data = _rising_data()

    result = run_production_end_to_end(
        data,
        results_dir=tmp_path,
        symbol="XAUUSD",
        interval="1d",
        reference_now=data["timestamp"].iloc[-1] + pd.Timedelta(days=1),
        min_stability_score=0.40,
        store_path=tmp_path / "store.json",
    )

    assert result["end_to_end_ready"] is True
    assert result["production_selection"]["stable_strategy"] == "momentum"
    assert result["production_selection"]["stability_score"] == pytest.approx(0.7458282289664787)
    assert result["production_selection"]["source"] == "promoted_candidate_binding"

    assert result["decision"]["decision"] == "BUY"
    assert result["overlay"]["decision"] == "BUY"
    assert result["release_gate"]["release_ready"] is True


def test_production_end_to_end_rejects_low_min_stability(
    tmp_path,
):
    result = run_production_end_to_end(
        _rising_data(),
        symbol="XAUUSD",
        interval="1d",
        reference_now=_rising_data()["timestamp"].iloc[-1] + pd.Timedelta(days=1),
        min_stability_score=0.90,
        results_dir=tmp_path,
        store_path=tmp_path / "store.json",
    )

    assert result["end_to_end_ready"] is False
    assert result["release_gate"]["release_ready"] is False


def test_production_end_to_end_rejects_missing_candidate(
    tmp_path,
):
    with pytest.raises(ValueError, match="No promoted candidate found"):
        run_production_end_to_end(
            _rising_data(),
            symbol="EURUSD",
            interval="1d",
            research_dir=tmp_path,
        )


def test_load_production_market_data(tmp_path):
    path = tmp_path / "market.csv"

    data = _rising_data(10)
    data.to_csv(path, index=False)

    loaded = load_production_market_data(path)

    assert len(loaded) == 10
    assert {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    }.issubset(loaded.columns)

    assert loaded["timestamp"].notna().all()


def test_load_production_market_data_rejects_missing_file(
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError,
        match="Market data file not found",
    ):
        load_production_market_data(
            tmp_path / "missing.csv"
        )


def test_load_production_market_data_rejects_missing_columns(
    tmp_path,
):
    path = tmp_path / "invalid.csv"

    pd.DataFrame(
        {
            "timestamp": [1],
            "open": [2000],
            "close": [2001],
        }
    ).to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        load_production_market_data(path)


def test_load_production_market_data_rejects_empty_data(
    tmp_path,
):
    path = tmp_path / "empty.csv"

    pd.DataFrame(
        {
            "timestamp": [None],
            "open": [None],
            "high": [None],
            "low": [None],
            "close": [None],
        }
    ).to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="No valid market data",
    ):
        load_production_market_data(path)
