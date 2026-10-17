from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.mtf_intelligence import CanonicalTimeframe
from src.evaluation.live_production_decision import (
    build_live_production_decision,
)
from src.evaluation.live_runtime import build_live_runtime
from src.evaluation.live_trade_display import (
    build_live_trade_display,
)
from src.visualization.live_trade_overlay import (
    build_live_trade_overlay,
)


@pytest.fixture(autouse=True)
def mock_candidate_resolution(monkeypatch):
    from tests.test_production_decision_integrity import make_promoted_evidence
    from src.evaluation.research_qualification import qualify_research_evidence
    from src.evaluation.research_robustness import assess_research_robustness
    from src.evaluation.live_production_decision import PromotedCandidateArtifact

    def synth_resolve(candidate_id=None, strategy_id=None, symbol="XAUUSD", timeframe="1d", base_dir=None, **kwargs):
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


def _realistic_rising_market(rows: int = 80) -> pd.DataFrame:
    timestamps = pd.date_range(
        pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=rows),
        periods=rows,
        freq="1D",
    )

    close = pd.Series(
        [2000.0 + index * 2.0 for index in range(rows)],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "isOpen": False,
            "open": close - 1.0,
            "high": close + 3.0,
            "low": close - 2.0,
            "close": close,
        }
    )


def test_end_to_end_live_proof():
    data = _realistic_rising_market()

    ref_now = pd.to_datetime(data["timestamp"], utc=True).iloc[-1].to_pydatetime()
    decision = build_live_production_decision(
        data,
        stable_strategy="momentum",
        stability_score=0.7458282289664787,
        symbol="XAUUSD",
        interval="1d",
    )

    assert decision["decision"] == "BUY"
    assert decision["trend"] == "UP"
    assert decision["signal_label"] == "BUY"
    assert decision["entry_price"] is not None
    assert decision["stop_loss"] is not None
    assert decision["take_profit"] is not None

    runtime = build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=None,
        symbol="XAUUSD",
        interval="1d",
        reference_now=ref_now,
    )

    display = build_live_trade_display(
        data,
        canonical_decision=runtime.canonical_decision,
        stable_strategy="momentum",
        stability_score=None,
        symbol="XAUUSD",
        interval="1d",
    )

    assert display["decision"] == "BUY"
    assert display["entry_price"] == pytest.approx(
        decision["entry_price"]
    )
    assert display["stop_loss"] == pytest.approx(
        decision["stop_loss"]
    )

    assert runtime.decision["decision"] == "BUY"
    assert runtime.display["decision"] == "BUY"

    overlay = build_live_trade_overlay(
        data,
        runtime.display,
    )

    assert overlay["symbol"] == "XAUUSD"
    assert CanonicalTimeframe.from_str(overlay["interval"]) == CanonicalTimeframe.ONE_DAY
    assert overlay["decision"] == "BUY"
    assert overlay["trend"] == "UP"
    assert overlay["stable_strategy"] == "momentum"

    levels = overlay["levels"]

    assert levels["stop_loss"] < levels["entry"]
    assert levels["entry"] < levels["tp1"]
    assert levels["tp1"] < levels["tp2"]
    assert levels["tp2"] < levels["tp3"]

    assert len(overlay["lines"]) == 5
    assert all(
        line["visible"]
        for line in overlay["lines"]
    )


def test_end_to_end_no_trade_proof():
    data = _realistic_rising_market()
    ref_now = pd.to_datetime(data["timestamp"], utc=True).iloc[-1].to_pydatetime()

    runtime = build_live_runtime(
        data,
        stable_strategy="momentum",
        min_stability_score=0.90,
        symbol="XAUUSD",
        interval="1d",
        reference_now=ref_now,
    )

    assert runtime.decision["decision"] == "NO TRADE"
    assert runtime.display["decision"] == "NO TRADE"

    overlay = build_live_trade_overlay(
        data,
        runtime.display,
    )

    assert overlay["decision"] == "NO TRADE"
    assert all(
        value is None
        for value in overlay["levels"].values()
    )
    assert all(
        not line["visible"]
        for line in overlay["lines"]
    )


def test_end_to_end_preserves_latest_market_timestamp():
    data = _realistic_rising_market()
    ref_now = pd.to_datetime(data["timestamp"], utc=True).iloc[-1].to_pydatetime()

    runtime = build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=None,
        symbol="XAUUSD",
        interval="1d",
        reference_now=ref_now,
    )

    overlay = build_live_trade_overlay(
        data,
        runtime.display,
    )

    assert overlay["timestamp"] == data["timestamp"].iloc[-1]
