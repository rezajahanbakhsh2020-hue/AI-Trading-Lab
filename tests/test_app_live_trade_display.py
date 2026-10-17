import pandas as pd
import pytest

from app_live_trade_display import (
    _build_chart,
    _load_data,
)
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

    def synth_resolve(candidate_id=None, strategy_id=None, symbol="XAUUSD", timeframe="5m", base_dir=None, **kwargs):
        strat = strategy_id or "momentum"
        tf = timeframe or "5m"
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
        freq="5min",
        tz="UTC",
    )

    close = pd.Series(
        [2000.0 + index for index in range(rows)],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "isOpen": False,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        }
    )


from src.evaluation.live_runtime import build_live_runtime


def test_chart_contains_candlestick_trace():
    data = _rising_data()
    ref_now = pd.to_datetime(data["timestamp"], utc=True).iloc[-1].to_pydatetime()

    runtime_res = build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=None,
        reference_now=ref_now,
        persist=False,
    )

    overlay = build_live_trade_overlay(
        data,
        runtime_res.display,
    )

    figure = _build_chart(
        data,
        overlay,
    )

    assert len(figure.data) == 1
    assert figure.data[0].type == "candlestick"


def test_buy_chart_contains_five_trade_lines():
    data = _rising_data()
    ref_now = pd.to_datetime(data["timestamp"], utc=True).iloc[-1].to_pydatetime()

    runtime_res = build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=None,
        reference_now=ref_now,
        persist=False,
    )

    overlay = build_live_trade_overlay(
        data,
        runtime_res.display,
    )

    figure = _build_chart(
        data,
        overlay,
    )

    assert len(figure.layout.shapes) == 5


def test_no_trade_chart_has_no_trade_lines():
    data = _rising_data()

    runtime_res = build_live_runtime(
        data,
        stable_strategy="momentum",
        min_stability_score=1.01,
        persist=False,
    )

    overlay = build_live_trade_overlay(
        data,
        runtime_res.display,
    )

    figure = _build_chart(
        data,
        overlay,
    )

    assert len(figure.layout.shapes) == 0


def test_chart_uses_latest_market_timestamp():
    data = _rising_data()

    runtime_res = build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=None,
        persist=False,
    )

    overlay = build_live_trade_overlay(
        data,
        runtime_res.display,
    )

    assert overlay["timestamp"] == data["timestamp"].iloc[-1]


def test_load_data_returns_required_market_columns():
    data = _load_data()

    assert {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    }.issubset(data.columns)

    assert not data.empty


def test_load_data_has_numeric_prices():
    data = _load_data()

    for column in (
        "open",
        "high",
        "low",
        "close",
    ):
        assert pd.api.types.is_numeric_dtype(
            data[column]
        )


def test_load_data_has_valid_timestamps():
    data = _load_data()

    assert data["timestamp"].notna().all()


def test_buy_chart_level_order_is_preserved():
    data = _rising_data()
    ref_now = pd.to_datetime(data["timestamp"], utc=True).iloc[-1].to_pydatetime()

    runtime_res = build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=None,
        reference_now=ref_now,
        persist=False,
    )

    overlay = build_live_trade_overlay(
        data,
        runtime_res.display,
    )

    levels = overlay["levels"]

    assert levels["stop_loss"] < levels["entry"]
    assert levels["entry"] < levels["tp1"]
    assert levels["tp1"] < levels["tp2"]
    assert levels["tp2"] < levels["tp3"]
