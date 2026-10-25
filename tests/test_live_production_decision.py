import pandas as pd
import pytest

from src.evaluation.live_production_decision import (
    DEFAULT_MIN_STABILITY_SCORE,
    build_live_production_decision,
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


def make_rising_market(rows: int = 80) -> pd.DataFrame:
    close = [100.0 + index for index in range(rows)]

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=rows,
                freq="5min",
                tz="UTC",
            ),
            "open": close,
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
        }
    )


def test_stable_momentum_and_uptrend_produce_buy_decision():
    data = make_rising_market()

    result = build_live_production_decision(
        data,
        reference_now=data["timestamp"].iloc[-1] + pd.Timedelta(minutes=5),
        stable_strategy="momentum",
        stability_score=0.7458282289664787,
    )

    assert result["decision"] == "BUY"
    assert "momentum_signal_1_and_trend_UP_confirmed" in result["reason"] or "live_signal_and_trend_confirmed" in result["reason"]
    assert result["stable_strategy"] == "momentum"
    assert result["stability_score"] == pytest.approx(0.7458282289664787)
    assert result["signal"] == 1
    assert result["signal_label"] == "BUY"
    assert result["trend"] == "UP"
    assert result["entry_price"] > 0
    assert result["stop_loss"] < result["entry_price"]
    assert result["take_profit"] > result["entry_price"]
    assert result["risk_reward_ratio"] == pytest.approx(2.0)


def test_stability_score_below_threshold_blocks_trade():
    data = make_rising_market()

    result = build_live_production_decision(
        data,
        reference_now=data["timestamp"].iloc[-1] + pd.Timedelta(minutes=5),
        stable_strategy="momentum",
        stability_score=0.7458282289664787,
        min_stability_score=0.80,
    )

    assert result["decision"] == "NO TRADE"
    assert result["reason"] == "stability_score_below_threshold"
    assert result["signal"] == 0
    assert result["trend"] == "UP"


def test_unsupported_stable_strategy_blocks_trade(monkeypatch):
    data = make_rising_market()

    from unittest.mock import MagicMock
    from src.evaluation.research_store import resolve_promoted_candidate
    mom_cand = resolve_promoted_candidate(strategy_id="momentum", symbol="XAUUSD", timeframe="5m")

    mock_cand = MagicMock(spec=mom_cand)
    mock_cand.candidate_id = "cand_unsupported_5m"
    mock_cand.strategy_name = "unsupported_strategy"
    mock_cand.strategy_version = "1.0"
    mock_cand.symbol = "XAUUSD"
    mock_cand.timeframe = "5m"
    mock_cand.evidence = mom_cand.evidence
    mock_cand.policy = mom_cand.policy
    mock_cand.governance_decision = None
    mock_cand.parameters = mom_cand.parameters
    mock_cand.operational_stability_score = 0.7458282289664787
    mock_cand.governance_decision_fingerprint = mom_cand.governance_decision_fingerprint
    mock_cand.artifact_fingerprint = "art_unsupported"
    mock_cand.campaign_selection_decision_fingerprint = None

    monkeypatch.setattr(
        "src.evaluation.research_store.resolve_promoted_candidate",
        lambda *args, **kwargs: mock_cand,
    )

    result = build_live_production_decision(
        data,
        reference_now=data["timestamp"].iloc[-1] + pd.Timedelta(minutes=5),
        stable_strategy="unsupported_strategy",
        stability_score=0.7458282289664787,
    )

    assert result["decision"] == "NO TRADE"
    assert (
        result["reason"]
        == "stable_strategy_not_supported_by_live_signal"
    )
    assert result["strategy_supported"] is False


def test_moving_average_supported_and_produces_buy_decision():
    data = make_rising_market()

    result = build_live_production_decision(
        data,
        reference_now=data["timestamp"].iloc[-1] + pd.Timedelta(minutes=5),
        stable_strategy="moving_average",
        stability_score=0.7458282289664787,
    )

    assert result["decision"] == "BUY"
    assert result["stable_strategy"] == "moving_average"
    assert result["strategy_supported"] is True
    assert result["signal"] == 1
    assert result["trend"] == "UP"


def test_downtrend_produces_sell_decision():
    data = make_rising_market()

    data["close"] = list(range(200, 120, -1))
    data["open"] = data["close"]
    data["high"] = data["close"] + 1.0
    data["low"] = data["close"] - 1.0

    result = build_live_production_decision(
        data,
        reference_now=data["timestamp"].iloc[-1] + pd.Timedelta(minutes=5),
        stable_strategy="momentum",
        stability_score=0.7458282289664787,
    )

    assert result["decision"] == "SELL"
    assert "momentum_signal_minus1_and_trend_DOWN_confirmed" in result["reason"]
    assert result["signal"] == -1
    assert result["trend"] == "DOWN"
    assert result["entry_price"] > 0
    assert result["stop_loss"] > result["entry_price"]
    assert result["take_profit"] < result["entry_price"]


def test_moving_average_buy_when_momentum_signal_is_zero():
    """Proves moving_average BUY is preserved even when momentum signal would be 0."""
    rows = 80
    now_dt = pd.Timestamp.now(tz="UTC")
    dates = [now_dt - pd.Timedelta(minutes=5 * i) for i in range(rows)]
    dates.reverse()

    # Create market data where fast MA (10) > slow MA (30) for last bar,
    # but close price of latest bar is <= close price 10 bars ago (momentum signal = 0).
    closes = [100.0 + idx * 0.5 for idx in range(rows)]
    closes[-1] = closes[-11] - 0.1

    data = pd.DataFrame({
        "timestamp": dates,
        "open": closes,
        "high": [c + 1.0 for c in closes],
        "low": [c - 1.0 for c in closes],
        "close": closes,
    })

    result = build_live_production_decision(
        data,
        reference_now=dates[-1] + pd.Timedelta(minutes=5),
        stable_strategy="moving_average",
        stability_score=0.7458282289664787,
        fast_window=10,
        slow_window=30,
    )

    assert result["decision"] == "BUY"
    assert result["stable_strategy"] == "moving_average"
    assert result["signal"] == 1
    assert result["signal_label"] == "BUY"


def test_moving_average_sell_when_in_downtrend():
    """Proves moving_average produces SELL when fast MA < slow MA in a downtrend."""
    rows = 80
    now_dt = pd.Timestamp.now(tz="UTC")
    dates = [now_dt - pd.Timedelta(minutes=5 * i) for i in range(rows)]
    dates.reverse()

    closes = [200.0 - idx * 0.5 for idx in range(rows)]
    closes[-1] = closes[-11] + 5.0

    data = pd.DataFrame({
        "timestamp": dates,
        "open": closes,
        "high": [c + 1.0 for c in closes],
        "low": [c - 1.0 for c in closes],
        "close": closes,
    })

    result = build_live_production_decision(
        data,
        reference_now=dates[-1] + pd.Timedelta(minutes=5),
        stable_strategy="moving_average",
        stability_score=0.7458282289664787,
    )

    assert result["decision"] == "SELL"
    assert result["stable_strategy"] == "moving_average"
    assert result["signal"] == -1
    assert result["signal_label"] == "SELL"


def test_moving_average_no_trade_when_trend_is_neutral():
    """Proves moving_average returns NO_TRADE when moving averages are equal or insufficient data."""
    rows = 80
    now_dt = pd.Timestamp.now(tz="UTC")
    dates = [now_dt - pd.Timedelta(minutes=5 * i) for i in range(rows)]
    dates.reverse()

    closes = [100.0] * rows

    data = pd.DataFrame({
        "timestamp": dates,
        "open": closes,
        "high": [c + 1.0 for c in closes],
        "low": [c - 1.0 for c in closes],
        "close": closes,
    })

    result = build_live_production_decision(
        data,
        reference_now=dates[-1] + pd.Timedelta(minutes=5),
        stable_strategy="moving_average",
        stability_score=0.7458282289664787,
    )

    assert result["decision"] == "NO TRADE"
    assert result["signal"] == 0
    assert result["signal_label"] == "NO TRADE"


def test_candidate_windows_are_authoritative(monkeypatch):
    """Proves candidate's real fast_window/slow_window parameters are consumed over caller wrapper defaults."""
    data = make_rising_market()

    from src.evaluation.research_store import resolve_promoted_candidate
    from src.evaluation.research_qualification import qualify_research_evidence
    from src.evaluation.research_robustness import assess_research_robustness
    from src.evaluation.live_production_decision import PromotedCandidateArtifact
    from dataclasses import replace

    ma_cand = resolve_promoted_candidate(strategy_id="moving_average", symbol="XAUUSD", timeframe="5m")
    custom_params = {"fast_window": 10, "slow_window": 30, "stop_loss_pct": 0.01, "take_profit_pct": 0.02}
    custom_spec = replace(ma_cand.evidence.spec, parameters=custom_params)
    custom_evidence = replace(ma_cand.evidence, spec=custom_spec, experiment_fingerprint=custom_spec.fingerprint)
    rob = assess_research_robustness(custom_evidence)
    gov = qualify_research_evidence(custom_evidence, robustness_assessment=rob)
    custom_cand = PromotedCandidateArtifact(
        candidate_id=ma_cand.candidate_id,
        strategy_name=ma_cand.strategy_name,
        strategy_version=ma_cand.strategy_version,
        evidence=custom_evidence,
        symbol=ma_cand.symbol,
        timeframe=ma_cand.timeframe,
        operational_stability_score=ma_cand.operational_stability_score,
        governance_decision=gov,
        governance_decision_fingerprint=gov.decision_fingerprint,
        parameters=custom_params,
    )

    monkeypatch.setattr(
        "src.evaluation.research_store.resolve_promoted_candidate",
        lambda *args, **kwargs: custom_cand,
    )

    result = build_live_production_decision(
        data,
        reference_now=data["timestamp"].iloc[-1] + pd.Timedelta(minutes=5),
        stable_strategy="moving_average",
        stability_score=0.7458282289664787,
        fast_window=20,
        slow_window=50,
    )

    assert result["fast_window"] == 10
    assert result["slow_window"] == 30


@pytest.mark.parametrize(
    ("fast", "slow"),
    [
        (True, 50),
        (20, True),
        (20.5, 50),
        (20, 50.5),
        (float("nan"), 50),
        (20, float("nan")),
        (float("inf"), 50),
        (20, float("inf")),
        (0, 50),
        (20, 0),
        (-10, 50),
        (20, -50),
        (50, 20),
        (20, 20),
        (None, 50),
        (20, None),
        ("20", 50),
    ],
)
def test_strict_window_validation_parameterized(fast, slow):
    from src.evaluation.live_production_decision import validate_window_pair

    with pytest.raises(ValueError):
        validate_window_pair(fast, slow)


def test_no_trade_signal_is_preserved_for_flat_prices():
    data = make_rising_market()

    data["close"] = [100.0] * len(data)
    data["open"] = data["close"]
    data["high"] = data["close"] + 1.0
    data["low"] = data["close"] - 1.0

    result = build_live_production_decision(
        data,
        reference_now=data["timestamp"].iloc[-1] + pd.Timedelta(minutes=5),
        stable_strategy="momentum",
        stability_score=0.7458282289664787,
    )

    assert result["decision"] == "NO TRADE"
    assert result["signal"] == 0
    assert result["signal_label"] == "NO TRADE"
    assert result["decision"] != "SELL"
    assert result["stop_loss"] is None
    assert result["take_profit"] is None


def test_default_threshold_matches_current_stability_gate():
    assert DEFAULT_MIN_STABILITY_SCORE == 0.50


def test_invalid_stability_score_is_rejected():
    data = make_rising_market()

    with pytest.raises(ValueError, match="between 0 and 1"):
        build_live_production_decision(
            data,
            stable_strategy="momentum",
            stability_score=1.5,
        )


def test_empty_strategy_is_rejected():
    data = make_rising_market()

    with pytest.raises(ValueError, match="must not be empty"):
        build_live_production_decision(
            data,
            stable_strategy="",
            stability_score=0.80,
        )
