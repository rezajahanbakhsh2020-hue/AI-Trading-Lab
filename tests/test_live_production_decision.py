import pandas as pd
import pytest

from src.evaluation.live_production_decision import (
    DEFAULT_MIN_STABILITY_SCORE,
    build_live_production_decision,
)


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
    assert (
        result["reason"]
        == "stable_strategy_live_signal_and_trend_confirmed"
    )
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
    assert result["signal"] == 1
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
    mock_cand.governance_decision_fingerprint = "gov_dec_unsupported"
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


def test_downtrend_blocks_buy_decision():
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

    assert result["decision"] == "NO TRADE"
    assert result["reason"] == "trend_not_confirmed"
    assert result["signal"] == 0
    assert result["trend"] == "DOWN"


def test_no_trade_signal_is_preserved_without_inventing_sell():
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
