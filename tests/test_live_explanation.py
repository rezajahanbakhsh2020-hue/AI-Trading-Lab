from src.evaluation.live_explanation import (
    build_live_explanation,
    validate_live_explanation,
)


def buy_snapshot() -> dict:
    return {
        "signal": 1,
        "signal_label": "BUY",
        "trend": "UP",
        "momentum": 0.012,
        "entry_price": 4400.0,
        "stop_loss": 4356.0,
        "take_profit": 4488.0,
        "risk_reward_ratio": 2.0,
        "market_state": "OPEN",
        "quote_stale": False,
    }


def no_trade_snapshot() -> dict:
    return {
        "signal": 0,
        "signal_label": "NO TRADE",
        "trend": "UP",
        "momentum": -0.005,
        "entry_price": 4400.0,
        "stop_loss": None,
        "take_profit": None,
        "risk_reward_ratio": None,
        "market_state": "OPEN",
        "quote_stale": False,
    }


def test_buy_explanation():
    explanation = build_live_explanation(
        buy_snapshot()
    )

    assert explanation["action"] == "BUY"
    assert explanation["signal_label"] == "BUY"
    assert explanation["trend"] == "UP"
    assert explanation["entry_price"] == 4400.0
    assert explanation["stop_loss"] == 4356.0
    assert explanation["take_profit"] == 4488.0
    assert len(explanation["reasons"]) >= 2
    assert "BUY" in explanation["human_text"]


def test_no_trade_explanation():
    explanation = build_live_explanation(
        no_trade_snapshot()
    )

    assert explanation["action"] == "NO TRADE"
    assert explanation["signal_label"] == "NO TRADE"
    assert explanation["stop_loss"] is None
    assert explanation["take_profit"] is None
    assert len(explanation["reasons"]) >= 1
    assert "No trade" in explanation["human_text"]


def test_stale_quote_is_explained():
    snapshot = buy_snapshot()
    snapshot["quote_stale"] = True

    explanation = build_live_explanation(
        snapshot
    )

    assert any(
        "stale" in reason.lower()
        for reason in explanation["reasons"]
    )


def test_closed_market_is_explained():
    snapshot = buy_snapshot()
    snapshot["market_state"] = "CLOSED"

    explanation = build_live_explanation(
        snapshot
    )

    assert any(
        "CLOSED" in reason
        for reason in explanation["reasons"]
    )


def test_downtrend_is_explained_for_no_trade():
    snapshot = no_trade_snapshot()
    snapshot["trend"] = "DOWN"

    explanation = build_live_explanation(
        snapshot
    )

    assert any(
        "DOWN" in reason
        for reason in explanation["reasons"]
    )


def test_explanation_validation():
    explanation = build_live_explanation(
        buy_snapshot()
    )

    assert validate_live_explanation(
        explanation
    ) is True


def test_invalid_explanation_is_rejected():
    assert validate_live_explanation({}) is False


def test_non_dictionary_snapshot_is_rejected():
    try:
        build_live_explanation([])
        assert False
    except ValueError:
        assert True
