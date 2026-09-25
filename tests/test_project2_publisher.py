"""Tests for Project 2 Integration Contract v1.0 Publisher Module."""
import json
import pytest
from unittest.mock import MagicMock, patch
import urllib.error

from src.integration.project2_publisher import (
    Project2Publisher,
    build_contract_v1_payload,
    _redact_secret,
)


def test_build_contract_v1_payload_structure() -> None:
    payload = build_contract_v1_payload(
        symbol="XAUUSD",
        interval="5m",
        decision="BUY",
        strategy="momentum",
        stability_score=0.85,
        signal_label="BUY",
        trend="BULLISH",
        entry_price=2000.0,
        stop_loss=1980.0,
        tp1=2020.0,
        tp2=2040.0,
        tp3=2060.0,
        take_profit=2040.0,
        risk_reward_ratio=2.0,
        timestamp="2025-01-01T12:00:00Z",
    )

    assert payload["contract_version"] == "1.0"
    assert payload["event_type"] == "TRADING_SIGNAL"
    assert "event_id" in payload
    assert payload["instrument"]["symbol"] == "XAUUSD"
    assert payload["instrument"]["interval"] == "5m"
    assert payload["signal"]["decision"] == "BUY"
    assert payload["signal"]["strategy"] == "momentum"
    assert payload["signal"]["stability_score"] == 0.85
    assert payload["trade_setup"]["entry_price"] == 2000.0
    assert payload["trade_setup"]["stop_loss"] == 1980.0
    assert payload["trade_setup"]["tp1"] == 2020.0
    assert payload["trade_setup"]["tp2"] == 2040.0
    assert payload["trade_setup"]["tp3"] == 2060.0


def test_redact_secret() -> None:
    secret = "super-secret-api-key"
    text = f"Failed request with Bearer {secret} to endpoint"
    redacted = _redact_secret(text, secret)
    assert secret not in redacted
    assert "[REDACTED]" in redacted


def test_publisher_disabled_by_default() -> None:
    publisher = Project2Publisher(enabled=False)
    payload = build_contract_v1_payload(
        symbol="XAUUSD",
        interval="5m",
        decision="BUY",
        strategy="momentum",
        stability_score=0.8,
        signal_label="BUY",
        trend="BULLISH",
        entry_price=2000.0,
        stop_loss=1980.0,
    )
    res = publisher.publish(payload)
    assert res["status"] == "SKIPPED_DISABLED"
    assert res["published"] is False


def test_publisher_skip_no_trade() -> None:
    publisher = Project2Publisher(publish_url="https://api.example.com/signals", api_key="test-key", enabled=True)
    payload = build_contract_v1_payload(
        symbol="XAUUSD",
        interval="5m",
        decision="NO TRADE",
        strategy="momentum",
        stability_score=0.8,
        signal_label="NO TRADE",
        trend="NEUTRAL",
        entry_price=None,
        stop_loss=None,
    )
    res = publisher.publish(payload, skip_if_no_trade=True)
    assert res["status"] == "SKIPPED_NO_TRADE"
    assert res["published"] is False


def test_publisher_stale_detection() -> None:
    publisher = Project2Publisher(
        publish_url="https://api.example.com/signals",
        api_key="test-key",
        enabled=True,
        max_age_seconds=60,
    )
    stale_payload = build_contract_v1_payload(
        symbol="XAUUSD",
        interval="5m",
        decision="BUY",
        strategy="momentum",
        stability_score=0.8,
        signal_label="BUY",
        trend="BULLISH",
        entry_price=2000.0,
        stop_loss=1980.0,
        timestamp="2020-01-01T00:00:00Z",
    )
    res = publisher.publish(stale_payload)
    assert res["status"] == "SKIPPED_STALE"
    assert res["published"] is False


@patch("urllib.request.urlopen")
def test_publisher_successful_delivery(mock_urlopen) -> None:
    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 200
    mock_resp.read.return_value = b'{"status": "received"}'
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    publisher = Project2Publisher(
        publish_url="https://api.example.com/signals",
        api_key="test-key",
        enabled=True,
    )
    payload = build_contract_v1_payload(
        symbol="XAUUSD",
        interval="5m",
        decision="BUY",
        strategy="momentum",
        stability_score=0.8,
        signal_label="BUY",
        trend="BULLISH",
        entry_price=2000.0,
        stop_loss=1980.0,
    )
    res = publisher.publish(payload)

    assert res["status"] == "PUBLISHED"
    assert res["published"] is True
    assert res["http_code"] == 200
    assert mock_urlopen.called


@patch("urllib.request.urlopen")
def test_publisher_retry_and_failure(mock_urlopen) -> None:
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

    publisher = Project2Publisher(
        publish_url="https://api.example.com/signals",
        api_key="secret-key",
        enabled=True,
        max_retries=2,
        backoff_factor=0.01,
    )
    payload = build_contract_v1_payload(
        symbol="XAUUSD",
        interval="5m",
        decision="BUY",
        strategy="momentum",
        stability_score=0.8,
        signal_label="BUY",
        trend="BULLISH",
        entry_price=2000.0,
        stop_loss=1980.0,
    )
    res = publisher.publish(payload)

    assert res["status"] == "FAILED"
    assert res["published"] is False
    assert res["attempts"] == 2
    assert "secret-key" not in res["error"]


def test_contract_v1_event_id_uniqueness_scope() -> None:
    """Verify that event_id changes when symbol, interval, decision, strategy, or timestamp differ."""
    base_kwargs = dict(
        symbol="XAUUSD",
        interval="5m",
        decision="BUY",
        strategy="momentum",
        stability_score=0.8,
        signal_label="BUY",
        trend="BULLISH",
        entry_price=2000.0,
        stop_loss=1980.0,
        candle_timestamp="2025-01-01T12:00:00Z",
    )

    base_payload = build_contract_v1_payload(**base_kwargs)
    base_id = base_payload["event_id"]

    # Different symbol -> different event_id
    diff_symbol = build_contract_v1_payload(**{**base_kwargs, "symbol": "BTCUSD"})
    assert diff_symbol["event_id"] != base_id

    # Different interval -> different event_id
    diff_interval = build_contract_v1_payload(**{**base_kwargs, "interval": "15m"})
    assert diff_interval["event_id"] != base_id

    # Different strategy -> different event_id
    diff_strategy = build_contract_v1_payload(**{**base_kwargs, "strategy": "mean_reversion"})
    assert diff_strategy["event_id"] != base_id

    # Different decision -> different event_id
    diff_decision = build_contract_v1_payload(**{**base_kwargs, "decision": "NO TRADE"})
    assert diff_decision["event_id"] != base_id

    # Different candle timestamp -> different event_id
    diff_ts = build_contract_v1_payload(**{**base_kwargs, "candle_timestamp": "2025-01-01T12:05:00Z"})
    assert diff_ts["event_id"] != base_id


def test_publisher_missing_api_key() -> None:
    publisher = Project2Publisher(
        publish_url="https://api.example.com/signals",
        api_key="",
        enabled=True,
    )
    payload = build_contract_v1_payload(
        symbol="XAUUSD",
        interval="5m",
        decision="BUY",
        strategy="momentum",
        stability_score=0.8,
        signal_label="BUY",
        trend="BULLISH",
        entry_price=2000.0,
        stop_loss=1980.0,
    )
    res = publisher.publish(payload)
    assert res["status"] == "FAILED"
    assert "PROJECT2_API_KEY" in res["reason"]


@patch("urllib.request.urlopen")
def test_publisher_rejected_http_401(mock_urlopen) -> None:
    err = urllib.error.HTTPError(
        url="https://api.example.com/signals",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=MagicMock(read=lambda: b'{"error": "Invalid API Key"}'),
    )
    mock_urlopen.side_effect = err

    publisher = Project2Publisher(
        publish_url="https://api.example.com/signals",
        api_key="bad-key",
        enabled=True,
    )
    payload = build_contract_v1_payload(
        symbol="XAUUSD",
        interval="5m",
        decision="BUY",
        strategy="momentum",
        stability_score=0.8,
        signal_label="BUY",
        trend="BULLISH",
        entry_price=2000.0,
        stop_loss=1980.0,
    )
    res = publisher.publish(payload)
    assert res["status"] == "REJECTED"
    assert res["http_code"] == 401
    assert res["attempts"] == 1
