from __future__ import annotations

import csv
import json
from io import StringIO

import pytest

from src.evaluation.live_decision_export import (
    export_live_decisions,
    export_live_decisions_csv,
    export_live_decisions_json,
)


def _record(
    timestamp: str = "2026-09-10T05:45:00+00:00",
    signal_label: str = "BUY",
) -> dict:
    return {
        "timestamp": timestamp,
        "symbol": "XAUUSD",
        "interval": "5m",
        "signal": 1 if signal_label == "BUY" else 0,
        "signal_label": signal_label,
        "trend": "UP",
        "strategy": "momentum",
        "entry_price": 4429.802,
        "stop_loss": 4385.50398,
        "take_profit": 4518.39804,
        "risk_reward_ratio": 2.0,
        "stability_score": 0.65,
        "market_state": "open",
        "quote_age_seconds": 0,
        "quote_stale": False,
        "candle_count": 201,
    }


def test_export_live_decisions_json_returns_valid_json():
    payload = export_live_decisions_json(
        [_record()]
    )

    data = json.loads(payload)

    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["symbol"] == "XAUUSD"
    assert data[0]["signal_label"] == "BUY"
    assert data[0]["entry_price"] == 4429.802


def test_export_live_decisions_json_preserves_order():
    payload = export_live_decisions_json(
        [
            _record("2026-09-10T05:45:00+00:00"),
            _record(
                "2026-09-10T05:50:00+00:00",
                "NO TRADE",
            ),
        ]
    )

    data = json.loads(payload)

    assert [
        item["timestamp"]
        for item in data
    ] == [
        "2026-09-10T05:45:00+00:00",
        "2026-09-10T05:50:00+00:00",
    ]


def test_export_live_decisions_json_preserves_missing_optional_values():
    record = _record()
    record["stability_score"] = None

    payload = export_live_decisions_json([record])
    data = json.loads(payload)

    assert data[0]["stability_score"] is None


def test_export_live_decisions_csv_returns_valid_csv():
    payload = export_live_decisions_csv(
        [_record()]
    )

    rows = list(csv.DictReader(StringIO(payload)))

    assert len(rows) == 1
    assert rows[0]["symbol"] == "XAUUSD"
    assert rows[0]["interval"] == "5m"
    assert rows[0]["signal_label"] == "BUY"


def test_export_live_decisions_csv_contains_expected_headers():
    payload = export_live_decisions_csv(
        [_record()]
    )

    reader = csv.reader(StringIO(payload))
    headers = next(reader)

    assert headers == [
        "timestamp",
        "symbol",
        "interval",
        "signal",
        "signal_label",
        "trend",
        "strategy",
        "entry_price",
        "stop_loss",
        "take_profit",
        "risk_reward_ratio",
        "stability_score",
        "market_state",
        "quote_age_seconds",
        "quote_stale",
        "candle_count",
    ]


def test_export_live_decisions_supports_json():
    payload = export_live_decisions(
        [_record()],
        format="json",
    )

    data = json.loads(payload)

    assert data[0]["symbol"] == "XAUUSD"


def test_export_live_decisions_supports_csv():
    payload = export_live_decisions(
        [_record()],
        format="csv",
    )

    rows = list(csv.DictReader(StringIO(payload)))

    assert rows[0]["symbol"] == "XAUUSD"


def test_export_live_decisions_format_is_case_insensitive():
    payload = export_live_decisions(
        [_record()],
        format="JSON",
    )

    data = json.loads(payload)

    assert data[0]["signal_label"] == "BUY"


def test_export_live_decisions_rejects_unsupported_format():
    with pytest.raises(
        ValueError,
        match="unsupported format",
    ):
        export_live_decisions(
            [_record()],
            format="xml",
        )


def test_export_live_decisions_rejects_invalid_history():
    with pytest.raises(TypeError):
        export_live_decisions_json("invalid")


def test_export_live_decisions_rejects_invalid_record():
    with pytest.raises(TypeError):
        export_live_decisions_csv([[]])


def test_export_does_not_recalculate_trading_values():
    record = _record()
    record["entry_price"] = 9999.0
    record["stop_loss"] = 8888.0
    record["take_profit"] = 7777.0

    payload = export_live_decisions_json([record])
    data = json.loads(payload)

    assert data[0]["entry_price"] == 9999.0
    assert data[0]["stop_loss"] == 8888.0
    assert data[0]["take_profit"] == 7777.0
