import json

import pandas as pd
import pytest

from app_live_proof_history import (
    append_history_record,
    build_history_chart,
    build_live_history_record,
    load_history,
    render_history_table,
    validate_history_record,
)


def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "openTime": pd.date_range(
                "2026-01-01",
                periods=60,
                freq="5min",
                tz="UTC",
            ),
            "open": [2500.0 + i for i in range(60)],
            "high": [2501.0 + i for i in range(60)],
            "low": [2499.0 + i for i in range(60)],
            "close": [2500.5 + i for i in range(60)],
        }
    )


def sample_quote() -> dict:
    return {
        "symbol": "XAUUSD",
        "mid": 2559.5,
        "bid": 2559.4,
        "ask": 2559.6,
        "marketState": "open",
        "stale": False,
        "quoteAgeSeconds": 0,
    }


def test_build_live_history_record_contains_core_fields():
    record = build_live_history_record(
        sample_data(),
        sample_quote(),
    )

    assert record["symbol"] == "XAUUSD"
    assert record["interval"] == "5m"
    assert record["signal_label"] in {
        "BUY",
        "NO TRADE",
    }
    assert record["trend"] in {
        "UP",
        "DOWN",
        "INSUFFICIENT DATA",
    }
    assert record["mid"] == 2559.5
    assert record["market_state"] == "OPEN"


def test_history_record_is_valid():
    record = build_live_history_record(
        sample_data(),
        sample_quote(),
    )

    assert validate_history_record(record) is True


def test_invalid_history_record_is_rejected():
    assert validate_history_record({}) is False


def test_append_history_record_persists_records(tmp_path):
    path = tmp_path / "history.json"

    record = build_live_history_record(
        sample_data(),
        sample_quote(),
    )

    history = append_history_record(
        record,
        path=str(path),
        limit=10,
    )

    assert len(history) == 1
    assert path.exists()

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        saved = json.load(handle)

    assert len(saved) == 1
    assert saved[0]["symbol"] == "XAUUSD"


def test_append_history_record_keeps_latest_limit(tmp_path):
    path = tmp_path / "history.json"

    record1 = build_live_history_record(
        sample_data(),
        sample_quote(),
    )

    record2 = dict(record1)
    record2["timestamp"] = "2026-01-01T01:00:00+00:00"

    record3 = dict(record1)
    record3["timestamp"] = "2026-01-01T02:00:00+00:00"

    append_history_record(
        record1,
        path=str(path),
        limit=2,
    )

    append_history_record(
        record2,
        path=str(path),
        limit=2,
    )

    history = append_history_record(
        record3,
        path=str(path),
        limit=2,
    )

    assert len(history) == 2
    assert history[0]["timestamp"] == record2["timestamp"]
    assert history[1]["timestamp"] == record3["timestamp"]


def test_load_history_returns_empty_for_missing_file(
    tmp_path,
):
    path = tmp_path / "missing.json"

    assert load_history(str(path)) == []


def test_build_history_chart_contains_signal_trace():
    history = [
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "signal": 0,
        },
        {
            "timestamp": "2026-01-01T00:05:00+00:00",
            "signal": 1,
        },
    ]

    figure = build_history_chart(history)

    assert len(figure.data) == 1
    assert figure.data[0].type == "scatter"


def test_build_history_chart_empty_history():
    figure = build_history_chart([])

    assert len(figure.data) == 0


def test_render_history_table_contains_expected_columns():
    history = [
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "signal_label": "NO TRADE",
            "trend": "UP",
            "entry_price": 2500.0,
            "stop_loss": None,
            "take_profit": None,
            "risk_reward_ratio": None,
            "mid": 2500.0,
            "market_state": "OPEN",
            "quote_stale": False,
        }
    ]

    frame = render_history_table(history)

    assert "timestamp" in frame.columns
    assert "signal_label" in frame.columns
    assert "trend" in frame.columns
    assert "entry_price" in frame.columns
    assert len(frame) == 1


def test_build_live_history_record_rejects_empty_data():
    with pytest.raises(ValueError):
        build_live_history_record(
            pd.DataFrame(),
            sample_quote(),
        )
