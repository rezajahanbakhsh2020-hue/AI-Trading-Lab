"""Tests for Headless Live Execution Runtime."""
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.evaluation.live_execution_runtime import LiveExecutionRuntime, load_live_market_data


def make_dummy_df() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01 10:00", periods=100, freq="5min")
    df = pd.DataFrame({
        "openTime": timestamps.view("int64") // 10**6,
        "open": [2000.0 + i for i in range(100)],
        "high": [2005.0 + i for i in range(100)],
        "low": [1995.0 + i for i in range(100)],
        "close": [2002.0 + i for i in range(100)],
    })
    return df


@patch("src.evaluation.live_execution_runtime.fetch_xauusd_ohlc")
def test_load_live_market_data(mock_fetch) -> None:
    mock_fetch.return_value = make_dummy_df()
    df = load_live_market_data("XAUUSD", "5m", 100)
    assert not df.empty
    assert "timestamp" in df.columns
    assert "close" in df.columns


@patch("src.evaluation.live_execution_runtime.load_production_selection")
@patch("src.evaluation.live_execution_runtime.load_live_market_data")
def test_live_execution_runtime_run_once(mock_load_data, mock_load_selection) -> None:
    mock_load_data.return_value = make_dummy_df()
    mock_load_selection.return_value = {
        "stable_strategy": "momentum",
        "stability_score": 0.85,
    }

    mock_publisher = MagicMock()
    mock_publisher.publish.return_value = {
        "status": "PUBLISHED",
        "published": True,
    }

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=mock_publisher,
    )

    result = runtime.run_once(publish=True)

    assert result["symbol"] == "XAUUSD"
    assert result["interval"] == "5m"
    assert result["strategy"] == "momentum"
    assert result["stability_score"] == 0.85
    assert "contract_payload" in result
    assert result["contract_payload"]["contract_version"] == "1.0"
    assert mock_publisher.publish.called
