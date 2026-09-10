import pandas as pd
import pytest

from src.data.xaut_usdt_adapter import (
    normalize_xaut_usdt_ohlc,
    xaut_usdt_symbol,
)


def test_xaut_usdt_symbol():
    assert xaut_usdt_symbol() == "XAUTUSDT"


def test_normalize_xaut_usdt_ohlc():
    data = pd.DataFrame(
        {
            "timestamp": [1735689600, 1735776000],
            "open": [2650.0, 2660.0],
            "high": [2670.0, 2680.0],
            "low": [2640.0, 2650.0],
            "close": [2660.0, 2670.0],
            "volume": [1000.0, 1200.0],
        }
    )

    result = normalize_xaut_usdt_ohlc(data)

    assert list(result.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    ]
    assert len(result) == 2
    assert str(result["timestamp"].dt.tz) == "UTC"


def test_normalize_sorts_timestamps():
    data = pd.DataFrame(
        {
            "timestamp": [1735776000, 1735689600],
            "open": [2660.0, 2650.0],
            "high": [2680.0, 2670.0],
            "low": [2650.0, 2640.0],
            "close": [2670.0, 2660.0],
        }
    )

    result = normalize_xaut_usdt_ohlc(data)

    assert result["timestamp"].is_monotonic_increasing


def test_missing_required_column_fails():
    data = pd.DataFrame(
        {
            "timestamp": [1735689600],
            "open": [2650.0],
            "high": [2670.0],
            "low": [2640.0],
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        normalize_xaut_usdt_ohlc(data)


def test_invalid_ohlc_fails():
    data = pd.DataFrame(
        {
            "timestamp": [1735689600],
            "open": [2650.0],
            "high": [2640.0],
            "low": [2630.0],
            "close": [2645.0],
        }
    )

    with pytest.raises(ValueError, match="invalid OHLC"):
        normalize_xaut_usdt_ohlc(data)


def test_duplicate_timestamps_fail():
    data = pd.DataFrame(
        {
            "timestamp": [1735689600, 1735689600],
            "open": [2650.0, 2660.0],
            "high": [2670.0, 2680.0],
            "low": [2640.0, 2650.0],
            "close": [2660.0, 2670.0],
        }
    )

    with pytest.raises(ValueError, match="timestamps must be unique"):
        normalize_xaut_usdt_ohlc(data)
