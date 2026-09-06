import pandas as pd
import pytest

from src.visualization.chart_data import prepare_chart_data


def valid_data():
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-03",
                "2026-01-01",
                "2026-01-02",
            ],
            "open": [103.0, 101.0, 102.0],
            "high": [105.0, 103.0, 104.0],
            "low": [100.0, 99.0, 101.0],
            "close": [104.0, 102.0, 103.0],
            "volume": [300, 100, 200],
            "signal": [1, 0, -1],
            "volatility_regime": [
                "high_volatility",
                "normal_volatility",
                "low_volatility",
            ],
        }
    )


def test_required_columns():
    result = prepare_chart_data(valid_data())

    assert {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    }.issubset(result.columns)


def test_optional_columns_preserved():
    result = prepare_chart_data(valid_data())

    assert {
        "volume",
        "signal",
        "volatility_regime",
    }.issubset(result.columns)


def test_sorted_by_timestamp():
    result = prepare_chart_data(valid_data())

    assert result["timestamp"].is_monotonic_increasing


def test_no_modify_input():
    data = valid_data()
    original = data.copy(deep=True)

    prepare_chart_data(data)

    pd.testing.assert_frame_equal(data, original)


def test_missing_column():
    data = valid_data().drop(columns=["close"])

    with pytest.raises(ValueError):
        prepare_chart_data(data)


def test_invalid_timestamp():
    data = valid_data()
    data.loc[0, "timestamp"] = "invalid"

    with pytest.raises(ValueError):
        prepare_chart_data(data)


def test_invalid_ohlc():
    data = valid_data()
    data.loc[0, "high"] = 90.0

    with pytest.raises(ValueError):
        prepare_chart_data(data)


def test_non_dataframe():
    with pytest.raises(TypeError):
        prepare_chart_data(None)
