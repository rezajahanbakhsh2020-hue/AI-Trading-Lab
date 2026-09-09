import pandas as pd

from app_live_proof_final_view import (
    build_latest_candle_table,
    build_system_status,
)


def sample_data() -> pd.DataFrame:
    index = pd.date_range(
        "2026-09-09",
        periods=2,
        freq="5min",
    )

    return pd.DataFrame(
        {
            "open": [4400.0, 4402.0],
            "high": [4405.0, 4407.0],
            "low": [4398.0, 4400.0],
            "close": [4402.0, 4406.0],
        },
        index=index,
    )


def sample_snapshot() -> dict:
    return {
        "signal": 1,
        "signal_label": "BUY",
        "trend": "UP",
        "market_state": "OPEN",
        "quote_stale": False,
        "candle_count": 200,
        "bid": 4405.8,
        "ask": 4406.2,
        "mid": 4406.0,
        "timestamp": "2026-09-09T15:30:00+00:00",
    }


def test_latest_candle_table_uses_last_candle():
    table = build_latest_candle_table(
        sample_data()
    )

    assert table.iloc[0]["Open"] == 4402.0
    assert table.iloc[0]["High"] == 4407.0
    assert table.iloc[0]["Low"] == 4400.0
    assert table.iloc[0]["Close"] == 4406.0


def test_latest_candle_table_has_expected_columns():
    table = build_latest_candle_table(
        sample_data()
    )

    assert list(table.columns) == [
        "Open",
        "High",
        "Low",
        "Close",
    ]


def test_latest_candle_table_rejects_empty_data():
    try:
        build_latest_candle_table(
            pd.DataFrame()
        )
        assert False
    except ValueError:
        assert True


def test_latest_candle_table_rejects_missing_columns():
    data = pd.DataFrame(
        {
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
        }
    )

    try:
        build_latest_candle_table(data)
        assert False
    except ValueError:
        assert True


def test_system_status_contains_all_components():
    table = build_system_status(
        sample_snapshot()
    )

    assert list(table["Component"]) == [
        "Signal Engine",
        "Trend Engine",
        "Market",
        "Quote",
        "Candles",
    ]


def test_system_status_shows_buy():
    table = build_system_status(
        sample_snapshot()
    )

    row = table[
        table["Component"] == "Signal Engine"
    ].iloc[0]

    assert row["Status"] == "BUY"


def test_system_status_shows_live_quote():
    table = build_system_status(
        sample_snapshot()
    )

    row = table[
        table["Component"] == "Quote"
    ].iloc[0]

    assert row["Status"] == "LIVE"


def test_system_status_shows_candle_count():
    table = build_system_status(
        sample_snapshot()
    )

    row = table[
        table["Component"] == "Candles"
    ].iloc[0]

    assert row["Status"] == "200"
