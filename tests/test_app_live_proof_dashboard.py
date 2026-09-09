import pandas as pd

from app_live_proof_dashboard import (
    build_dashboard_chart,
    build_status_table,
)


def sample_data() -> pd.DataFrame:
    index = pd.date_range(
        "2026-09-09",
        periods=3,
        freq="5min",
    )

    return pd.DataFrame(
        {
            "open": [4400.0, 4402.0, 4401.0],
            "high": [4405.0, 4406.0, 4404.0],
            "low": [4398.0, 4400.0, 4399.0],
            "close": [4402.0, 4401.0, 4403.0],
        },
        index=index,
    )


def sample_snapshot() -> dict:
    return {
        "signal": 1,
        "signal_label": "BUY",
        "trend": "UP",
        "momentum": 0.01,
        "fast_ma": 4402.0,
        "slow_ma": 4395.0,
        "entry_price": 4403.0,
        "stop_loss": 4358.97,
        "take_profit": 4491.06,
        "risk_reward_ratio": 2.0,
        "timestamp": "2026-09-09T15:30:00+00:00",
        "market_state": "OPEN",
        "quote_stale": False,
        "quote_age_seconds": 0,
    }


def test_build_dashboard_chart_contains_candles():
    figure = build_dashboard_chart(
        sample_data(),
        sample_snapshot(),
    )

    assert len(figure.data) == 1
    assert figure.data[0].type == "candlestick"


def test_dashboard_chart_contains_entry_line():
    figure = build_dashboard_chart(
        sample_data(),
        sample_snapshot(),
    )

    assert len(figure.layout.shapes) == 3


def test_dashboard_chart_rejects_empty_data():
    try:
        build_dashboard_chart(
            pd.DataFrame(),
            sample_snapshot(),
        )
        assert False
    except ValueError:
        assert True


def test_dashboard_chart_rejects_missing_ohlc():
    data = pd.DataFrame(
        {
            "open": [1.0],
            "high": [2.0],
            "close": [1.5],
        }
    )

    try:
        build_dashboard_chart(
            data,
            sample_snapshot(),
        )
        assert False
    except ValueError:
        assert True


def test_build_status_table_contains_main_fields():
    table = build_status_table(
        sample_snapshot()
    )

    assert list(table["Field"]) == [
        "Signal",
        "Trend",
        "Market",
        "Quote Stale",
        "Quote Age",
        "Timestamp",
    ]


def test_build_status_table_contains_signal():
    table = build_status_table(
        sample_snapshot()
    )

    signal_row = table[
        table["Field"] == "Signal"
    ].iloc[0]

    assert signal_row["Value"] == "BUY"


def test_build_status_table_contains_market_state():
    table = build_status_table(
        sample_snapshot()
    )

    market_row = table[
        table["Field"] == "Market"
    ].iloc[0]

    assert market_row["Value"] == "OPEN"
