import pandas as pd

from src.visualization.live_proof_chart import (
    build_live_proof_chart,
)


def sample_data() -> pd.DataFrame:
    index = pd.date_range(
        "2026-09-09",
        periods=60,
        freq="5min",
    )

    return pd.DataFrame(
        {
            "open": range(4400, 4460),
            "high": range(4402, 4462),
            "low": range(4398, 4458),
            "close": range(4401, 4461),
        },
        index=index,
    )


def sample_snapshot() -> dict:
    return {
        "signal": 1,
        "signal_label": "BUY",
        "trend": "UP",
        "fast_window": 20,
        "slow_window": 50,
        "entry_price": 4460.0,
        "stop_loss": 4415.4,
        "take_profit": 4549.2,
    }


def test_chart_contains_candles_and_moving_averages():
    figure = build_live_proof_chart(
        sample_data(),
        sample_snapshot(),
    )

    names = [
        trace.name
        for trace in figure.data
    ]

    assert "XAU/USD" in names
    assert "Fast MA (20)" in names
    assert "Slow MA (50)" in names


def test_chart_contains_buy_signal():
    figure = build_live_proof_chart(
        sample_data(),
        sample_snapshot(),
    )

    names = [
        trace.name
        for trace in figure.data
    ]

    assert "BUY Signal" in names


def test_chart_contains_risk_lines():
    figure = build_live_proof_chart(
        sample_data(),
        sample_snapshot(),
    )

    assert len(figure.layout.shapes) == 3


def test_no_trade_has_no_buy_marker():
    snapshot = sample_snapshot()
    snapshot["signal"] = 0

    figure = build_live_proof_chart(
        sample_data(),
        snapshot,
    )

    names = [
        trace.name
        for trace in figure.data
    ]

    assert "BUY Signal" not in names


def test_chart_rejects_empty_data():
    try:
        build_live_proof_chart(
            pd.DataFrame(),
            sample_snapshot(),
        )
        assert False
    except ValueError:
        assert True


def test_chart_rejects_missing_ohlc():
    data = pd.DataFrame(
        {
            "open": [1.0],
            "high": [2.0],
            "close": [1.5],
        }
    )

    try:
        build_live_proof_chart(
            data,
            sample_snapshot(),
        )
        assert False
    except ValueError:
        assert True
