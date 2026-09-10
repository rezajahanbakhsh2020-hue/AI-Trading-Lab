import pandas as pd
import plotly.graph_objects as go
import pytest

from src.visualization.live_history_chart import (
    build_live_history_chart,
    build_live_history_frame,
)


def sample_snapshots() -> list[dict]:
    return [
        {
            "timestamp": "2026-09-10T08:00:00",
            "signal_label": "NO TRADE",
            "trend": "UP",
            "strategy": "momentum",
            "entry_price": 4420.0,
            "stop_loss": 4375.0,
            "tp1": 4510.0,
            "tp2": 4550.0,
            "tp3": 4600.0,
            "stability_score": 0.51,
            "quote_stale": False,
        },
        {
            "timestamp": "2026-09-10T08:05:00",
            "signal_label": "BUY",
            "trend": "UP",
            "strategy": "momentum",
            "entry_price": 4430.0,
            "stop_loss": 4385.0,
            "tp1": 4520.0,
            "tp2": 4560.0,
            "tp3": 4610.0,
            "stability_score": 0.517268,
            "quote_stale": False,
        },
        {
            "timestamp": "2026-09-10T08:10:00",
            "signal_label": "BUY",
            "trend": "UP",
            "strategy": "momentum",
            "entry_price": 4440.0,
            "stop_loss": 4395.0,
            "tp1": 4530.0,
            "tp2": 4570.0,
            "tp3": 4620.0,
            "stability_score": 0.52,
            "quote_stale": True,
        },
    ]


def test_build_live_history_frame():
    frame = build_live_history_frame(
        sample_snapshots()
    )

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 3

    assert list(frame["signal"]) == [
        "NO TRADE",
        "BUY",
        "BUY",
    ]

    assert list(frame["trend"]) == [
        "UP",
        "UP",
        "UP",
    ]

    assert frame.iloc[1]["entry"] == 4430.0
    assert frame.iloc[1]["stop_loss"] == 4385.0
    assert frame.iloc[1]["tp1"] == 4520.0
    assert frame.iloc[1]["tp2"] == 4560.0
    assert frame.iloc[1]["tp3"] == 4610.0

    assert (
        frame.iloc[1]["stability_score"]
        == 0.517268
    )

    assert (
        frame["quote_stale"].tolist()
        == [False, False, True]
    )


def test_history_frame_supports_take_profit_fallback():
    snapshots = [
        {
            "signal_label": "BUY",
            "entry_price": 4400.0,
            "stop_loss": 4350.0,
            "take_profit": 4500.0,
        }
    ]

    frame = build_live_history_frame(
        snapshots
    )

    assert frame.iloc[0]["tp1"] == 4500.0


def test_build_live_history_chart():
    figure = build_live_history_chart(
        sample_snapshots()
    )

    assert isinstance(
        figure,
        go.Figure,
    )

    assert len(figure.data) > 0

    names = {
        trace.name
        for trace in figure.data
    }

    assert "Entry" in names
    assert "SL" in names
    assert "TP1" in names
    assert "TP2" in names
    assert "TP3" in names
    assert "Stability Score" in names
    assert "Signal History" in names

    html = figure.to_html(
        include_plotlyjs=False
    )

    assert (
        "Live Decision History"
        in html
    )

    assert "Stale Quotes" in html


def test_empty_history_returns_empty_figure():
    figure = build_live_history_chart([])

    assert isinstance(
        figure,
        go.Figure,
    )

    assert len(figure.data) == 0


def test_invalid_snapshot_sequence_item():
    with pytest.raises(TypeError):
        build_live_history_frame(
            [
                {
                    "signal_label": "BUY",
                },
                "invalid",
            ]
        )


def test_invalid_snapshots_container():
    with pytest.raises(TypeError):
        build_live_history_frame(
            {"signal_label": "BUY"}
        )
