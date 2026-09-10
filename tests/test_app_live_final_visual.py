import pandas as pd
import plotly.graph_objects as go

from app_live_final_visual import (
    build_final_visual_figure,
    build_final_visual_snapshot,
    build_production_visual_state,
)


def sample_candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "openTime": pd.date_range(
                "2025-01-01",
                periods=60,
                freq="5min",
            ),
            "open": [
                4390.0 + i
                for i in range(60)
            ],
            "high": [
                4391.0 + i
                for i in range(60)
            ],
            "low": [
                4389.0 + i
                for i in range(60)
            ],
            "close": [
                4390.5 + i
                for i in range(60)
            ],
        }
    )


def sample_quote() -> dict:
    return {
        "symbol": "XAUUSD",
        "mid": 4449.50,
        "marketState": "OPEN",
        "stale": False,
        "quoteAgeSeconds": 2.5,
    }


def sample_production_result() -> dict:
    return {
        "strategy": "momentum",
        "stability_score": 0.517268,
        "report": {
            "observations": 312,
            "total_return": 0.497303,
            "max_drawdown": -0.069050,
            "sharpe_ratio": 0.142056,
        },
    }


def test_build_production_visual_state_exposes_metrics():
    state = build_production_visual_state(
        sample_production_result()
    )

    assert (
        state["stable_strategy"]
        == "momentum"
    )

    assert (
        state["stability_score"]
        == 0.517268
    )

    assert (
        state["production_status"]
        == "SUCCESS"
    )

    assert (
        state["production_observations"]
        == 312
    )

    assert (
        state["production_total_return"]
        == 0.497303
    )

    assert (
        state["production_max_drawdown"]
        == -0.069050
    )

    assert (
        state["production_sharpe_ratio"]
        == 0.142056
    )


def test_final_visual_snapshot_combines_live_and_production():
    candles = sample_candles()

    production = (
        build_production_visual_state(
            sample_production_result()
        )
    )

    snapshot = build_final_visual_snapshot(
        candles,
        sample_quote(),
        production,
        interval="5m",
    )

    assert snapshot["symbol"] == "XAUUSD"
    assert snapshot["interval"] == "5m"
    assert snapshot["market_state"] == "OPEN"

    assert snapshot["quote_stale"] is False
    assert (
        snapshot["quote_age_seconds"]
        == 2.5
    )

    assert snapshot["quote_mid"] == 4449.50
    assert snapshot["candle_count"] == 60

    assert (
        snapshot["stable_strategy"]
        == "momentum"
    )

    assert (
        snapshot["stability_score"]
        == 0.517268
    )

    assert (
        snapshot["production_status"]
        == "SUCCESS"
    )

    assert "signal_label" in snapshot
    assert "trend" in snapshot
    assert "entry_price" in snapshot
    assert "stop_loss" in snapshot
    assert "take_profit" in snapshot


def test_final_visual_figure_is_plotly_figure():
    candles = sample_candles()

    production = (
        build_production_visual_state(
            sample_production_result()
        )
    )

    snapshot = build_final_visual_snapshot(
        candles,
        sample_quote(),
        production,
        interval="5m",
    )

    figure = build_final_visual_figure(
        candles,
        snapshot,
    )

    assert isinstance(
        figure,
        go.Figure,
    )

    assert len(figure.data) > 0

    html = figure.to_html(
        include_plotlyjs=False
    )

    assert (
        "AI-Trading-Lab"
        in html
    )

    assert "momentum" in html
    assert "0.517268" in html
    assert "SUCCESS" in html
    assert "LIVE" in html
    assert "FRESH" in html


def test_final_visual_marks_stale_quote():
    candles = sample_candles()

    quote = sample_quote()
    quote["stale"] = True

    production = (
        build_production_visual_state(
            sample_production_result()
        )
    )

    snapshot = build_final_visual_snapshot(
        candles,
        quote,
        production,
        interval="5m",
    )

    assert (
        snapshot["quote_stale"]
        is True
    )

    figure = build_final_visual_figure(
        candles,
        snapshot,
    )

    html = figure.to_html(
        include_plotlyjs=False
    )

    assert "STALE QUOTE" in html


def test_final_visual_preserves_existing_risk_levels():
    candles = sample_candles()

    production = (
        build_production_visual_state(
            sample_production_result()
        )
    )

    snapshot = build_final_visual_snapshot(
        candles,
        sample_quote(),
        production,
        interval="5m",
    )

    assert "entry_price" in snapshot
    assert "stop_loss" in snapshot
    assert "take_profit" in snapshot

    figure = build_final_visual_figure(
        candles,
        snapshot,
    )

    assert isinstance(
        figure,
        go.Figure,
    )
