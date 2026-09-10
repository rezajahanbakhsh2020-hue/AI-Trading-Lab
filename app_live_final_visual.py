from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_live import (
    DEFAULT_INTERVAL,
    DEFAULT_LIMIT,
    MAX_LIMIT,
    SUPPORTED_INTERVALS,
    fetch_xauusd_ohlc,
    fetch_xauusd_quote,
)
from app_live_signal_trend_risk import (
    build_signal_trend_risk_snapshot,
)
from src.evaluation.production_pipeline import (
    run_production_pipeline,
)
from src.visualization.live_visual_suite import (
    build_live_visual_suite,
)


ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    ROOT
    / "data"
    / "raw"
    / "xauusd_daily_2025.csv"
)

WALK_FORWARD_DIR = (
    ROOT
    / "results"
    / "walk_forward"
)

PRODUCTION_DIR = (
    ROOT
    / "results"
    / "production"
)

DEFAULT_REFRESH_SECONDS = 15


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_production_visual_state(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")

    report = result.get("report")

    if not isinstance(report, Mapping):
        raise ValueError(
            "production result must contain a report mapping"
        )

    return {
        "stable_strategy": result.get("strategy"),
        "stability_score": _float_or_none(
            result.get("stability_score")
        ),
        "production_status": "SUCCESS",
        "production_observations": report.get(
            "observations"
        ),
        "production_total_return": _float_or_none(
            report.get("total_return")
        ),
        "production_max_drawdown": _float_or_none(
            report.get("max_drawdown")
        ),
        "production_sharpe_ratio": _float_or_none(
            report.get("sharpe_ratio")
        ),
    }


def build_final_visual_snapshot(
    candles: pd.DataFrame,
    quote: Mapping[str, Any],
    production_state: Mapping[str, Any],
    *,
    interval: str,
) -> dict[str, Any]:
    if not isinstance(candles, pd.DataFrame):
        raise TypeError(
            "candles must be a pandas DataFrame"
        )

    if candles.empty:
        raise ValueError(
            "candles must not be empty"
        )

    if not isinstance(quote, Mapping):
        raise TypeError(
            "quote must be a mapping"
        )

    if not isinstance(production_state, Mapping):
        raise TypeError(
            "production_state must be a mapping"
        )

    signal_state = (
        build_signal_trend_risk_snapshot(candles)
    )

    snapshot = dict(signal_state)

    snapshot.update(
        {
            "symbol": "XAUUSD",
            "interval": interval,
            "market_state": str(
                quote.get(
                    "marketState",
                    "UNKNOWN",
                )
            ).upper(),
            "quote_stale": bool(
                quote.get("stale", False)
            ),
            "quote_age_seconds": quote.get(
                "quoteAgeSeconds"
            ),
            "quote_mid": _float_or_none(
                quote.get("mid")
            ),
            "candle_count": int(
                len(candles)
            ),
            "stable_strategy": (
                production_state.get(
                    "stable_strategy"
                )
            ),
            "stability_score": (
                production_state.get(
                    "stability_score"
                )
            ),
            "production_status": (
                production_state.get(
                    "production_status",
                    "UNKNOWN",
                )
            ),
            "production_observations": (
                production_state.get(
                    "production_observations"
                )
            ),
            "production_total_return": (
                production_state.get(
                    "production_total_return"
                )
            ),
            "production_max_drawdown": (
                production_state.get(
                    "production_max_drawdown"
                )
            ),
            "production_sharpe_ratio": (
                production_state.get(
                    "production_sharpe_ratio"
                )
            ),
        }
    )

    return snapshot


def build_final_visual_figure(
    candles: pd.DataFrame,
    snapshot: Mapping[str, Any],
) -> go.Figure:
    if not isinstance(candles, pd.DataFrame):
        raise TypeError(
            "candles must be a pandas DataFrame"
        )

    if candles.empty:
        raise ValueError(
            "candles must not be empty"
        )

    if not isinstance(snapshot, Mapping):
        raise TypeError(
            "snapshot must be a mapping"
        )

    figure = build_live_visual_suite(
        candles,
        snapshot,
    )

    quote_mid = _float_or_none(
        snapshot.get("quote_mid")
    )

    if quote_mid is not None:
        figure.add_hline(
            y=quote_mid,
            line_dash="dot",
            annotation_text="LIVE",
        )

    stable_strategy = snapshot.get(
        "stable_strategy",
        "N/A",
    )

    stability_score = _float_or_none(
        snapshot.get("stability_score")
    )

    if stability_score is None:
        stability_text = "N/A"
    else:
        stability_text = (
            f"{stability_score:.6f}"
        )

    production_status = snapshot.get(
        "production_status",
        "UNKNOWN",
    )

    figure.add_annotation(
        x=0.5,
        y=-0.075,
        xref="paper",
        yref="paper",
        text=(
            f"<b>Stable Strategy:</b> "
            f"{stable_strategy} | "
            f"<b>Stability:</b> "
            f"{stability_text} | "
            f"<b>Production:</b> "
            f"{production_status}"
        ),
        showarrow=False,
        xanchor="center",
        yanchor="top",
    )

    quote_status = (
        "STALE"
        if bool(
            snapshot.get(
                "quote_stale",
                False,
            )
        )
        else "FRESH"
    )

    quote_age = snapshot.get(
        "quote_age_seconds",
        "N/A",
    )

    figure.add_annotation(
        x=0.5,
        y=-0.12,
        xref="paper",
        yref="paper",
        text=(
            f"<b>Quote:</b> "
            f"{quote_status} | "
            f"<b>Age:</b> "
            f"{quote_age}s | "
            f"<b>Candles:</b> "
            f"{snapshot.get('candle_count', 'N/A')}"
        ),
        showarrow=False,
        xanchor="center",
        yanchor="top",
    )

    figure.update_layout(
        height=980,
        margin=dict(
            l=40,
            r=40,
            t=190,
            b=125,
        ),
    )

    return figure


@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def load_production_visual_state() -> dict[str, Any]:
    result = run_production_pipeline(
        data_path=DATA_PATH,
        results_dir=WALK_FORWARD_DIR,
        production_dir=PRODUCTION_DIR,
        save_result=True,
    )

    return build_production_visual_state(
        result
    )


def _render_production_metrics(
    snapshot: Mapping[str, Any],
) -> None:
    st.subheader(
        "Stable Strategy & Production"
    )

    strategy = snapshot.get(
        "stable_strategy",
        "N/A",
    )

    stability = _float_or_none(
        snapshot.get("stability_score")
    )

    production = snapshot.get(
        "production_status",
        "UNKNOWN",
    )

    stability_text = (
        "N/A"
        if stability is None
        else f"{stability:.6f}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Stable Strategy",
        str(strategy),
    )

    col2.metric(
        "Stability Score",
        stability_text,
    )

    col3.metric(
        "Production",
        str(production),
    )

    col4, col5, col6 = st.columns(3)

    total_return = _float_or_none(
        snapshot.get(
            "production_total_return"
        )
    )

    max_drawdown = _float_or_none(
        snapshot.get(
            "production_max_drawdown"
        )
    )

    sharpe = _float_or_none(
        snapshot.get(
            "production_sharpe_ratio"
        )
    )

    col4.metric(
        "Production Return",
        (
            "N/A"
            if total_return is None
            else f"{total_return:.6f}"
        ),
    )

    col5.metric(
        "Max Drawdown",
        (
            "N/A"
            if max_drawdown is None
            else f"{max_drawdown:.6f}"
        ),
    )

    col6.metric(
        "Sharpe Ratio",
        (
            "N/A"
            if sharpe is None
            else f"{sharpe:.6f}"
        ),
    )


def _render_dashboard() -> None:
    st.set_page_config(
        page_title=(
            "AI-Trading-Lab | Final Visual"
        ),
        page_icon="📈",
        layout="wide",
    )

    st.title(
        "AI-Trading-Lab — Final Visual Dashboard"
    )

    st.caption(
        "Real XAU/USD → Signal → Trend → "
        "Risk → Decision → Stability → Production"
    )

    with st.sidebar:
        st.header("Live Visual")

        interval = st.selectbox(
            "Timeframe",
            SUPPORTED_INTERVALS,
            index=SUPPORTED_INTERVALS.index(
                DEFAULT_INTERVAL
            ),
        )

        limit = st.number_input(
            "Candles",
            min_value=1,
            max_value=MAX_LIMIT,
            value=DEFAULT_LIMIT,
            step=1,
        )

        st.caption(
            f"Automatic refresh: "
            f"{DEFAULT_REFRESH_SECONDS}s"
        )

    try:
        candles = fetch_xauusd_ohlc(
            interval=interval,
            limit=int(limit),
        )

        quote = fetch_xauusd_quote()

        production_state = (
            load_production_visual_state()
        )

        snapshot = build_final_visual_snapshot(
            candles,
            quote,
            production_state,
            interval=interval,
        )

    except Exception as exc:
        st.error(
            f"Unable to build final visual: {exc}"
        )
        return

    if bool(
        snapshot.get(
            "quote_stale",
            False,
        )
    ):
        st.warning(
            "The latest XAU/USD quote is "
            "marked as stale by the data provider."
        )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Signal",
        str(
            snapshot.get(
                "signal_label",
                "NO TRADE",
            )
        ),
    )

    col2.metric(
        "Trend",
        str(
            snapshot.get(
                "trend",
                "UNKNOWN",
            )
        ),
    )

    entry = _float_or_none(
        snapshot.get("entry_price")
    )

    col3.metric(
        "Entry",
        (
            "N/A"
            if entry is None
            else f"{entry:.2f}"
        ),
    )

    col4.metric(
        "Market",
        str(
            snapshot.get(
                "market_state",
                "UNKNOWN",
            )
        ),
    )

    st.plotly_chart(
        build_final_visual_figure(
            candles,
            snapshot,
        ),
        use_container_width=True,
    )

    _render_production_metrics(
        snapshot
    )

    st.subheader("Risk Levels")

    risk1, risk2, risk3 = st.columns(3)

    stop_loss = _float_or_none(
        snapshot.get("stop_loss")
    )

    take_profit = _float_or_none(
        snapshot.get("take_profit")
    )

    risk1.metric(
        "Stop Loss",
        (
            "N/A"
            if stop_loss is None
            else f"{stop_loss:.2f}"
        ),
    )

    risk2.metric(
        "Take Profit",
        (
            "N/A"
            if take_profit is None
            else f"{take_profit:.2f}"
        ),
    )

    risk3.metric(
        "Risk / Reward",
        str(
            snapshot.get(
                "risk_reward_ratio",
                "N/A",
            )
        ),
    )

    signal = snapshot.get(
        "signal_label",
        "NO TRADE",
    )

    if signal == "BUY":
        st.success(
            "BUY — active trade setup is visible."
        )
    else:
        st.info(
            "NO TRADE — waiting for an active setup."
        )

    st.caption(
        f"Strategy: "
        f"{snapshot.get('strategy', 'N/A')} | "
        f"Quote age: "
        f"{snapshot.get('quote_age_seconds', 'N/A')}s | "
        f"Candles: "
        f"{snapshot.get('candle_count', 'N/A')}"
    )


def main() -> None:
    if hasattr(st, "fragment"):

        @st.fragment(
            run_every=(
                f"{DEFAULT_REFRESH_SECONDS}s"
            )
        )
        def live_fragment() -> None:
            _render_dashboard()

        live_fragment()

    else:
        _render_dashboard()


if __name__ == "__main__":
    main()
