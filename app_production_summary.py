from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.evaluation.production_summary import (
    build_production_summary,
    get_latest_production_result,
)
from src.evaluation.production_report import (
    build_production_report,
)


DATA_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = DATA_ROOT / "results" / "production"


st.set_page_config(
    page_title="AI-Trading-Lab Production Summary",
    layout="wide",
)

st.title("AI-Trading-Lab")
st.subheader("Production Summary")

summary = build_production_summary(RESULTS_DIR)

if summary.empty:
    st.warning("No production results available.")
    st.stop()

latest = get_latest_production_result(RESULTS_DIR)

if latest is not None:
    report = build_production_report(
        latest["backtest"]
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Strategy",
        latest["metadata"].get(
            "strategy",
            "N/A",
        ),
    )

    col2.metric(
        "Stability Score",
        f"{float(latest['metadata'].get('stability_score', 0.0)):.4f}",
    )

    col3.metric(
        "Total Return",
        f"{report['total_return']:.2%}",
    )

    col4.metric(
        "Sharpe Ratio",
        f"{report['sharpe_ratio']:.4f}",
    )

    st.subheader("Latest Production Equity")

    backtest = latest["backtest"].copy()

    if "strategy_return" in backtest.columns:
        equity = (
            1.0
            + backtest["strategy_return"].fillna(0.0)
        ).cumprod()

        st.line_chart(equity)

st.subheader("Production Runs")

st.dataframe(
    summary,
    use_container_width=True,
)
