from pathlib import Path

import pandas as pd
import streamlit as st

from src.evaluation.production_backtest import (
    run_production_backtest,
)
from src.evaluation.production_report import (
    build_production_report,
)


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "xauusd_daily_2025.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "walk_forward"
)


st.set_page_config(
    page_title="AI Trading Lab - Production Report",
    page_icon="📊",
    layout="wide",
)

st.title("AI Trading Lab")
st.subheader("Production Performance Report")


if not DATA_PATH.exists():
    st.error(
        f"Market data not found: {DATA_PATH}"
    )
    st.stop()


if not RESULTS_DIR.exists():
    st.warning(
        "No stored walk-forward experiments "
        "are available."
    )
    st.stop()


run = st.button(
    "Run Production Report",
    type="primary",
    use_container_width=True,
)


if run:
    with st.spinner(
        "Running production strategy..."
    ):
        try:
            result = run_production_backtest(
                data_path=DATA_PATH,
                results_dir=RESULTS_DIR,
            )
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
            st.stop()

    backtest = result["backtest"]

    report = build_production_report(
        backtest
    )

    strategy = result["strategy"]
    stability_score = result[
        "stability_score"
    ]

    st.success(
        "Production report completed."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Strategy",
        strategy,
    )

    col2.metric(
        "Total Return",
        f"{report['total_return']:.4f}",
    )

    col3.metric(
        "Max Drawdown",
        f"{report['max_drawdown']:.4f}",
    )

    col4.metric(
        "Sharpe Ratio",
        f"{report['sharpe_ratio']:.4f}",
    )

    st.metric(
        "Stability Score",
        f"{stability_score:.4f}",
    )

    st.subheader("Performance Summary")

    summary = pd.DataFrame(
        [
            {
                "strategy": strategy,
                "observations": report[
                    "observations"
                ],
                "total_return": report[
                    "total_return"
                ],
                "max_drawdown": report[
                    "max_drawdown"
                ],
                "sharpe_ratio": report[
                    "sharpe_ratio"
                ],
                "stability_score": (
                    stability_score
                ),
            }
        ]
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

    if "strategy_return" in backtest.columns:
        returns = (
            backtest["strategy_return"]
            .fillna(0.0)
        )

        equity = (
            1.0 + returns
        ).cumprod()

        st.subheader("Equity Curve")

        if "timestamp" in backtest.columns:
            chart_data = pd.DataFrame(
                {
                    "Equity": equity.values,
                },
                index=pd.to_datetime(
                    backtest["timestamp"]
                ),
            )
        else:
            chart_data = pd.DataFrame(
                {
                    "Equity": equity.values,
                }
            )

        st.line_chart(chart_data)

    st.subheader("Backtest Data")

    st.dataframe(
        backtest,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Stability Ranking")

    st.dataframe(
        result["stability_report"],
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info(
        "Click the button to generate the "
        "production performance report."
    )
