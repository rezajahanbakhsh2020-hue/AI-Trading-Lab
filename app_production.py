from pathlib import Path

import streamlit as st

from src.evaluation.production_backtest import (
    run_production_backtest,
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
    page_title="AI Trading Lab - Production",
    page_icon="📈",
    layout="wide",
)

st.title("AI Trading Lab")
st.subheader("Production Strategy Backtest")

if not DATA_PATH.exists():
    st.error(
        f"Market data not found: {DATA_PATH}"
    )
    st.stop()


st.write(
    "The production strategy is selected "
    "from cross-experiment stability."
)

run = st.button(
    "Run Production Backtest",
    type="primary",
    use_container_width=True,
)


if run:
    if not RESULTS_DIR.exists():
        st.warning(
            "No stored walk-forward experiments "
            "are available."
        )
        st.stop()

    with st.spinner(
        "Running selected production strategy..."
    ):
        try:
            result = run_production_backtest(
                data_path=DATA_PATH,
                results_dir=RESULTS_DIR,
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    strategy = result["strategy"]
    stability_score = result[
        "stability_score"
    ]
    backtest = result["backtest"]

    st.success(
        "Production backtest completed."
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Selected Strategy",
        strategy,
    )

    col2.metric(
        "Stability Score",
        f"{stability_score:.4f}",
    )

    st.subheader("Backtest Result")

    st.dataframe(
        backtest,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Stability Ranking"
    )

    st.dataframe(
        result["stability_report"],
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info(
        "Click the button to run the selected "
        "production strategy."
    )
