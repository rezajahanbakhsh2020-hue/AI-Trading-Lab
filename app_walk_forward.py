from pathlib import Path

import pandas as pd
import streamlit as st

from src.evaluation.live_workflow import (
    DEFAULT_STEP,
    DEFAULT_TEST_SIZE,
    DEFAULT_TRAIN_SIZE,
    run_xauusd_walk_forward,
)
from src.evaluation.stable_selection import (
    build_stability_report,
    select_stable_strategy,
)


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "xauusd_daily_2025.csv"
)


st.set_page_config(
    page_title="AI Trading Lab - XAU/USD",
    page_icon="📈",
    layout="wide",
)


st.title("AI Trading Lab")
st.subheader("XAU/USD Walk-Forward Research")


if not DATA_PATH.exists():
    st.error(
        f"Market data not found: {DATA_PATH}"
    )
    st.stop()


raw_data = pd.read_csv(DATA_PATH)
data_size = len(raw_data)

default_train = min(
    DEFAULT_TRAIN_SIZE,
    max(20, data_size // 2),
)

default_test = min(
    DEFAULT_TEST_SIZE,
    max(10, default_train // 3),
)

default_step = min(
    DEFAULT_STEP,
    max(5, default_test // 2),
)


with st.sidebar:
    st.header("Walk-Forward Configuration")

    train_size = st.number_input(
        "Train Size",
        min_value=20,
        max_value=data_size,
        value=int(default_train),
        step=10,
    )

    test_size = st.number_input(
        "Test Size",
        min_value=5,
        max_value=data_size,
        value=int(default_test),
        step=5,
    )

    step = st.number_input(
        "Step",
        min_value=1,
        max_value=data_size,
        value=int(default_step),
        step=5,
    )

    st.divider()

    st.header("Eligibility")

    min_total_return = st.number_input(
        "Minimum Total Return",
        min_value=0.0,
        value=0.0,
        step=0.01,
    )

    max_drawdown = st.number_input(
        "Maximum Drawdown",
        min_value=0.0,
        value=0.20,
        step=0.01,
    )

    min_sharpe_ratio = st.number_input(
        "Minimum Sharpe Ratio",
        min_value=0.0,
        value=0.0,
        step=0.10,
    )

    min_positive_window_rate = st.number_input(
        "Minimum Positive Window Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
    )


if train_size + test_size > data_size:
    st.warning(
        "Train Size + Test Size exceeds "
        "available data."
    )
    st.stop()


col1, col2, col3, col4 = st.columns(4)

col1.metric("Data Rows", f"{data_size:,}")
col2.metric("Train", f"{train_size:,}")
col3.metric("Test", f"{test_size:,}")
col4.metric("Step", f"{step:,}")


run = st.button(
    "Run XAU/USD Walk-Forward",
    type="primary",
    use_container_width=True,
)


if run:
    with st.spinner(
        "Running out-of-sample evaluation..."
    ):
        result = run_xauusd_walk_forward(
            path=DATA_PATH,
            train_size=int(train_size),
            test_size=int(test_size),
            step=int(step),
            min_total_return=float(
                min_total_return
            ),
            max_drawdown=float(max_drawdown),
            min_sharpe_ratio=float(
                min_sharpe_ratio
            ),
            min_positive_window_rate=float(
                min_positive_window_rate
            ),
        )

    st.success(
        "Walk-forward evaluation completed."
    )

    final_report = result[
        "final_report"
    ].copy()

    eligible = result[
        "eligible_strategies"
    ].copy()

    best_strategy = result[
        "best_strategy"
    ]

    st.subheader("Current Ranking")

    st.dataframe(
        final_report,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Strategy Selection")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Best Current Strategy",
        str(best_strategy),
    )

    if eligible.empty:
        col2.metric(
            "Eligible Strategies",
            "0",
        )
    else:
        col2.metric(
            "Eligible Strategies",
            str(len(eligible)),
        )

    stable_strategy = select_stable_strategy()

    col3.metric(
        "Stable Strategy",
        stable_strategy
        if stable_strategy
        else "N/A",
    )

    st.subheader("Eligibility")

    if eligible.empty:
        st.warning(
            "No strategy passed the "
            "eligibility criteria."
        )
    else:
        st.success(
            f"{len(eligible)} strategy(s) "
            "passed eligibility."
        )

        st.dataframe(
            eligible,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader(
        "Cross-Experiment Stability"
    )

    stability_report = (
        build_stability_report()
    )

    if stability_report.empty:
        st.info(
            "No stored experiments are "
            "available for stability analysis."
        )
    else:
        st.dataframe(
            stability_report,
            use_container_width=True,
            hide_index=True,
        )

        if stable_strategy:
            stable_row = stability_report[
                stability_report["strategy"]
                == stable_strategy
            ]

            if not stable_row.empty:
                score = float(
                    stable_row.iloc[0][
                        "stability_score"
                    ]
                )

                st.success(
                    f"Stable Strategy: "
                    f"{stable_strategy} | "
                    f"Stability Score: "
                    f"{score:.4f}"
                )

    st.subheader(
        "Performance Comparison"
    )

    performance_columns = [
        column
        for column in [
            "strategy",
            "total_return",
            "max_drawdown",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "positive_window_rate",
            "windows",
            "observations",
        ]
        if column in final_report.columns
    ]

    if performance_columns:
        st.dataframe(
            final_report[
                performance_columns
            ],
            use_container_width=True,
            hide_index=True,
        )

else:
    st.info(
        "Configure the parameters and "
        "run the evaluation."
    )
