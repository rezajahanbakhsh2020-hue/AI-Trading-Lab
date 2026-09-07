from pathlib import Path

import pandas as pd

from src.evaluation.live_workflow import (
    DEFAULT_STEP,
    DEFAULT_TEST_SIZE,
    DEFAULT_TRAIN_SIZE,
    build_default_strategies,
    prepare_xauusd_data,
    run_xauusd_walk_forward,
)


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "xauusd_daily_2025.csv"
)


def test_default_workflow_configuration() -> None:
    assert DEFAULT_TRAIN_SIZE > 0
    assert DEFAULT_TEST_SIZE > 0
    assert DEFAULT_STEP > 0


def test_build_default_strategies() -> None:
    strategies = build_default_strategies()

    assert set(strategies) == {
        "moving_average",
        "momentum",
    }

    assert all(
        callable(strategy)
        for strategy in strategies.values()
    )


def test_prepare_xauusd_data() -> None:
    df = prepare_xauusd_data(DATA_PATH)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    required_columns = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "return",
    }

    assert required_columns.issubset(
        df.columns
    )

    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].is_unique


def test_run_xauusd_walk_forward() -> None:
    result = run_xauusd_walk_forward(
        path=DATA_PATH,
        train_size=60,
        test_size=20,
        step=20,
    )

    assert isinstance(result, dict)

    assert isinstance(
        result["data"],
        pd.DataFrame,
    )

    assert isinstance(
        result["comparison"],
        dict,
    )

    assert set(result["comparison"]) == {
        "moving_average",
        "momentum",
    }

    assert isinstance(
        result["final_report"],
        pd.DataFrame,
    )

    assert not result["final_report"].empty

    assert {
        "rank",
        "strategy",
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
        "positive_window_rate",
    }.issubset(
        result["final_report"].columns
    )

    assert isinstance(
        result["eligible_strategies"],
        pd.DataFrame,
    )

    assert result["best_strategy"] in {
        "moving_average",
        "momentum",
    }


def test_run_xauusd_walk_forward_can_use_sharpe() -> None:
    result = run_xauusd_walk_forward(
        path=DATA_PATH,
        train_size=60,
        test_size=20,
        step=20,
        metric="sharpe_ratio",
    )

    assert not result["final_report"].empty
    assert result["best_strategy"] in {
        "moving_average",
        "momentum",
    }
