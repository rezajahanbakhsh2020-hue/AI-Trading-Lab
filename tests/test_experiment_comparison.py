import json

import pandas as pd

from src.evaluation.experiment_comparison import (
    compare_experiments,
    find_experiments,
    load_experiment,
    summarize_experiments,
)


def create_experiment(
    root,
    name,
    strategy,
    total_return,
    drawdown,
    sharpe,
    positive_rate,
    rank,
):
    run_dir = root / name
    run_dir.mkdir()

    report = pd.DataFrame(
        [
            {
                "strategy": strategy,
                "rank": rank,
                "total_return": total_return,
                "max_drawdown": drawdown,
                "sharpe_ratio": sharpe,
                "sortino_ratio": sharpe,
                "calmar_ratio": sharpe,
                "positive_window_rate": positive_rate,
            }
        ]
    )

    report.to_csv(
        run_dir / "final_report.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "strategy": strategy,
            }
        ]
    ).to_csv(
        run_dir / "eligible_strategies.csv",
        index=False,
    )

    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": (
                    f"2026-09-07T00:00:{name[-2:]}Z"
                ),
                "best_strategy": strategy,
            }
        ),
        encoding="utf-8",
    )


def test_find_experiments(tmp_path):
    create_experiment(
        tmp_path,
        "20260907_000001",
        "momentum",
        0.10,
        -0.08,
        1.20,
        0.75,
        1,
    )

    assert len(
        find_experiments(tmp_path)
    ) == 1


def test_load_experiment(tmp_path):
    create_experiment(
        tmp_path,
        "20260907_000001",
        "momentum",
        0.10,
        -0.08,
        1.20,
        0.75,
        1,
    )

    result = load_experiment(
        tmp_path / "20260907_000001"
    )

    assert isinstance(
        result["final_report"],
        pd.DataFrame,
    )

    assert result["metadata"][
        "best_strategy"
    ] == "momentum"


def test_compare_experiments(tmp_path):
    create_experiment(
        tmp_path,
        "20260907_000001",
        "momentum",
        0.10,
        -0.08,
        1.20,
        0.75,
        1,
    )

    create_experiment(
        tmp_path,
        "20260907_000002",
        "moving_average",
        0.06,
        -0.10,
        0.90,
        0.60,
        2,
    )

    result = compare_experiments(
        tmp_path
    )

    assert len(result) == 2

    assert {
        "run_id",
        "strategy",
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
        "positive_window_rate",
    }.issubset(result.columns)


def test_summarize_experiments(tmp_path):
    create_experiment(
        tmp_path,
        "20260907_000001",
        "momentum",
        0.10,
        -0.08,
        1.20,
        0.75,
        1,
    )

    create_experiment(
        tmp_path,
        "20260907_000002",
        "momentum",
        0.06,
        -0.10,
        0.80,
        0.60,
        2,
    )

    summary = summarize_experiments(
        tmp_path
    )

    assert len(summary) == 1
    assert summary.iloc[0]["strategy"] == (
        "momentum"
    )
    assert summary.iloc[0]["experiments"] == 2
    assert summary.iloc[0]["avg_total_return"] == 0.08


def test_empty_experiment_directory(tmp_path):
    result = compare_experiments(
        tmp_path
    )

    summary = summarize_experiments(
        tmp_path
    )

    assert result.empty
    assert summary.empty
