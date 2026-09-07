import json

import pandas as pd

from src.evaluation.stable_selection import (
    build_stability_report,
    select_stable_strategy,
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

    pd.DataFrame(
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
    ).to_csv(
        run_dir / "final_report.csv",
        index=False,
    )

    pd.DataFrame(
        [{"strategy": strategy}]
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


def test_build_stability_report(tmp_path):
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
        0.08,
        -0.07,
        1.10,
        0.70,
        1,
    )

    create_experiment(
        tmp_path,
        "20260907_000003",
        "moving_average",
        0.03,
        -0.15,
        0.50,
        0.55,
        2,
    )

    result = build_stability_report(
        tmp_path
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert not result.empty

    assert result.iloc[0]["strategy"] == (
        "momentum"
    )

    assert "stability_score" in result.columns


def test_select_stable_strategy(tmp_path):
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
        0.08,
        -0.07,
        1.10,
        0.70,
        1,
    )

    create_experiment(
        tmp_path,
        "20260907_000003",
        "moving_average",
        0.03,
        -0.15,
        0.50,
        0.55,
        2,
    )

    assert (
        select_stable_strategy(tmp_path)
        == "momentum"
    )


def test_empty_results(tmp_path):
    assert (
        select_stable_strategy(tmp_path)
        is None
    )

    assert build_stability_report(
        tmp_path
    ).empty
