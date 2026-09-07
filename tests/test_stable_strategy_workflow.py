from pathlib import Path

import pandas as pd

from src.evaluation.experiment_comparison import (
    compare_experiments,
)
from src.evaluation.result_store import (
    save_walk_forward_result,
)
from src.evaluation.stable_selection import (
    build_stability_report,
    select_stable_strategy,
)


def create_result(
    strategy: str,
    total_return: float,
    sharpe_ratio: float,
    max_drawdown: float,
    positive_window_rate: float,
) -> dict:
    final_report = pd.DataFrame(
        [
            {
                "rank": 1,
                "strategy": strategy,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sharpe_ratio,
                "calmar_ratio": sharpe_ratio,
                "positive_window_rate": (
                    positive_window_rate
                ),
            }
        ]
    )

    eligible = final_report.copy()

    return {
        "data": pd.DataFrame(
            {"timestamp": [1, 2, 3]}
        ),
        "comparison": {
            strategy: {
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "positive_window_rate": (
                    positive_window_rate
                ),
            }
        },
        "final_report": final_report,
        "eligible_strategies": eligible,
        "best_strategy": strategy,
    }


def test_saved_experiments_are_available_for_stability(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "walk_forward"

    first_result = create_result(
        strategy="moving_average",
        total_return=0.20,
        sharpe_ratio=1.20,
        max_drawdown=-0.10,
        positive_window_rate=0.75,
    )

    second_result = create_result(
        strategy="momentum",
        total_return=0.05,
        sharpe_ratio=0.40,
        max_drawdown=-0.18,
        positive_window_rate=0.50,
    )

    save_walk_forward_result(
        first_result,
        output_dir=results_dir,
    )

    save_walk_forward_result(
        second_result,
        output_dir=results_dir,
    )

    comparison = compare_experiments(
        results_dir
    )

    assert not comparison.empty
    assert set(comparison["strategy"]) == {
        "moving_average",
        "momentum",
    }


def test_stable_strategy_is_selected_from_saved_experiments(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "walk_forward"

    for _ in range(3):
        result = create_result(
            strategy="moving_average",
            total_return=0.20,
            sharpe_ratio=1.20,
            max_drawdown=-0.10,
            positive_window_rate=0.75,
        )

        save_walk_forward_result(
            result,
            output_dir=results_dir,
        )

    result = create_result(
        strategy="momentum",
        total_return=0.03,
        sharpe_ratio=0.20,
        max_drawdown=-0.25,
        positive_window_rate=0.40,
    )

    save_walk_forward_result(
        result,
        output_dir=results_dir,
    )

    stability_report = build_stability_report(
        results_dir
    )

    assert not stability_report.empty
    assert "stability_score" in (
        stability_report.columns
    )

    stable_strategy = select_stable_strategy(
        results_dir
    )

    assert stable_strategy == "moving_average"
