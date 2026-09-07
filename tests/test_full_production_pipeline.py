from pathlib import Path

import pandas as pd

from src.evaluation.production_pipeline import (
    run_production_pipeline,
)


def test_full_production_pipeline_with_saved_runs(
    tmp_path: Path,
) -> None:
    data_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "raw"
        / "xauusd_daily_2025.csv"
    )

    if not data_path.exists():
        return

    walk_forward_dir = tmp_path / "walk_forward"
    production_dir = tmp_path / "production"

    run_dir = (
        walk_forward_dir
        / "20260907_000000_000000_test"
    )
    run_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "strategy": ["moving_average"],
            "rank": [1],
            "total_return": [0.10],
            "max_drawdown": [-0.05],
            "sharpe_ratio": [1.5],
            "positive_window_rate": [0.75],
        }
    ).to_csv(
        run_dir / "final_report.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "strategy": ["moving_average"],
        }
    ).to_csv(
        run_dir / "eligible_strategies.csv",
        index=False,
    )

    (run_dir / "metadata.json").write_text(
        '{"created_at_utc": '
        '"2026-09-07T00:00:00+00:00", '
        '"best_strategy": "moving_average", '
        '"strategies": ["moving_average"]}',
        encoding="utf-8",
    )

    result = run_production_pipeline(
        data_path=data_path,
        results_dir=walk_forward_dir,
        production_dir=production_dir,
        save_result=False,
    )

    assert result["strategy"] == "moving_average"
    assert result["backtest"] is not None
    assert isinstance(
        result["backtest"],
        pd.DataFrame,
    )
    assert result["report"]["observations"] > 0
