from pathlib import Path

import pandas as pd

from src.evaluation.production_summary import (
    build_production_summary,
    get_latest_production_result,
)


def create_production_run(
    root: Path,
    run_id: str,
    strategy: str,
    stability_score: float,
) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "strategy_return": [
                0.01,
                -0.005,
                0.02,
                0.01,
            ]
        }
    ).to_csv(
        run_dir / "backtest.csv",
        index=False,
    )

    metadata = {
        "created_at_utc": (
            f"2026-09-06T20:00:00+00:00"
        ),
        "strategy": strategy,
        "stability_score": stability_score,
        "observations": 4,
    }

    import json

    (run_dir / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def test_build_production_summary(
    tmp_path: Path,
) -> None:
    create_production_run(
        tmp_path,
        "run_001",
        "moving_average",
        0.75,
    )

    summary = build_production_summary(
        tmp_path
    )

    assert len(summary) == 1
    assert summary.iloc[0]["strategy"] == (
        "moving_average"
    )
    assert summary.iloc[0]["observations"] == 4
    assert "total_return" in summary.columns
    assert "max_drawdown" in summary.columns
    assert "sharpe_ratio" in summary.columns


def test_empty_production_summary(
    tmp_path: Path,
) -> None:
    summary = build_production_summary(
        tmp_path
    )

    assert summary.empty
    assert "total_return" in summary.columns


def test_get_latest_production_result(
    tmp_path: Path,
) -> None:
    create_production_run(
        tmp_path,
        "run_001",
        "moving_average",
        0.75,
    )

    result = get_latest_production_result(
        tmp_path
    )

    assert result is not None
    assert result["metadata"]["strategy"] == (
        "moving_average"
    )


def test_get_latest_production_result_empty(
    tmp_path: Path,
) -> None:
    result = get_latest_production_result(
        tmp_path
    )

    assert result is None
