from pathlib import Path

import pandas as pd

from src.evaluation.production_pipeline import (
    run_production_pipeline,
)


def test_run_production_pipeline(
    monkeypatch,
    tmp_path: Path,
) -> None:
    backtest = pd.DataFrame(
        {
            "strategy_return": [
                0.01,
                -0.005,
                0.02,
                0.01,
            ]
        }
    )

    def fake_run_xauusd_walk_forward(
        path,
        save_result,
    ):
        return {
            "final_report": pd.DataFrame(),
            "eligible_strategies": pd.DataFrame(),
            "best_strategy": "moving_average",
        }

    def fake_run_production_backtest(
        data_path,
        results_dir,
        save_result,
    ):
        return {
            "strategy": "moving_average",
            "stability_score": 0.75,
            "stability_report": pd.DataFrame(
                {
                    "strategy": ["moving_average"],
                    "stability_score": [0.75],
                }
            ),
            "backtest": backtest,
        }

    monkeypatch.setattr(
        "src.evaluation.production_pipeline.run_xauusd_walk_forward",
        fake_run_xauusd_walk_forward,
    )

    monkeypatch.setattr(
        "src.evaluation.production_pipeline.run_production_backtest",
        fake_run_production_backtest,
    )

    result = run_production_pipeline(
        data_path=tmp_path / "data.csv",
        results_dir=tmp_path / "walk_forward",
        production_dir=tmp_path / "production",
        save_result=False,
    )

    assert result["strategy"] == "moving_average"
    assert result["stability_score"] == 0.75
    assert result["report"]["observations"] == 4
    assert "total_return" in result["report"]
    assert "max_drawdown" in result["report"]
    assert "sharpe_ratio" in result["report"]
    assert isinstance(
        result["summary"],
        pd.DataFrame,
    )
    assert result["saved_result"] is None
