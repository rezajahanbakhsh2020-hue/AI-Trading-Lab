from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "walk_forward"
)


def load_experiment(
    run_dir: str | Path,
) -> dict:
    run_path = Path(run_dir)

    final_report_path = (
        run_path / "final_report.csv"
    )
    eligible_path = (
        run_path / "eligible_strategies.csv"
    )
    metadata_path = run_path / "metadata.json"

    if not final_report_path.exists():
        raise FileNotFoundError(
            f"Missing final report: "
            f"{final_report_path}"
        )

    if not eligible_path.exists():
        raise FileNotFoundError(
            f"Missing eligible strategies: "
            f"{eligible_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata: "
            f"{metadata_path}"
        )

    final_report = pd.read_csv(
        final_report_path
    )

    eligible = pd.read_csv(
        eligible_path
    )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    return {
        "run_dir": str(run_path),
        "final_report": final_report,
        "eligible_strategies": eligible,
        "metadata": metadata,
    }


def find_experiments(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> list[Path]:
    root = Path(results_dir)

    if not root.exists():
        return []

    experiments = []

    for path in root.iterdir():
        if not path.is_dir():
            continue

        if (
            (path / "final_report.csv").exists()
            and (path / "metadata.json").exists()
        ):
            experiments.append(path)

    return sorted(
        experiments,
        reverse=True,
    )


def compare_experiments(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    experiments = find_experiments(
        results_dir
    )

    rows: list[dict] = []

    for run_dir in experiments:
        experiment = load_experiment(
            run_dir
        )

        report = experiment["final_report"]
        metadata = experiment["metadata"]

        for _, row in report.iterrows():
            rows.append(
                {
                    "run_id": run_dir.name,
                    "created_at_utc": metadata.get(
                        "created_at_utc"
                    ),
                    "strategy": row.get(
                        "strategy"
                    ),
                    "rank": row.get("rank"),
                    "total_return": row.get(
                        "total_return"
                    ),
                    "max_drawdown": row.get(
                        "max_drawdown"
                    ),
                    "sharpe_ratio": row.get(
                        "sharpe_ratio"
                    ),
                    "sortino_ratio": row.get(
                        "sortino_ratio"
                    ),
                    "calmar_ratio": row.get(
                        "calmar_ratio"
                    ),
                    "positive_window_rate": row.get(
                        "positive_window_rate"
                    ),
                    "best_strategy": metadata.get(
                        "best_strategy"
                    ),
                }
            )

    columns = [
        "run_id",
        "created_at_utc",
        "strategy",
        "rank",
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "positive_window_rate",
        "best_strategy",
    ]

    if not rows:
        return pd.DataFrame(
            columns=columns
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "created_at_utc",
                "rank",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def summarize_experiments(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    comparison = compare_experiments(
        results_dir
    )

    if comparison.empty:
        return pd.DataFrame(
            columns=[
                "strategy",
                "experiments",
                "avg_total_return",
                "avg_max_drawdown",
                "avg_sharpe_ratio",
                "avg_positive_window_rate",
                "best_rank",
            ]
        )

    summary = (
        comparison.groupby(
            "strategy",
            as_index=False,
        )
        .agg(
            experiments=(
                "run_id",
                "nunique",
            ),
            avg_total_return=(
                "total_return",
                "mean",
            ),
            avg_max_drawdown=(
                "max_drawdown",
                "mean",
            ),
            avg_sharpe_ratio=(
                "sharpe_ratio",
                "mean",
            ),
            avg_positive_window_rate=(
                "positive_window_rate",
                "mean",
            ),
            best_rank=(
                "rank",
                "min",
            ),
        )
        .sort_values(
            [
                "avg_total_return",
                "avg_sharpe_ratio",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return summary
