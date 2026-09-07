from pathlib import Path

from src.evaluation.live_workflow import (
    run_xauusd_walk_forward,
)
from src.evaluation.stable_strategy_workflow import (
    run_stable_strategy_selection,
)


def main() -> None:
    print("AI-Trading-Lab")
    print("Running XAUUSD walk-forward research...")

    data_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "raw"
        / "xauusd_daily_2025.csv"
    )

    result = run_xauusd_walk_forward(
        path=data_path
    )

    print("Walk-forward completed.")
    print()

    print("Final Strategy Ranking:")
    print(
        result["final_report"].to_string(
            index=False
        )
    )

    print()

    print("Eligible Strategies:")
    eligible = result[
        "eligible_strategies"
    ]

    if eligible.empty:
        print("No eligible strategy.")
    else:
        print(
            eligible.to_string(
                index=False
            )
        )

    print()

    print(
        f"Best Current Strategy: "
        f"{result['best_strategy']}"
    )

    stable_result = (
        run_stable_strategy_selection()
    )

    print()

    print("Stability Ranking:")

    stability_report = stable_result[
        "stability_report"
    ]

    if stability_report.empty:
        print(
            "No stored experiments "
            "available."
        )
    else:
        print(
            stability_report.to_string(
                index=False
            )
        )

    print()

    stable_strategy = stable_result[
        "stable_strategy"
    ]

    if stable_strategy is None:
        print(
            "Stable Strategy: N/A"
        )
    else:
        print(
            f"Stable Strategy: "
            f"{stable_strategy}"
        )


if __name__ == "__main__":
    main()
