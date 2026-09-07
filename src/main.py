from pathlib import Path

from src.evaluation.live_workflow import (
    run_xauusd_walk_forward,
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
        path=data_path,
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

    eligible = result["eligible_strategies"]

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
        f"Best Strategy: "
        f"{result['best_strategy']}"
    )


if __name__ == "__main__":
    main()
