from pathlib import Path

from src.evaluation.production_pipeline import (
    run_production_pipeline,
)


def main() -> None:
    root = Path(__file__).resolve().parent

    data_path = (
        root
        / "data"
        / "raw"
        / "xauusd_daily_2025.csv"
    )

    walk_forward_dir = (
        root
        / "results"
        / "walk_forward"
    )

    production_dir = (
        root
        / "results"
        / "production"
    )

    result = run_production_pipeline(
        data_path=data_path,
        results_dir=walk_forward_dir,
        production_dir=production_dir,
        save_result=True,
    )

    print("AI-Trading-Lab")
    print("Production pipeline completed.")
    print()
    print(
        f"Selected Strategy: "
        f"{result['strategy']}"
    )
    print(
        f"Stability Score: "
        f"{result['stability_score']:.4f}"
    )
    print()
    print("Production Report:")

    for key, value in result["report"].items():
        print(f"{key}: {value}")

    print()

    saved_result = result.get("saved_result")

    if saved_result:
        print(
            f"Production Result Saved: "
            f"{saved_result['run_dir']}"
        )


if __name__ == "__main__":
    main()
