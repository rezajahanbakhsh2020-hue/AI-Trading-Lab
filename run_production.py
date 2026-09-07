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

    report = result["report"]

    print("========================================")
    print("AI-Trading-Lab Production Result")
    print("========================================")
    print(
        f"Strategy: {result['strategy']}"
    )
    print(
        f"Stability Score: "
        f"{result['stability_score']:.6f}"
    )
    print(
        f"Observations: "
        f"{report['observations']}"
    )
    print(
        f"Total Return: "
        f"{report['total_return']:.6f}"
    )
    print(
        f"Max Drawdown: "
        f"{report['max_drawdown']:.6f}"
    )
    print(
        f"Sharpe Ratio: "
        f"{report['sharpe_ratio']:.6f}"
    )

    saved_result = result.get("saved_result")

    if saved_result:
        print(
            f"Production Result Saved: "
            f"{saved_result['run_dir']}"
        )

    print("========================================")
    print("PRODUCTION PIPELINE SUCCESS")
    print("========================================")


if __name__ == "__main__":
    main()
