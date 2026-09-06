from pathlib import Path

from src.pipeline import run_strategy_backtest


def main() -> None:
    print("AI-Trading-Lab")
    print("Running XAUUSD backtest...")

    data_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "raw"
        / "xauusd_daily_2025.csv"
    )

    _, report = run_strategy_backtest(str(data_path))

    print("Backtest completed.")
    print("Report:")

    for metric, value in report.items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
