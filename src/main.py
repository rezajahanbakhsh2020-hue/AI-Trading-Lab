from data.loader import load_csv
from features.indicators import add_returns
from strategies.baseline import baseline_signal
from backtest.engine import run_backtest


def main() -> None:
    print("AI-Trading-Lab")
    print("Backtest pipeline is ready.")


if __name__ == "__main__":
    main()
