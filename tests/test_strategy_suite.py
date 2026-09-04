import pandas as pd

from src.evaluation.strategy_suite import run_default_strategy_suite


def test_default_strategy_suite_returns_both_strategies():
    df = pd.DataFrame({
        "close": [
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
            123,
            124,
            125,
            126,
            127,
            128,
            129,
            130,
            131,
            132,
            133,
            134,
            135,
            136,
            137,
            138,
            139,
            140,
            141,
            142,
            143,
            144,
            145,
            146,
            147,
            148,
            149,
            150,
            151,
            152,
            153,
            154,
            155,
        ],
    })

    df["return"] = df["close"].pct_change()

    results = run_default_strategy_suite(df)

    assert set(results.keys()) == {
        "moving_average",
        "momentum",
    }


def test_default_strategy_suite_returns_evaluation_reports():
    df = pd.DataFrame({
        "close": list(range(100, 156)),
    })

    df["return"] = df["close"].pct_change()

    results = run_default_strategy_suite(df)

    for report in results.values():
        assert isinstance(report, dict)
        assert "total_return" in report
        assert "max_drawdown" in report
        assert "sharpe_ratio" in report
        assert "calmar_ratio" in report
        assert "sortino_ratio" in report
        assert "exposure" in report
        assert "win_rate" in report
        assert "profit_factor" in report


def test_default_strategy_suite_does_not_modify_input():
    df = pd.DataFrame({
        "close": list(range(100, 156)),
    })

    df["return"] = df["close"].pct_change()

    original = df.copy()

    run_default_strategy_suite(df)

    pd.testing.assert_frame_equal(df, original)


def test_default_strategy_suite_rejects_invalid_input():
    try:
        run_default_strategy_suite("invalid")
    except TypeError as exc:
        assert str(exc) == "df must be a pandas DataFrame."
    else:
        raise AssertionError("Expected TypeError was not raised.")
