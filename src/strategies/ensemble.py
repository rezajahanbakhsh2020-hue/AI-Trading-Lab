from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


def _validate_signal_frame(
    name: str,
    frame: pd.DataFrame,
    signal_column: str,
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"strategy '{name}' output must be a pandas DataFrame."
        )

    if signal_column not in frame.columns:
        raise ValueError(
            f"strategy '{name}' output must contain "
            f"'{signal_column}'."
        )


def _normalize_weight(value: float) -> float:
    weight = float(value)

    if pd.isna(weight):
        raise ValueError("ensemble weights must be finite.")

    if weight < 0:
        raise ValueError("ensemble weights must be non-negative.")

    return weight


def build_weighted_ensemble(
    strategy_outputs: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
    *,
    signal_column: str = "signal",
    output_column: str = "signal",
) -> pd.DataFrame:
    """
    Build a lightweight weighted-voting composite strategy.

    Each registered strategy supplies a signal series. The composite
    calculates the weighted mean of those signals and emits:

        1 -> positive consensus
        0 -> no consensus

    The function never changes the original strategy outputs.
    """

    if not isinstance(strategy_outputs, Mapping):
        raise TypeError("strategy_outputs must be a mapping.")

    if not strategy_outputs:
        raise ValueError("strategy_outputs must not be empty.")

    if not isinstance(weights, Mapping):
        raise TypeError("weights must be a mapping.")

    if not weights:
        raise ValueError("weights must not be empty.")

    selected_names = tuple(weights.keys())

    missing = [
        name
        for name in selected_names
        if name not in strategy_outputs
    ]

    if missing:
        raise ValueError(
            "Missing strategy outputs: "
            + ", ".join(str(name) for name in missing)
        )

    normalized_weights = {
        name: _normalize_weight(weights[name])
        for name in selected_names
    }

    total_weight = sum(normalized_weights.values())

    if total_weight <= 0:
        raise ValueError("at least one ensemble weight must be positive.")

    first = strategy_outputs[selected_names[0]]
    _validate_signal_frame(
        selected_names[0],
        first,
        signal_column,
    )

    result = pd.DataFrame(index=first.index)

    weighted_signal = pd.Series(
        0.0,
        index=first.index,
        dtype=float,
    )

    for name in selected_names:
        frame = strategy_outputs[name]

        _validate_signal_frame(
            name,
            frame,
            signal_column,
        )

        if len(frame) != len(first):
            raise ValueError(
                "All strategy outputs must have the same length."
            )

        if not frame.index.equals(first.index):
            raise ValueError(
                "All strategy outputs must have the same index."
            )

        signal = pd.to_numeric(
            frame[signal_column],
            errors="coerce",
        ).fillna(0.0)

        weight = normalized_weights[name]

        weighted_signal = weighted_signal.add(
            signal * weight,
            fill_value=0.0,
        )

    weighted_signal = weighted_signal / total_weight

    result["ensemble_score"] = weighted_signal
    result[output_column] = (
        weighted_signal > 0.0
    ).astype(int)

    return result


def equal_weights(
    names: Sequence[str],
) -> dict[str, float]:
    """
    Create equal weights for a collection of strategy names.
    """

    normalized = [
        str(name).strip().lower()
        for name in names
        if str(name).strip()
    ]

    if not normalized:
        raise ValueError("names must not be empty.")

    if len(set(normalized)) != len(normalized):
        raise ValueError("strategy names must be unique.")

    return {
        name: 1.0
        for name in normalized
    }


def ranking_weights(
    evaluations: pd.DataFrame,
    *,
    top_n: int = 3,
    score_column: str = "ranking_score",
    name_column: str = "name",
) -> dict[str, float]:
    """
    Convert ranked strategy scores into non-negative ensemble weights.

    Only the selected Top-N candidates are included.

    Negative ranking scores receive zero weight. If all selected
    scores are non-positive, equal weights are used instead.
    """

    if not isinstance(evaluations, pd.DataFrame):
        raise TypeError("evaluations must be a pandas DataFrame.")

    if not isinstance(top_n, int):
        raise TypeError("top_n must be an integer.")

    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    required = {name_column, score_column}

    missing = required.difference(evaluations.columns)

    if missing:
        raise ValueError(
            "evaluations is missing columns: "
            + ", ".join(sorted(missing))
        )

    selected = evaluations.head(top_n)

    if selected.empty:
        raise ValueError("evaluations must contain at least one row.")

    raw = {
        str(row[name_column]): max(
            0.0,
            float(row[score_column]),
        )
        for _, row in selected.iterrows()
    }

    if sum(raw.values()) <= 0:
        return equal_weights(tuple(raw.keys()))

    return raw


def build_ensemble_from_evaluations(
    strategy_outputs: Mapping[str, pd.DataFrame],
    evaluations: pd.DataFrame,
    *,
    top_n: int = 3,
    signal_column: str = "signal",
    output_column: str = "signal",
) -> pd.DataFrame:
    """
    Build a Top-N ranking-weighted composite.

    This function only constructs the composite. It does not claim
    that the composite is superior to its components.
    """

    weights = ranking_weights(
        evaluations,
        top_n=top_n,
    )

    return build_weighted_ensemble(
        strategy_outputs,
        weights,
        signal_column=signal_column,
        output_column=output_column,
    )
