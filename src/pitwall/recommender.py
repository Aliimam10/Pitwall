"""Interpretable, empirical strategy recommendations from historic race rows.

There is deliberately no claim that this finds a counterfactual optimum.  It
summarises outcomes of similar observed starts and applies small-sample
shrinkage so a one-off result cannot dominate a recommendation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from pitwall.config import EVALUATION_PATH

WET_COMPOUNDS = ("INTERMEDIATE", "WET")


def _weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=float)
    weights = np.asarray(list(weights), dtype=float)
    return float(np.average(values, weights=weights))


def _weighted_quantile(
    values: Iterable[float], weights: Iterable[float], quantile: float
) -> float:
    """A short weighted quantile implementation, sufficient for small samples."""
    values = np.asarray(list(values), dtype=float)
    weights = np.asarray(list(weights), dtype=float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return float(np.interp(quantile, cumulative, values))


def _effective_n(weights: pd.Series) -> float:
    total = float(weights.sum())
    return total**2 / float((weights**2).sum()) if total else 0.0


def _dry_history(history: pd.DataFrame) -> pd.DataFrame:
    """Keep pre-race dry plans; weather is deliberately not an app input."""
    wet_pattern = "|".join(WET_COMPOUNDS)
    return history[~history["strategy"].str.contains(wet_pattern, na=False)].copy()


def _parse_stops(value: object) -> list[int]:
    if pd.isna(value) or value == "":
        return []
    return [int(part) for part in str(value).split(",")]


def _pit_windows(group: pd.DataFrame) -> tuple[list[dict[str, int]], str]:
    """Summarise the stop lap(s) observed for one tyre-sequence strategy."""
    stops = _parse_stops(group.iloc[0]["stop_laps"])
    if not stops:
        return [], "No planned tyre-change stop"

    windows: list[dict[str, int]] = []
    for stop_index in range(len(stops)):
        values = [
            _parse_stops(row.stop_laps)[stop_index]
            for row in group.itertuples()
            if len(_parse_stops(row.stop_laps)) > stop_index
        ]
        weights = group.loc[
            [len(_parse_stops(value)) > stop_index for value in group["stop_laps"]], "weight"
        ]
        median = round(_weighted_quantile(values, weights, 0.5))
        low = round(_weighted_quantile(values, weights, 0.25))
        high = round(_weighted_quantile(values, weights, 0.75))
        windows.append({"stop": stop_index + 1, "median_lap": median, "low_lap": low, "high_lap": high})

    text = "; ".join(
        f"Stop {window['stop']}: lap {window['median_lap']} (historic IQR {window['low_lap']}–{window['high_lap']})"
        for window in windows
    )
    return windows, text


def _evidence_label(examples: int, effective_examples: float) -> str:
    if examples >= 8 and effective_examples >= 5:
        return "strong for this small dataset"
    if examples >= 4 and effective_examples >= 3:
        return "moderate"
    return "limited — treat as a loose historical guide"


def _context_pool(
    history: pd.DataFrame, circuit: str, team: str, grid: int
) -> pd.DataFrame:
    """Weight historical rows by circuit, constructor family and grid similarity."""
    pool = history[history["circuit"] == circuit].copy()
    if pool.empty:
        return pool

    # A four-place grid difference halves a row's influence roughly every 2.8 places.
    grid_weight = np.exp(-np.abs(pool["grid"] - grid) / 4.0)
    team_weight = np.where(pool["team_family"] == team, 2.5, 1.0)
    year_span = max(int(pool["year"].max() - pool["year"].min()), 1)
    recency_weight = 1.0 + 0.15 * (pool["year"] - pool["year"].min()) / year_span
    pool["weight"] = grid_weight * team_weight * recency_weight
    return pool


def strategy_table(
    history: pd.DataFrame,
    circuit: str,
    team: str,
    grid: int,
    min_strategy_examples: int = 2,
) -> pd.DataFrame:
    """Score observed strategies for a particular circuit/team/grid situation.

    The outcome is grid places gained.  Each strategy estimate is shrunk toward
    the same-circuit, similar-grid context mean with five pseudo-observations.
    This straightforward empirical-Bayes step stabilises tiny samples without
    hiding the raw number of examples.
    """
    pool = _context_pool(_dry_history(history), circuit, team, grid)
    if pool.empty:
        return pd.DataFrame()

    context_gain = _weighted_mean(pool["finish_gain"], pool["weight"])
    context_points = _weighted_mean((pool["finish_position"] <= 10).astype(float), pool["weight"])
    summaries: list[dict[str, object]] = []

    for strategy, group in pool.groupby("strategy", sort=False):
        raw_examples = len(group)
        if raw_examples < min_strategy_examples:
            continue
        weights = group["weight"]
        effective_examples = _effective_n(weights)
        gain = _weighted_mean(group["finish_gain"], weights)
        gain_sd = float(np.sqrt(_weighted_mean((group["finish_gain"] - gain) ** 2, weights)))
        # Five pseudo-examples from the local circuit/grid context, not a global F1 prior.
        shrunk_gain = (effective_examples * gain + 5 * context_gain) / (effective_examples + 5)
        points_probability = (
            effective_examples * _weighted_mean((group["finish_position"] <= 10).astype(float), weights)
            + 5 * context_points
        ) / (effective_examples + 5)
        gain_low = _weighted_quantile(group["finish_gain"], weights, 0.2)
        gain_high = _weighted_quantile(group["finish_gain"], weights, 0.8)
        windows, window_text = _pit_windows(group)
        expected_finish = float(np.clip(grid - shrunk_gain, 1, 20))
        finish_best = int(np.clip(round(grid - gain_high), 1, 20))
        finish_worst = int(np.clip(round(grid - gain_low), 1, 20))
        summaries.append(
            {
                "strategy": strategy,
                "stops": int(group["stops"].iloc[0]),
                "pit_windows": windows,
                "pit_window_text": window_text,
                "expected_finish": expected_finish,
                "expected_gain": shrunk_gain,
                "finish_range": f"P{finish_best}–P{finish_worst}",
                "points_probability": points_probability,
                "raw_examples": raw_examples,
                "effective_examples": effective_examples,
                "evidence": _evidence_label(raw_examples, effective_examples),
                "observed_gain": gain,
                # Prefer an expected gain, but gently favour less volatile samples.
                "score": shrunk_gain - 0.25 * gain_sd,
            }
        )

    table = pd.DataFrame(summaries)
    return table.sort_values(["score", "raw_examples"], ascending=False).reset_index(drop=True) if not table.empty else table


def recommend(
    history: pd.DataFrame, circuit: str, team: str, grid: int, limit: int = 3
) -> pd.DataFrame:
    """Return a primary recommendation and observed alternatives.

    If no tyre sequence has two examples, one-example sequences are shown, but
    their evidence label stays explicitly limited.
    """
    table = strategy_table(history, circuit, team, grid, min_strategy_examples=2)
    if table.empty:
        table = strategy_table(history, circuit, team, grid, min_strategy_examples=1)
    return table.head(limit).copy()


def evaluate_chronologically(rows: pd.DataFrame) -> dict[str, float | int | str]:
    """Evaluate strategy-conditioned outcome estimates without looking forward.

    For every historic driver-race, training contains only races that took place
    earlier.  The test asks: if that driver's *actual* tyre sequence had already
    been observed, how close was its estimated finish position?  It deliberately
    does not pretend the chosen alternative caused the observed race outcome.
    """
    data = _dry_history(rows).sort_values("race_date").copy()
    predictions: list[tuple[float, float, float]] = []
    attempted = 0
    for _, row in data.iterrows():
        earlier = data[data["race_date"] < row["race_date"]]
        if earlier.empty or not (earlier["circuit"] == row["circuit"]).any():
            continue
        attempted += 1
        estimates = strategy_table(
            earlier, row["circuit"], row["team_family"], int(row["grid"]), min_strategy_examples=1
        )
        actual_strategy = estimates[estimates["strategy"] == row["strategy"]]
        if actual_strategy.empty:
            continue
        predicted = float(actual_strategy.iloc[0]["expected_finish"])
        predictions.append((predicted, float(row["finish_position"]), float(row["grid"])))

    if not predictions:
        return {
            "method": "chronological dry-strategy-conditioned holdout",
            "attempted_rows": attempted,
            "evaluated_rows": 0,
            "message": "No holdout rows had an already-observed tyre sequence.",
        }
    predicted, actual, grid = (np.asarray(values) for values in zip(*predictions))
    return {
        "method": "chronological dry-strategy-conditioned holdout",
        "attempted_rows": attempted,
        "evaluated_rows": int(len(predictions)),
        "strategy_seen_coverage": round(len(predictions) / attempted, 3) if attempted else 0.0,
        "outcome_mae": round(float(np.mean(np.abs(predicted - actual))), 3),
        "grid_baseline_mae": round(float(np.mean(np.abs(grid - actual))), 3),
        "message": "MAE covers dry strategies only; it is descriptive accuracy for an observed strategy, not causal proof that a recommendation changes the result.",
    }


def write_evaluation(rows: pd.DataFrame, path: Path = EVALUATION_PATH) -> dict[str, float | int | str]:
    """Persist evaluation next to the processed dataset for the Streamlit app."""
    report = evaluate_chronologically(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    """Write and print the reproducible chronological evaluation report."""
    from pitwall.data import load_rows

    report = write_evaluation(load_rows())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
