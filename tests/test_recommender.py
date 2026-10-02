"""Checks for the small, interpretable historical recommendation model."""
import pandas as pd

from pitwall.recommender import comparable_rows, evaluate_chronologically, recommend, strategy_table


def _history():
    """Small same-circuit sample with two repeated observed tyre plans."""
    return pd.DataFrame(
        [
            {"year": 2023, "race_date": "2023-05-01", "circuit": "Example", "team_family": "Ferrari", "grid": 10, "strategy": "MEDIUM → HARD", "stops": 1, "stop_laps": "20", "finish_position": 7, "finish_gain": 3},
            {"year": 2023, "race_date": "2023-05-01", "circuit": "Example", "team_family": "Mercedes", "grid": 11, "strategy": "MEDIUM → HARD", "stops": 1, "stop_laps": "22", "finish_position": 8, "finish_gain": 3},
            {"year": 2024, "race_date": "2024-05-01", "circuit": "Example", "team_family": "Ferrari", "grid": 9, "strategy": "SOFT → HARD", "stops": 1, "stop_laps": "15", "finish_position": 10, "finish_gain": -1},
            {"year": 2024, "race_date": "2024-05-01", "circuit": "Example", "team_family": "Williams", "grid": 10, "strategy": "SOFT → HARD", "stops": 1, "stop_laps": "17", "finish_position": 12, "finish_gain": -2},
            {"year": 2025, "race_date": "2025-05-01", "circuit": "Example", "team_family": "Ferrari", "grid": 10, "strategy": "MEDIUM → HARD", "stops": 1, "stop_laps": "21", "finish_position": 6, "finish_gain": 4},
        ]
    )


def test_strategy_table_returns_observed_plans_with_windows_and_evidence():
    table = strategy_table(_history(), "Example", "Ferrari", 10)

    assert list(table["strategy"]) == ["MEDIUM → HARD", "SOFT → HARD"]
    primary = table.iloc[0]
    assert primary["stops"] == 1
    assert primary["pit_windows"][0]["median_lap"] in {20, 21}
    assert primary["expected_finish"] < 10
    assert primary["raw_examples"] == 3
    assert 0 <= primary["points_probability"] <= 1


def test_recommend_falls_back_to_a_single_observed_example_when_needed():
    one_row = _history().iloc[[0]]

    result = recommend(one_row, "Example", "Ferrari", 10)

    assert len(result) == 1
    assert result.iloc[0]["evidence"].startswith("limited")


def test_strategy_table_excludes_weather_dependent_plans_without_weather_input():
    wet_row = _history().iloc[[0]].assign(strategy="INTERMEDIATE → WET")
    table = strategy_table(pd.concat([_history(), wet_row]), "Example", "Ferrari", 10)

    assert not table["strategy"].str.contains("INTERMEDIATE|WET").any()


def test_comparable_rows_exposes_the_historic_examples_behind_a_plan():
    rows = comparable_rows(_history(), "Example", "Ferrari", 10, "MEDIUM → HARD")

    assert len(rows) == 3
    assert set(rows["strategy"]) == {"MEDIUM → HARD"}
    assert "weight" in rows


def test_chronological_evaluation_only_scores_already_seen_strategies():
    report = evaluate_chronologically(_history())

    # The first race has no prior circuit history; only the 2025 repeated plan is seen.
    assert report["attempted_rows"] == 3
    assert report["evaluated_rows"] == 1
    assert report["outcome_mae"] >= 0
    assert report["grid_baseline_mae"] >= 0
