"""Focused checks for the OpenF1-to-strategy-table transformation."""
import json

import pandas as pd

from pitwall import data


def _write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records))


def test_opening_grid_uses_the_earliest_position_for_each_driver():
    positions = [
        {"driver_number": 1, "position": 2, "date": "2024-01-01T12:01:00+00:00"},
        {"driver_number": 1, "position": 5, "date": "2024-01-01T12:02:00+00:00"},
        {"driver_number": 44, "position": 1, "date": "2024-01-01T12:01:00+00:00"},
    ]

    assert data._opening_grid(positions) == {1: 2, 44: 1}


def test_strategy_by_driver_orders_stints_and_derives_stop_laps():
    stints = [
        {"driver_number": 1, "stint_number": 2, "lap_start": 18, "compound": "HARD"},
        {"driver_number": 1, "stint_number": 1, "lap_start": 1, "compound": "medium"},
        {"driver_number": 1, "stint_number": 3, "lap_start": 40, "compound": "SOFT"},
    ]

    strategy = data._strategy_by_driver(stints)[1]

    assert strategy == {
        "strategy": "MEDIUM → HARD → SOFT",
        "stops": 2,
        "stop_laps": "17,39",
    }


def test_build_strategy_rows_keeps_status_but_only_scores_classified_finishers(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "RAW_DIR", tmp_path)
    session_dir = tmp_path / "999"
    _write(
        session_dir / "drivers.json",
        [
            {"driver_number": 1, "full_name": "Driver One", "team_name": "Alfa Romeo"},
            {"driver_number": 2, "full_name": "Driver Two", "team_name": "Ferrari"},
        ],
    )
    _write(
        session_dir / "position.json",
        [
            {"driver_number": 1, "position": 10, "date": "2024-03-01T10:00:00+00:00"},
            {"driver_number": 2, "position": 3, "date": "2024-03-01T10:00:00+00:00"},
        ],
    )
    _write(
        session_dir / "stints.json",
        [
            {"driver_number": 1, "stint_number": 1, "lap_start": 1, "compound": "MEDIUM"},
            {"driver_number": 1, "stint_number": 2, "lap_start": 20, "compound": "HARD"},
            {"driver_number": 2, "stint_number": 1, "lap_start": 1, "compound": "SOFT"},
        ],
    )
    _write(
        session_dir / "session_result.json",
        [
            {"driver_number": 1, "position": 7, "points": 6, "number_of_laps": 57, "dnf": False},
            {"driver_number": 2, "position": 18, "points": 0, "number_of_laps": 40, "dnf": True},
        ],
    )
    race = {
        "year": 2024,
        "date_start": "2024-03-01T10:00:00+00:00",
        "session_key": 999,
        "meeting_key": 1,
        "circuit_short_name": "Example",
        "country_name": "Testland",
    }

    rows = data.build_strategy_rows([race]).set_index("driver_number")

    assert rows.loc[1, "team_family"] == "Sauber"
    assert rows.loc[1, "strategy"] == "MEDIUM → HARD"
    assert rows.loc[1, "stop_laps"] == "19"
    assert rows.loc[1, "finish_gain"] == 3
    assert bool(rows.loc[2, "finished"]) is False
    assert pd.isna(rows.loc[2, "finish_gain"])
