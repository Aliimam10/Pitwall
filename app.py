"""FastAPI adapter for PitWall's existing historical strategy analysis."""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pitwall.config import EVALUATION_PATH, ROWS_PATH, SUMMARY_PATH  # noqa: E402
from pitwall.data import load_rows  # noqa: E402
from pitwall.recommender import comparable_rows, recommend  # noqa: E402


@lru_cache
def get_rows() -> pd.DataFrame:
    if not ROWS_PATH.exists():
        raise FileNotFoundError("Run `python -m pitwall.data` before starting the interface.")
    return load_rows()


@lru_cache
def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def empty_table(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def analyse(circuit: str, team: str, grid: int):
    """Turn three user controls into recommendation, alternatives and evidence."""
    rows = get_rows()
    table = recommend(rows, circuit, team, int(grid))
    if table.empty:
        message = "## No comparable dry-race evidence\nTry another circuit; this circuit has no usable classified rows."
        return message, "—", empty_table([]), empty_table([]), "No evaluation report found."

    primary = table.iloc[0]
    result = f"""## Historically preferred plan: {primary['strategy']}

**Estimated finish:** P{primary['expected_finish']:.1f} ({primary['expected_gain']:+.1f} places from the grid)<br>
**Historic points chance:** {primary['points_probability']:.0%}<br>
**Evidence:** {primary['raw_examples']} historic examples — *{primary['evidence']}*<br>

Comparable historic outcomes span **{primary['finish_range']}**. This is uncertainty, not a guarantee.
"""
    pit_window = f"### Pit-window guide\n{primary['pit_window_text']}"
    alternatives = table.iloc[1:][
        ["strategy", "stops", "pit_window_text", "expected_finish", "points_probability", "raw_examples", "evidence"]
    ].copy()
    alternatives.columns = ["tyre plan", "stops", "historic pit window", "expected finish", "points chance", "examples", "evidence"]
    alternatives["expected finish"] = alternatives["expected finish"].map(lambda value: f"P{value:.1f}")
    alternatives["points chance"] = alternatives["points chance"].map(lambda value: f"{value:.0%}")

    evidence = comparable_rows(rows, circuit, team, int(grid), primary["strategy"])[
        ["year", "driver", "team", "grid", "stop_laps", "finish_position", "finish_gain", "weight"]
    ].copy()
    evidence.columns = ["year", "driver", "team", "grid", "pit lap(s)", "finish", "places gained", "similarity weight"]
    evidence["similarity weight"] = evidence["similarity weight"].round(2)

    evaluation = "No evaluation report found. Run `python -m pitwall.recommender`."
    if EVALUATION_PATH.exists():
        report = read_json(EVALUATION_PATH)
        evaluation = (
            f"{report['message']}  \n\n"
            f"**Evaluable holdouts:** {report.get('evaluated_rows', '—')} · "
            f"**Outcome MAE:** {report.get('outcome_mae', '—')} · "
            f"**Grid-only baseline MAE:** {report.get('grid_baseline_mae', '—')}"
        )
    return result, pit_window, alternatives, evidence, evaluation


server = FastAPI(title="PitWall")


@server.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "pitwall_ui.html").read_text(encoding="utf-8")


@server.get("/api/circuits")
def circuits() -> JSONResponse:
    rows = get_rows()
    data = rows[["circuit", "country"]].drop_duplicates().sort_values("circuit").to_dict(orient="records")
    return JSONResponse(data)


@server.get("/api/teams")
def teams() -> JSONResponse:
    return JSONResponse(sorted(get_rows()["team_family"].unique().tolist()))


@server.get("/api/analyse")
def api_analyse(
    circuit: str = Query(...), team: str = Query(...), grid: int = Query(10, ge=1, le=20)
) -> JSONResponse:
    rows = get_rows()
    table = recommend(rows, circuit, team, grid)
    if table.empty:
        return JSONResponse({"error": "no_data"})

    primary = table.iloc[0]
    alternatives = table.iloc[1:][
        ["strategy", "stops", "pit_window_text", "expected_finish", "points_probability", "raw_examples", "evidence"]
    ].rename(
        columns={
            "strategy": "tyres",
            "pit_window_text": "window",
            "expected_finish": "finish",
            "points_probability": "points",
            "raw_examples": "examples",
            "evidence": "quality",
        }
    )
    evidence = comparable_rows(rows, circuit, team, grid, primary["strategy"])[
        ["year", "driver", "team", "grid", "stop_laps", "finish_position", "finish_gain", "weight"]
    ]
    return JSONResponse(
        {
            "primary": {
                "strategy": [part.strip() for part in str(primary["strategy"]).split("→")],
                "finish": round(float(primary["expected_finish"]), 1),
                "gain": round(float(primary["expected_gain"]), 1),
                "points": round(float(primary["points_probability"]) * 100),
                "range": str(primary["finish_range"]),
                "pit_text": str(primary["pit_window_text"]),
                "examples": int(primary["raw_examples"]),
                "quality": str(primary["evidence"]),
            },
            "alternatives": alternatives.to_dict(orient="records"),
            "evidence": evidence.to_dict(orient="records"),
        }
    )


if __name__ == "__main__":
    uvicorn.run(server, host="0.0.0.0", port=7860)
