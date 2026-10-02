"""Download OpenF1 race data and turn it into one driver-race strategy table.

The raw API response is cached locally.  This makes the data collection step
repeatable and avoids repeatedly asking OpenF1 for the same historic records.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from pitwall.config import (
    API_ROOT,
    PROCESSED_DIR,
    RAW_DIR,
    REQUEST_SPACING_SECONDS,
    ROWS_PATH,
    SUMMARY_PATH,
    YEARS,
)


# These are constructor continuity labels, not strategy assumptions.  They let
# a selector such as "Sauber" use the team's earlier Alfa Romeo/Kick Sauber rows.
TEAM_FAMILIES = {
    "AlphaTauri": "Racing Bulls",
    "RB": "Racing Bulls",
    "Visa Cash App RB": "Racing Bulls",
    "Visa Cash App RB F1 Team": "Racing Bulls",
    "Racing Bulls": "Racing Bulls",
    "Alfa Romeo": "Sauber",
    "Kick Sauber": "Sauber",
    "Stake F1 Team": "Sauber",
    "Stake F1 Team Kick Sauber": "Sauber",
    "Sauber": "Sauber",
}


def team_family(team: str | None) -> str:
    """Return a stable constructor label while retaining the raw team label."""
    team = team or "Unknown"
    return TEAM_FAMILIES.get(team, team)


class OpenF1Client:
    """Very small rate-limited client for the public OpenF1 API."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.last_request = 0.0

    def get(self, endpoint: str, **params: Any) -> list[dict[str, Any]]:
        wait = REQUEST_SPACING_SECONDS - (time.monotonic() - self.last_request)
        if wait > 0:
            time.sleep(wait)

        for attempt in range(7):
            response = self.session.get(
                f"{API_ROOT}/{endpoint}", params=params, timeout=45
            )
            self.last_request = time.monotonic()
            if response.status_code != 429:
                response.raise_for_status()
                payload = response.json()
                return payload if isinstance(payload, list) else []
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 1.5 * (attempt + 1)
            except ValueError:
                delay = 1.5 * (attempt + 1)
            time.sleep(delay)
        raise RuntimeError(f"OpenF1 kept rate-limiting {endpoint} with {params}")


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return json.load(handle)


def _write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(records, handle)


def race_sessions(client: OpenF1Client, force: bool = False) -> list[dict[str, Any]]:
    """Get completed Grand Prix races only (not practice, qualifying or sprint)."""
    races: list[dict[str, Any]] = []
    for year in YEARS:
        cache = RAW_DIR / f"sessions_{year}.json"
        records = _read_json(cache) if cache.exists() and not force else client.get("sessions", year=year)
        if force or not cache.exists():
            _write_json(cache, records)
        races.extend(
            row
            for row in records
            if row.get("session_name") == "Race" and not row.get("is_cancelled", False)
        )
    return sorted(races, key=lambda row: row["date_start"])


def download_raw(
    force: bool = False, max_sessions: int | None = None
) -> list[dict[str, Any]]:
    """Cache strategy-relevant endpoint responses for every race.

    ``max_sessions`` is useful when a short-running environment needs to resume
    a long public-API collection in chunks. Cached, complete sessions are
    skipped on the next invocation.
    """
    client = OpenF1Client()
    races = race_sessions(client, force=force)
    endpoints = ("drivers", "stints", "session_result", "position")
    cached_this_run = 0
    for index, race in enumerate(races, start=1):
        session_key = race["session_key"]
        missing = [
            endpoint
            for endpoint in endpoints
            if force or not (RAW_DIR / str(session_key) / f"{endpoint}.json").exists()
        ]
        if not missing:
            continue
        if max_sessions is not None and cached_this_run >= max_sessions:
            break
        for endpoint in missing:
            cache = RAW_DIR / str(session_key) / f"{endpoint}.json"
            records = client.get(endpoint, session_key=session_key)
            _write_json(cache, records)
        cached_this_run += 1
        print(f"Cached {index}/{len(races)}: {race['circuit_short_name']} {race['year']}")
    return races


def _opening_grid(records: list[dict[str, Any]]) -> dict[int, int]:
    """OpenF1's first position snapshot is the recorded grid order.

    The API has no populated historic starting-grid endpoint for these seasons.
    Sorting the position feed by timestamp and taking each driver's first value
    produces the Bahrain 2023 order, for example, before the race begins.
    """
    first: dict[int, tuple[str, int]] = {}
    for row in records:
        driver = row.get("driver_number")
        position = row.get("position")
        timestamp = row.get("date", "")
        if driver is None or position is None:
            continue
        driver = int(driver)
        if driver not in first or timestamp < first[driver][0]:
            first[driver] = (timestamp, int(position))
    return {driver: position for driver, (_, position) in first.items()}


def _strategy_by_driver(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Create an ordered tyre sequence and approximate pit laps from stints."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in records:
        if row.get("driver_number") is not None:
            grouped.setdefault(int(row["driver_number"]), []).append(row)

    strategies: dict[int, dict[str, Any]] = {}
    for driver, stints in grouped.items():
        stints.sort(key=lambda row: (row.get("stint_number", 0), row.get("lap_start", 0)))
        compounds = [str(row.get("compound") or "UNKNOWN").upper() for row in stints]
        stops = [max(1, int(row["lap_start"]) - 1) for row in stints[1:] if row.get("lap_start")]
        strategies[driver] = {
            "strategy": " → ".join(compounds),
            "stops": len(stops),
            "stop_laps": ",".join(str(lap) for lap in stops),
        }
    return strategies


def build_strategy_rows(races: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    """Combine cached endpoints into an analysis-ready driver-race table."""
    if races is None:
        races = race_sessions(OpenF1Client())

    rows: list[dict[str, Any]] = []
    for race in races:
        cache_dir = RAW_DIR / str(race["session_key"])
        drivers = {
            int(row["driver_number"]): row
            for row in _read_json(cache_dir / "drivers.json")
            if row.get("driver_number") is not None
        }
        grid = _opening_grid(_read_json(cache_dir / "position.json"))
        strategies = _strategy_by_driver(_read_json(cache_dir / "stints.json"))
        results = _read_json(cache_dir / "session_result.json")

        for result in results:
            driver_number = result.get("driver_number")
            if driver_number is None:
                continue
            driver_number = int(driver_number)
            if driver_number not in grid or driver_number not in strategies:
                continue  # Cannot model a start or a tyre plan the API did not record.
            driver = drivers.get(driver_number, {})
            raw_team = driver.get("team_name", "Unknown")
            finish_position = result.get("position")
            finished = not any(result.get(flag, False) for flag in ("dnf", "dns", "dsq"))
            row = {
                "year": int(race["year"]),
                "race_date": race["date_start"],
                "session_key": int(race["session_key"]),
                "meeting_key": int(race["meeting_key"]),
                "circuit": race["circuit_short_name"],
                "country": race.get("country_name", ""),
                "driver_number": driver_number,
                "driver": driver.get("full_name", str(driver_number)),
                "team": raw_team,
                "team_family": team_family(raw_team),
                "grid": grid[driver_number],
                "finish_position": finish_position,
                "finish_gain": (grid[driver_number] - finish_position)
                if finished and finish_position is not None
                else None,
                "points": result.get("points", 0.0),
                "finished": finished,
                "total_laps": result.get("number_of_laps"),
                **strategies[driver_number],
            }
            rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No strategy rows were built; run the download command first.")
    frame = frame.sort_values(["race_date", "driver_number"]).reset_index(drop=True)
    frame["grid"] = frame["grid"].astype(int)
    frame["finish_position"] = pd.to_numeric(frame["finish_position"], errors="coerce")
    frame["finish_gain"] = pd.to_numeric(frame["finish_gain"], errors="coerce")
    return frame


def write_processed_data(rows: pd.DataFrame) -> None:
    """Save the compact table that the app reads and a transparent data summary."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(ROWS_PATH, index=False)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seasons": list(YEARS),
        "race_sessions": int(rows["session_key"].nunique()),
        "driver_race_rows": int(len(rows)),
        "classified_finisher_rows": int(rows["finished"].sum()),
        "circuits": int(rows["circuit"].nunique()),
        "source": "https://api.openf1.org/v1",
        "grid_method": "First timestamped OpenF1 position record per driver",
        "excluded_from_outcome_model": "DNS, DNF and DSQ rows",
    }
    with SUMMARY_PATH.open("w") as handle:
        json.dump(summary, handle, indent=2)


def refresh(force: bool = False) -> pd.DataFrame:
    """Run the complete reproducible raw-data-to-table step."""
    races = download_raw(force=force)
    rows = build_strategy_rows(races)
    write_processed_data(rows)
    return rows


def load_rows() -> pd.DataFrame:
    """Load eligible historical starts for recommendations and evaluation."""
    rows = pd.read_csv(ROWS_PATH, parse_dates=["race_date"])
    return rows[rows["finished"] & rows["finish_gain"].notna()].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare PitWall's OpenF1 data.")
    parser.add_argument("--force", action="store_true", help="refresh existing raw API caches")
    parser.add_argument("--download-only", action="store_true", help="cache API responses without building CSV")
    parser.add_argument(
        "--max-sessions",
        type=int,
        help="cache at most this many incomplete sessions, then exit cleanly",
    )
    parser.add_argument("--build-only", action="store_true", help="build CSV from an existing raw cache")
    args = parser.parse_args()
    if args.download_only and args.build_only:
        parser.error("--download-only and --build-only cannot be used together")
    if args.max_sessions is not None and not args.download_only:
        parser.error("--max-sessions requires --download-only")
    if args.build_only:
        rows = build_strategy_rows(race_sessions(OpenF1Client()))
        write_processed_data(rows)
    elif args.download_only:
        races = download_raw(force=args.force, max_sessions=args.max_sessions)
        complete = sum(
            all((RAW_DIR / str(race["session_key"]) / f"{endpoint}.json").exists()
            for endpoint in ("drivers", "stints", "session_result", "position"))
            for race in races
        )
        print(f"Raw cache complete for {complete}/{len(races)} sessions")
        return
    else:
        rows = refresh(force=args.force)
    print(f"Wrote {len(rows)} driver-race rows to {ROWS_PATH}")


if __name__ == "__main__":
    main()
