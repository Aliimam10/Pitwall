"""Checks for the FastAPI adapter around the unchanged strategy analysis."""
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[1]))

from app import server

client = TestClient(server)


def test_dropdown_endpoints_return_real_choices():
    circuits = client.get("/api/circuits")
    teams = client.get("/api/teams")

    assert circuits.status_code == 200
    assert {"circuit", "country"} <= circuits.json()[0].keys()
    assert teams.status_code == 200
    assert "Ferrari" in teams.json()


def test_index_serves_the_live_api_frontend():
    page = client.get("/")

    assert page.status_code == 200
    assert "id=\"circuitSel\"" in page.text
    assert "async function loadDropdowns()" in page.text
    assert "async function runAnalysis()" in page.text
    assert "function parsePitText" in page.text


def test_analysis_endpoint_returns_live_recommendation_and_evidence():
    response = client.get("/api/analyse", params={"circuit": "Silverstone", "team": "Ferrari", "grid": 10})

    assert response.status_code == 200
    payload = response.json()
    assert payload["primary"]["strategy"]
    assert isinstance(payload["primary"]["strategy"], list)
    assert "pit_text" in payload["primary"]
    assert payload["evidence"]
