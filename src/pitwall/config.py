"""Shared paths and deliberately small project configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ROWS_PATH = PROCESSED_DIR / "strategy_rows.csv"
SUMMARY_PATH = PROCESSED_DIR / "dataset_summary.json"
EVALUATION_PATH = PROCESSED_DIR / "evaluation.json"

API_ROOT = "https://api.openf1.org/v1"
YEARS = (2023, 2024, 2025)
# Deliberately below OpenF1's documented three-request/second ceiling. Large
# position payloads can still trigger brief bursts of 429 responses.
REQUEST_SPACING_SECONDS = 0.55
