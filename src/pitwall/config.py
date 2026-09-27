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
REQUEST_SPACING_SECONDS = 0.38  # OpenF1 documents a three-request/second limit.
