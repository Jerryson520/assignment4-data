from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

from scripts.process_wet import OUTPUT_DIRECTORY

INPUT_DIRECTORY = Path("local-shared-data/filtered-data/final")
OUTPUT_DIRECTORY = Path("local-shared-data/filtered-data/final-exact-dedup")
REPORT_PATH = OUTPUT_DIRECTORY / "deduplication_report.json"

def normalize_for_exact_deduplication(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = text.lower()
    return " ".join(text.split())

def has_document(text: str) -> bytes:
    