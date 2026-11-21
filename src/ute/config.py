"""Paths, column names and seeds shared by the pipeline."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "criteo-research-uplift-v2.1.csv.gz"
PARQUET = DATA_DIR / "criteo_uplift.parquet"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
DOCS_DIR = ROOT / "docs"

# Official dataset is hosted by Criteo on Hugging Face (the old go.criteo.net link is gone).
DATA_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
    "criteo-research-uplift-v2.1.csv.gz"
)

FEATURES = [f"f{i}" for i in range(12)]
TREATMENT = "treatment"
OUTCOMES = ("conversion", "visit")
# Recorded *after* the treatment was assigned. Never valid model inputs (see README, Leakage).
POST_TREATMENT = ["exposure", "visit"]

SEED = 42
SPLIT = (0.6, 0.2, 0.2)  # train / valid / test
BUDGETS = (0.1, 0.2, 0.3)
