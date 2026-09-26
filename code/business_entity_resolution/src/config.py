"""Central paths. Override any of them with environment variables (e.g. on Kaggle)."""
import os
import sys
from pathlib import Path

# The Windows console defaults to cp1252 and crashes on Indic/accented text.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Raw TSVs as shipped by the organizers (contains train/ and test/).
RAW_DIR = Path(os.environ.get(
    "ER_RAW_DIR",
    r"C:\Users\heman\OneDrive - Indian Institute of Technology Patna\my acads"
    r"\hacthons\ml challaenge\dataset\student_resource\dataset",
))

# Parquet copies of the raw TSVs (fast to load).
PARQUET_DIR = Path(os.environ.get("ER_PARQUET_DIR", r"C:\ml2026\data\parquet"))

# Intermediate artefacts (candidates, features, models).
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", r"C:\ml2026\data\work"))

# Final submission files.
OUTPUT_DIR = Path(os.environ.get(
    "ER_OUTPUT_DIR", Path(__file__).resolve().parents[1] / "output"
))

SOURCE_FILES = {
    ("train", 1): "train/train_source1.tsv",
    ("train", 2): "train/train_source2.tsv",
    ("train", 3): "train/train_source3.tsv",
    ("test", 1): "test/test_source1.tsv",
    ("test", 2): "test/test_source2.tsv",
    ("test", 3): "test/test_source3.tsv",
}
GROUND_TRUTH_FILE = "train/train_ground_truth.tsv"


def parquet_path(split: str, source: int) -> Path:
    return PARQUET_DIR / f"{split}_source{source}.parquet"


def ground_truth_parquet() -> Path:
    return PARQUET_DIR / "train_ground_truth.parquet"
