"""Central path configuration for the project. All data, cache, and output paths are
defined relative to the repository root."""

from pathlib import Path

# Locate root path
ROOT = Path(__file__).parent.parent

# SET PATHS
PATH_API_KEYS = ROOT / "api_keys"
PATH_HF_CACHE = ROOT / "cache"
PATH_OFFLOAD = ROOT / "offload"

PATH_DATA = ROOT / "data"
PATH_SCENARIOS = PATH_DATA / "scenarios"
PATH_DISTRACTORS = PATH_DATA / "distractors"
PATH_RESULTS = PATH_DATA / "responses"
PATH_CSV_RESULTS = PATH_DATA / "csv_results"
PATH_BATCH = PATH_DATA / "batch"
PATH_FIG = ROOT / "fig"
