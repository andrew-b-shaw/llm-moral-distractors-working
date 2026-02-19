"""PATH CONFIG"""

from pathlib import Path

# Locate root path
ROOT = Path(__file__).parent.parent

# SET PATHS
PATH_API_KEYS = ROOT / "api_keys"
PATH_HF_CACHE = ROOT / "cache"
PATH_OFFLOAD = ROOT / "offload"

PATH_DATA = ROOT / "data"
PATH_SCENARIOS = PATH_DATA / "scenarios"
PATH_RESULTS = PATH_DATA / "responses"
PATH_CSV_RESULTS = PATH_DATA / "csv_results"
PATH_QUESTION_TEMPLATES = PATH_DATA / "question_templates"
PATH_RESPONSE_TEMPLATES = PATH_DATA / "response_templates"
