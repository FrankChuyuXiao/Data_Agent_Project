from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "backend" / "datasets" / "housing.csv"
RESULTS_DIR = BASE_DIR / "results"
STATE_DIR = RESULTS_DIR / "states"
RESULT_CSV = RESULTS_DIR / "results.csv"
DATASET_NAME = "housing.csv"

load_dotenv(BASE_DIR / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
