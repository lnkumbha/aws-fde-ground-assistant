from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", PROJECT_ROOT / "data" / "documents"))
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.12"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "3"))
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "120"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "25"))
