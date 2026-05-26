import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
VECTOR_DB_DIR = PROJECT_ROOT / "chroma_db"
INDEX_MANIFEST = OUTPUT_DIR / "index_manifest.json"
CHUNK_EXPORT_PATH = OUTPUT_DIR / "chunks.jsonl"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 6
KEYWORD_TOP_K = 4
MAX_CHUNKS_PER_SOURCE = 2
EMBED_BATCH_SIZE = 32

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "phi3:mini")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

APP_ACCESS_PASSWORD = os.getenv("APP_ACCESS_PASSWORD", "")
APP_SESSION_TIMEOUT_MINUTES = int(os.getenv("APP_SESSION_TIMEOUT_MINUTES", "20"))
APP_SHOW_DETAILED_ERRORS = (
    os.getenv("APP_SHOW_DETAILED_ERRORS", "false").lower() == "true"
)
GITHUB_REPO_URL = "https://github.com/msa-1988/rag-for-technical-documents"
