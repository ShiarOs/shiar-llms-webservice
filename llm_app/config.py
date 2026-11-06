"""Environment-backed configuration for the application."""

import math
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3.5:2b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))

QDRANT_PERSIST_DIR = Path(
    os.getenv("QDRANT_PERSIST_DIR", "./llm_app/qdrant_climate_db")
)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "climate-rag")
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.60"))

if not math.isfinite(OLLAMA_TIMEOUT) or OLLAMA_TIMEOUT <= 0:
    raise ValueError("OLLAMA_TIMEOUT must be finite and positive")
if CACHE_TTL <= 0:
    raise ValueError("CACHE_TTL must be positive")
if not 0 <= MIN_RELEVANCE <= 1:
    raise ValueError("MIN_RELEVANCE must be between 0 and 1")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
