"""Keep unit tests independent of live-service settings before app imports."""

import os

os.environ.update(
    {
        "PYTHON_DOTENV_DISABLED": "1",
        "OLLAMA_BASE_URL": "http://127.0.0.1:1",
        "OLLAMA_LLM_MODEL": "qwen3.5:2b",
        "OLLAMA_EMBED_MODEL": "mxbai-embed-large",
        "OLLAMA_TIMEOUT": "120",
        "REDIS_URL": "redis://127.0.0.1:1",
        "CACHE_TTL": "3600",
        "QDRANT_COLLECTION": "climate-test",
        "MIN_RELEVANCE": "0.60",
    }
)
