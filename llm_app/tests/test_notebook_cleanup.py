"""The starter must release Qdrant even after partial initialization."""

import json
import runpy
from pathlib import Path

import pytest
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models


@pytest.fixture
def notebook_scripts(tmp_path) -> dict[str, Path]:
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / "rag-extension-starter.ipynb").read_text())
    scripts = {}
    for cell in notebook["cells"]:
        if cell["id"] in {"05-cell-12", "05-cell-close"}:
            path = tmp_path / f"{cell['id']}.py"
            path.write_text("".join(cell["source"]))
            scripts[cell["id"]] = path
    return scripts


def test_cleanup_before_initialization_is_safe(notebook_scripts):
    runpy.run_path(str(notebook_scripts["05-cell-close"]))
    runpy.run_path(str(notebook_scripts["05-cell-close"]))


def test_cleanup_after_embedding_failure_releases_index(notebook_scripts, tmp_path):
    class FailedEmbeddings:
        def __init__(self, **kwargs):
            pass

        def embed_query(self, text):
            raise ConnectionError("Ollama unavailable in this offline test")

    directory = tmp_path / "index"
    clients: list[QdrantClient] = []

    def local_client(**kwargs):
        client = QdrantClient(**kwargs)
        clients.append(client)
        return client

    namespace = {
        "OllamaEmbeddings": FailedEmbeddings,
        "OLLAMA_EMBED_MODEL": "offline-test",
        "OLLAMA_BASE_URL": "http://127.0.0.1:1",
        "OLLAMA_TIMEOUT": 120.0,
        "QdrantClient": local_client,
        "models": models,
        "QDRANT_PERSIST_DIR": directory,
        "QDRANT_COLLECTION": "test",
        "chunks": [Document(page_content="Synthetic evidence")],
    }
    try:
        with pytest.raises(ConnectionError, match="Ollama unavailable"):
            runpy.run_path(str(notebook_scripts["05-cell-12"]), init_globals=namespace)
        assert len(clients) == 1
        for _ in range(2):
            runpy.run_path(
                str(notebook_scripts["05-cell-close"]),
                init_globals={"client": clients[0]},
            )
        reopened = QdrantClient(path=str(directory))
        reopened.close()
    finally:
        for client in clients:
            client.close()
