"""The exercise notebook must use the same settings as the application."""

import json
import runpy
from pathlib import Path
from unittest.mock import Mock

from llm_app import config


def test_starter_notebook_uses_service_settings(monkeypatch, tmp_path):
    settings = {
        "OLLAMA_BASE_URL": "http://127.0.0.1:12345",
        "OLLAMA_LLM_MODEL": "custom-generation",
        "OLLAMA_EMBED_MODEL": "custom-embeddings",
        "OLLAMA_TIMEOUT": 25.0,
        "QDRANT_COLLECTION": "custom-collection",
        "QDRANT_PERSIST_DIR": tmp_path / "custom-index",
        "MIN_RELEVANCE": 0.72,
        "CHUNK_SIZE": 333,
        "CHUNK_OVERLAP": 30,
    }
    for name, value in settings.items():
        monkeypatch.setattr(config, name, value)

    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / "rag-extension-starter.ipynb").read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    script = tmp_path / "notebook_settings.py"
    script.write_text(cells["05-cell-04"])
    namespace = runpy.run_path(str(script))

    assert {name: namespace[name] for name in settings} == settings

    embeddings = Mock()
    generation = Mock()
    namespace.update(chunks=[], OllamaEmbeddings=embeddings, OllamaLLM=generation)
    for cell_id in ("05-cell-12", "05-cell-19"):
        script.write_text(cells[cell_id])
        namespace = runpy.run_path(str(script), init_globals=namespace)

    assert embeddings.call_args.kwargs["client_kwargs"] == {"timeout": 25.0}
    assert embeddings.call_args.kwargs["async_client_kwargs"] == {"timeout": 25.0}
    assert generation.call_args.kwargs["async_client_kwargs"] == {"timeout": 25.0}
