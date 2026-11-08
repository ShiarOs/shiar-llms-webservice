"""The exercise notebook must use the same settings as the application."""

import ast
import asyncio
import inspect
import json
import runpy
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from llm_app import config


def run_notebook_cell(source, namespace, filename):
    code = compile(source, str(filename), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    result = eval(code, namespace)
    if inspect.isawaitable(result):
        asyncio.run(result)


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
    namespace.update(
        chunks=[],
        retrieved_docs=[],
        OllamaEmbeddings=embeddings,
        OllamaLLM=generation,
    )
    for cell_id in ("05-cell-12", "05-cell-19"):
        run_notebook_cell(cells[cell_id], namespace, script)

    assert embeddings.call_args.kwargs["client_kwargs"] == {"timeout": 25.0}
    assert embeddings.call_args.kwargs["async_client_kwargs"] == {"timeout": 25.0}
    assert generation.call_args.kwargs["async_client_kwargs"] == {"timeout": 25.0}


@pytest.mark.parametrize("has_documents", [False, True])
def test_generation_cell_handles_notebook_await(tmp_path, has_documents):
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / "rag-extension-starter.ipynb").read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    model = Mock()
    model.ainvoke = AsyncMock(return_value="Answer [page-1]")
    namespace = {
        "OllamaLLM": Mock(return_value=model),
        "OLLAMA_LLM_MODEL": "test-model",
        "OLLAMA_BASE_URL": "http://127.0.0.1:1",
        "OLLAMA_TIMEOUT": 25.0,
        "retrieved_docs": [object()] if has_documents else [],
        "prompt": "test prompt",
    }

    run_notebook_cell(cells["05-cell-19"], namespace, tmp_path / "generation.py")

    if has_documents:
        model.ainvoke.assert_awaited_once_with("test prompt")
        assert namespace["answer"] == "Answer [page-1]"
    else:
        model.ainvoke.assert_not_awaited()
        assert namespace["answer"] == "I don't know based on the supplied climate report."
