"""Check real deadlines and retrieval boundaries, not just mocked timeout errors."""

import asyncio
import importlib
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore

from solution.llm_app import rag


@pytest.mark.parametrize(
    "module_name", ["llm_app.fastapi_llms", "solution.llm_app.fastapi_llms"]
)
def test_api_shutdown_closes_redis_pool(module_name, monkeypatch):
    module = importlib.import_module(module_name)
    close = AsyncMock()
    monkeypatch.setattr(module.redis_client, "aclose", close)
    with TestClient(module.app) as client:
        assert client.get("/").status_code == 200
    close.assert_awaited_once()


@pytest.mark.parametrize("module_name", ["llm_app.utils", "solution.llm_app.inference"])
@pytest.mark.parametrize("empty", ["", "  \n"])
def test_empty_generation_is_not_cached(module_name, empty, monkeypatch):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module.redis_client, "get", AsyncMock(return_value=None))
    cache_set = AsyncMock()
    monkeypatch.setattr(module.redis_client, "set", cache_set)
    model = MagicMock(return_value=MagicMock(ainvoke=AsyncMock(return_value=empty)))
    monkeypatch.setattr(module, "OllamaLLM", model)
    with pytest.raises(HTTPException) as error:
        asyncio.run(module.send_prompt(module.GenerateRequest(prompt="Hello")))
    assert error.value.status_code == 502
    assert model.call_args.kwargs["reasoning"] is False
    cache_set.assert_not_awaited()


@pytest.mark.parametrize("module_name", ["llm_app.utils", "solution.llm_app.inference"])
def test_empty_cached_value_is_replaced(module_name, monkeypatch):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module.redis_client, "get", AsyncMock(return_value=" "))
    cache_set = AsyncMock()
    monkeypatch.setattr(module.redis_client, "set", cache_set)
    monkeypatch.setattr(
        module,
        "OllamaLLM",
        MagicMock(return_value=MagicMock(ainvoke=AsyncMock(return_value="Answer"))),
    )
    result = asyncio.run(module.send_prompt(module.GenerateRequest(prompt="Hello")))
    assert result.response == "Answer"
    cache_set.assert_awaited_once()


@pytest.mark.parametrize("module_name", ["llm_app.utils", "solution.llm_app.inference"])
def test_generation_deadline_cancels_slow_provider(module_name, monkeypatch):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module.redis_client, "get", AsyncMock(return_value=None))
    monkeypatch.setattr(module, "OLLAMA_TIMEOUT", 0.01)

    async def slow(_prompt):
        await asyncio.sleep(10)
        return "Late answer"

    monkeypatch.setattr(
        module, "OllamaLLM", MagicMock(return_value=MagicMock(ainvoke=slow))
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(module.send_prompt(module.GenerateRequest(prompt="Hello")))
    assert error.value.status_code == 504


@pytest.mark.parametrize("module_name", ["llm_app.utils", "solution.llm_app.inference"])
def test_redis_write_failure_keeps_generated_response(module_name, monkeypatch):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module.redis_client, "get", AsyncMock(return_value=None))
    monkeypatch.setattr(
        module.redis_client,
        "set",
        AsyncMock(side_effect=module.redis.RedisError("offline")),
    )
    monkeypatch.setattr(
        module,
        "OllamaLLM",
        MagicMock(return_value=MagicMock(ainvoke=AsyncMock(return_value="Answer"))),
    )
    response = asyncio.run(module.send_prompt(module.GenerateRequest(prompt="Hello")))
    assert response.response == "Answer"


@pytest.mark.parametrize(
    "answer",
    [
        "Uncited claim",
        "A claim [page-unknown]",
        "A claim [page-99]",
        "Mixed [page-1] [page-unknown]",
    ],
)
def test_missing_or_invalid_citation_abstains(answer):
    assert rag.validate_answer_citations(answer, ["page-1"]) == rag.ABSTENTION


def test_index_failure_becomes_502(monkeypatch):
    monkeypatch.setattr(
        rag, "load_vector_store", MagicMock(side_effect=RuntimeError("private detail"))
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(rag.answer_climate_question("Question"))
    assert error.value.status_code == 502
    assert "private detail" not in error.value.detail


def test_rag_generation_deadline(monkeypatch):
    model = MagicMock()
    monkeypatch.setattr(rag, "OllamaLLM", model)
    monkeypatch.setattr(rag, "load_vector_store", MagicMock())
    monkeypatch.setattr(rag, "build_retriever", MagicMock())
    monkeypatch.setattr(rag, "OLLAMA_TIMEOUT", 0.01)

    async def slow(*_args):
        await asyncio.sleep(10)

    monkeypatch.setattr(rag, "rag_generation", slow)
    with pytest.raises(HTTPException) as error:
        asyncio.run(rag.answer_climate_question("Question"))
    assert error.value.status_code == 504
    assert model.call_args.kwargs["num_predict"] == 256


class ConstantEmbeddings(Embeddings):
    """Keep Qdrant/PDF integration real without a model download."""

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def test_real_pdf_qdrant_index_is_repeatable(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "QDRANT_PERSIST_DIR", tmp_path / "qdrant")
    monkeypatch.setattr(rag, "OllamaEmbeddings", lambda **kwargs: ConstantEmbeddings())
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            stores = list(pool.map(lambda _: rag.load_vector_store(), range(8)))
        store = stores[0]
        assert all(item is store for item in stores)
        count = store.client.count(rag.QDRANT_COLLECTION).count
        assert count == len(rag.load_source_chunks())
        rag.close_vector_store()
        reopened = rag.load_vector_store()
        assert reopened.client.count(rag.QDRANT_COLLECTION).count == count
        documents = asyncio.run(rag.build_retriever(reopened).ainvoke("Climate"))
        assert documents
        assert all(type(doc.metadata["page"]) is int for doc in documents)
    finally:
        rag.close_vector_store()


def test_failed_initialization_releases_qdrant_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "QDRANT_PERSIST_DIR", tmp_path / "qdrant")
    broken = ConstantEmbeddings()
    monkeypatch.setattr(
        broken, "embed_query", MagicMock(side_effect=RuntimeError("offline"))
    )
    monkeypatch.setattr(rag, "OllamaEmbeddings", lambda **kwargs: broken)
    with pytest.raises(RuntimeError, match="offline"):
        rag.load_vector_store()
    monkeypatch.setattr(rag, "OllamaEmbeddings", lambda **kwargs: ConstantEmbeddings())
    try:
        assert rag.load_vector_store().client.count(rag.QDRANT_COLLECTION).count > 0
    finally:
        rag.close_vector_store()


def test_qdrant_threshold_uses_raw_cosine(monkeypatch):
    embeddings = ConstantEmbeddings()
    monkeypatch.setattr(
        embeddings,
        "embed_documents",
        lambda texts: [
            [1.0, 0.0] if text == "supported" else [0.5, 0.75**0.5] for text in texts
        ],
    )
    store = QdrantVectorStore.from_documents(
        [Document(page_content="supported"), Document(page_content="weak")],
        embedding=embeddings,
        location=":memory:",
        collection_name="threshold-test",
    )
    try:
        documents = rag.build_retriever(store, min_relevance=0.60).invoke("Climate")
        assert [doc.page_content for doc in documents] == ["supported"]
        monkeypatch.setattr(embeddings, "embed_query", lambda text: [0.0, -1.0])
        assert rag.build_retriever(store, min_relevance=0.60).invoke("Unrelated") == []
    finally:
        store.client.close()


def test_interrupted_indexing_is_completed_on_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "QDRANT_PERSIST_DIR", tmp_path / "qdrant")
    monkeypatch.setattr(rag, "OllamaEmbeddings", lambda **kwargs: ConstantEmbeddings())
    add_documents = QdrantVectorStore.add_documents

    def interrupted(store, documents, *, ids):
        add_documents(store, documents[:1], ids=ids[:1])
        raise RuntimeError("interrupted")

    monkeypatch.setattr(QdrantVectorStore, "add_documents", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        rag.load_vector_store()
    monkeypatch.setattr(QdrantVectorStore, "add_documents", add_documents)
    try:
        store = rag.load_vector_store()
        assert store.client.count(rag.QDRANT_COLLECTION).count == len(
            rag.load_source_chunks()
        )
    finally:
        rag.close_vector_store()


@pytest.mark.parametrize(
    "change", ["pdf", "shorter_pdf", "chunk_size", "overlap", "model", "legacy"]
)
def test_changed_index_requires_explicit_rebuild(change, tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "QDRANT_PERSIST_DIR", tmp_path / "qdrant")
    monkeypatch.setattr(rag, "OllamaEmbeddings", lambda **kwargs: ConstantEmbeddings())
    try:
        store = rag.load_vector_store()
        count = store.client.count(rag.QDRANT_COLLECTION).count
        rag.close_vector_store()
        if change in {"pdf", "shorter_pdf"}:
            source = tmp_path / "changed.pdf"
            source.write_bytes(b"different" if change == "pdf" else b"")
            monkeypatch.setattr(rag, "DOCUMENT_PATH", source)
        elif change == "legacy":
            for manifest in rag.QDRANT_PERSIST_DIR.glob("*.fingerprint"):
                manifest.unlink()
        else:
            name = {
                "chunk_size": "CHUNK_SIZE",
                "overlap": "CHUNK_OVERLAP",
                "model": "OLLAMA_EMBED_MODEL",
            }[change]
            value = getattr(rag, name)
            monkeypatch.setattr(
                rag, name, value + 1 if isinstance(value, int) else "other-model"
            )
        with pytest.raises(rag.StaleIndexError, match="rebuild"):
            rag.load_vector_store()
        # Rejection closes the file lock without deleting existing points.
        client = rag.QdrantClient(path=str(rag.QDRANT_PERSIST_DIR))
        try:
            assert client.count(rag.QDRANT_COLLECTION).count == count
        finally:
            client.close()
    finally:
        rag.close_vector_store()


def test_stale_index_returns_actionable_conflict(monkeypatch):
    monkeypatch.setattr(
        rag,
        "load_vector_store",
        MagicMock(side_effect=rag.StaleIndexError("Please rebuild")),
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(rag.answer_climate_question("Question"))
    assert error.value.status_code == 409
    assert "rebuild" in error.value.detail
