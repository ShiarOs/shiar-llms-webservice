"""Starter functions for the climate document RAG extension."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore

from llm_app.config import MIN_RELEVANCE, QDRANT_COLLECTION, QDRANT_PERSIST_DIR
from llm_app.rag_models import RagGenerateResponse


class Retriever(Protocol):
    """Minimal async retrieval interface used by the RAG generation path."""

    async def ainvoke(self, question: str) -> Sequence[Document]: ...


class AsyncLanguageModel(Protocol):
    """Minimal async generation interface used by the RAG generation path."""

    async def ainvoke(self, prompt: str) -> str: ...


def load_vector_store(
    embedding_fn: Embeddings,
    collection_name: str = QDRANT_COLLECTION,
    persist_directory: Path = QDRANT_PERSIST_DIR,
) -> QdrantVectorStore:
    """Load the persisted collection created in the starter notebook."""
    raise NotImplementedError("Complete this function after notebook section 3")


def build_retriever(
    vector_store: QdrantVectorStore, k: int = 5, min_relevance: float = MIN_RELEVANCE
) -> Retriever:
    """Build a retriever that returns only sufficiently relevant chunks."""
    raise NotImplementedError("Complete this function after notebook section 4")


async def rag_generation(
    question: str,
    retriever: Retriever,
    llm: AsyncLanguageModel,
) -> RagGenerateResponse:
    """Retrieve context and generate an answer with source page labels."""
    raise NotImplementedError("Complete this function after notebook section 5")
