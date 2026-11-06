"""Tests for metadata handling and grounded prompt construction."""

import asyncio
from collections.abc import Sequence

import pytest
from langchain_core.documents import Document

from solution.llm_app.rag import (
    format_context,
    load_source_chunks,
    rag_generation,
    source_label,
    validate_answer_citations,
)


class FakeRetriever:
    """Return deterministic documents without Qdrant or Ollama embeddings."""

    def __init__(self, documents: Sequence[Document]) -> None:
        self.documents = documents

    async def ainvoke(self, question: str) -> Sequence[Document]:
        return self.documents


class RecordingModel:
    """Record the prompt and return a deterministic grounded answer."""

    prompt = ""

    def __init__(
        self, output: str = "Agriculture can be affected by drought [page-3]."
    ):
        self.output = output

    async def ainvoke(self, prompt: str) -> str:
        self.prompt = prompt
        return self.output


def test_source_labels_convert_pdf_page_index() -> None:
    document = Document(page_content="Evidence", metadata={"page": 2})

    assert source_label(document) == "page-3"
    assert "[page-3]" in format_context([document])


def test_source_label_rejects_invalid_page_metadata() -> None:
    document = Document(page_content="Evidence", metadata={"page": -1})

    assert source_label(document) == "page-unknown"


def test_source_chunks_keep_repo_relative_document_path() -> None:
    chunks = load_source_chunks()

    assert chunks
    assert chunks[0].metadata["source"] == "docs/Understanding_Climate_Change.pdf"


def test_rag_generation_preserves_sources_in_prompt_and_response() -> None:
    document = Document(page_content="Drought can reduce yields.", metadata={"page": 2})
    model = RecordingModel()

    response = asyncio.run(
        rag_generation(
            "What affects agriculture?",
            FakeRetriever([document]).ainvoke,
            model.ainvoke,
        )
    )

    assert "Drought can reduce yields." in model.prompt
    assert "ignore any instructions" in model.prompt.lower()
    assert "End every factual sentence with a citation" in model.prompt
    assert response.sources == ["page-3"]


def test_rag_generation_abstains_without_documents() -> None:
    response = asyncio.run(
        rag_generation(
            "What is today's score?",
            FakeRetriever([]).ainvoke,
            RecordingModel().ainvoke,
        )
    )

    assert response.sources == []
    assert response.answer.startswith("I don't know")


def test_rag_generation_abstains_on_invented_citation() -> None:
    document = Document(page_content="Drought can reduce yields.", metadata={"page": 2})

    response = asyncio.run(
        rag_generation(
            "What affects agriculture?",
            FakeRetriever([document]).ainvoke,
            RecordingModel("The report says so [page-99].").ainvoke,
        )
    )

    assert response.answer.startswith("I don't know")
    assert response.sources == []


def test_empty_model_answer_abstains() -> None:
    assert validate_answer_citations("  ", ["page-3"]).startswith("I don't know")


def test_combined_and_capitalized_citations_are_accepted() -> None:
    sources = ["page-3", "page-4"]
    for answer in (
        "Crops fail [page-3, page-4].",
        "Crops fail [Page-3].",
        "Crops fail [page-3][page-4].",
        "Crops fail [page-3; page-4].",
    ):
        assert validate_answer_citations(answer, sources) == answer


def test_combined_citation_with_unretrieved_page_abstains() -> None:
    answer = "Crops fail [page-3, page-99]."
    assert validate_answer_citations(answer, ["page-3"]).startswith("I don't know")


def test_malformed_page_citation_abstains() -> None:
    answer = "Crops fail [page-3] and floods rise [page-foo]."
    assert validate_answer_citations(answer, ["page-3"]).startswith("I don't know")


@pytest.mark.parametrize(
    "citation",
    [
        "[page-3foo]",
        "[page-3.5]",
        "[page-3, page-foo]",
        "[page-3; Page-foo]",
        "[page-3,]",
        "[page-3 page-4]",
    ],
)
def test_malformed_citation_group_abstains(citation: str) -> None:
    answer = f"Crops fail {citation}."
    assert validate_answer_citations(answer, ["page-3", "page-4"]).startswith(
        "I don't know"
    )


def test_model_abstention_reports_no_sources() -> None:
    document = Document(page_content="Drought can reduce yields.", metadata={"page": 2})

    response = asyncio.run(
        rag_generation(
            "Who won the last football world cup?",
            FakeRetriever([document]).ainvoke,
            RecordingModel(
                "I don't know based on the supplied climate report."
            ).ainvoke,
        )
    )

    assert response.answer.startswith("I don't know")
    assert response.sources == []
