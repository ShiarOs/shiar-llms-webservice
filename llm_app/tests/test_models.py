"""Tests for the application's Pydantic contracts."""

import pytest
from pydantic import ValidationError

from llm_app.models import GenerateRequest, GenerateResponse
from llm_app.rag_models import RagGenerateRequest, RagGenerateResponse


def test_generation_request_uses_local_model_defaults() -> None:
    request = GenerateRequest(prompt="Hello")

    assert request.model == "qwen3.5:2b"
    assert request.max_tokens == 200
    assert request.temperature == 0.3


@pytest.mark.parametrize("prompt", ["", "   ", None])
def test_generation_request_requires_non_empty_prompt(prompt: object) -> None:
    with pytest.raises(ValidationError):
        GenerateRequest.model_validate({"prompt": prompt})


def test_response_contracts_preserve_values() -> None:
    direct = GenerateResponse(model="qwen3.5:2b", response="Hello")
    grounded = RagGenerateResponse(answer="From the report", sources=["page-4"])

    assert direct.response == "Hello"
    assert grounded.sources == ["page-4"]


def test_rag_request_requires_non_empty_question() -> None:
    for question in ("", "   "):
        with pytest.raises(ValidationError):
            RagGenerateRequest(question=question)
