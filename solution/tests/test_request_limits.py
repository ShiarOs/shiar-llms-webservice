"""Oversized input must fail before expensive model or retrieval work."""

import importlib

import pytest
from pydantic import ValidationError


@pytest.mark.parametrize("module_name", ["llm_app.models", "solution.llm_app.models"])
@pytest.mark.parametrize(
    "values",
    [
        {"prompt": "x" * 8001},
        {"prompt": "hello", "model": "x" * 129},
    ],
)
def test_generation_input_is_bounded(module_name, values):
    model = importlib.import_module(module_name).GenerateRequest
    with pytest.raises(ValidationError):
        model(**values)


@pytest.mark.parametrize(
    "module_name", ["llm_app.rag_models", "solution.llm_app.models"]
)
def test_rag_question_is_bounded(module_name):
    model = importlib.import_module(module_name).RagGenerateRequest
    with pytest.raises(ValidationError):
        model(question="x" * 2001)
