"""Focused tests for cache identity and inference failure boundaries."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import redis.asyncio as redis
from fastapi import HTTPException

from llm_app.models import GenerateRequest
from llm_app.utils import build_cache_key, send_prompt


def test_cache_key_changes_with_generation_parameters() -> None:
    baseline = GenerateRequest(prompt="Hello")
    changed = GenerateRequest(prompt="Hello", temperature=0.8)

    assert build_cache_key(baseline) != build_cache_key(changed)


def test_cache_key_includes_all_request_fields() -> None:
    baseline = GenerateRequest(prompt="Hello")

    assert build_cache_key(baseline) != build_cache_key(
        GenerateRequest(prompt="Hello", model="other-model")
    )
    assert build_cache_key(baseline) != build_cache_key(
        GenerateRequest(prompt="Different")
    )
    assert build_cache_key(baseline) != build_cache_key(
        GenerateRequest(prompt="Hello", max_tokens=201)
    )


def test_cache_hit_skips_ollama() -> None:
    payload = GenerateRequest(prompt="Hello")
    with (
        patch("llm_app.utils.redis_client.get", new=AsyncMock(return_value="Cached")),
        patch("llm_app.utils.OllamaLLM") as llm_class,
    ):
        response = asyncio.run(send_prompt(payload))

    assert response.response == "Cached"
    llm_class.assert_not_called()


def test_cache_miss_generates_and_stores_response() -> None:
    payload = GenerateRequest(prompt="Hello")
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value="Generated")
    with (
        patch("llm_app.utils.redis_client.get", new=AsyncMock(return_value=None)),
        patch("llm_app.utils.redis_client.set", new=AsyncMock()) as cache_set,
        patch("llm_app.utils.OllamaLLM", return_value=llm),
    ):
        response = asyncio.run(send_prompt(payload))

    assert response.response == "Generated"
    cache_set.assert_awaited_once()


def test_redis_failure_falls_back_to_ollama() -> None:
    payload = GenerateRequest(prompt="Hello")
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value="Generated")
    cache_error = redis.RedisError("unavailable")
    with (
        patch("llm_app.utils.redis_client.get", new=AsyncMock(side_effect=cache_error)),
        patch("llm_app.utils.redis_client.set", new=AsyncMock()),
        patch("llm_app.utils.OllamaLLM", return_value=llm),
    ):
        response = asyncio.run(send_prompt(payload))

    assert response.response == "Generated"


def test_ollama_failure_becomes_bad_gateway() -> None:
    payload = GenerateRequest(prompt="Hello")
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=RuntimeError("offline"))
    with (
        patch("llm_app.utils.redis_client.get", new=AsyncMock(return_value=None)),
        patch("llm_app.utils.OllamaLLM", return_value=llm),
        pytest.raises(HTTPException) as exc_info,
    ):
        asyncio.run(send_prompt(payload))

    assert exc_info.value.status_code == 502


def test_ollama_timeout_becomes_gateway_timeout() -> None:
    payload = GenerateRequest(prompt="Hello")
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
    with (
        patch("llm_app.utils.redis_client.get", new=AsyncMock(return_value=None)),
        patch("llm_app.utils.OllamaLLM", return_value=llm),
        pytest.raises(HTTPException) as exc_info,
    ):
        asyncio.run(send_prompt(payload))

    assert exc_info.value.status_code == 504
