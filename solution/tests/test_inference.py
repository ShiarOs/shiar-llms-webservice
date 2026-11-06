"""Tests for reference inference failure boundaries."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from solution.llm_app.inference import send_prompt
from solution.llm_app.models import GenerateRequest


def test_ollama_timeout_becomes_gateway_timeout() -> None:
    payload = GenerateRequest(prompt="Hello")
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
    with (
        patch(
            "solution.llm_app.inference.redis_client.get",
            new=AsyncMock(return_value=None),
        ),
        patch("solution.llm_app.inference.OllamaLLM", return_value=llm),
        pytest.raises(HTTPException) as exc_info,
    ):
        asyncio.run(send_prompt(payload))

    assert exc_info.value.status_code == 504
