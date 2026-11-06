"""Tests for request validation and /generate delegation."""

import asyncio
from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient, Response

from llm_app.fastapi_llms import app
from llm_app.models import GenerateResponse

MOCK_RESPONSE = GenerateResponse(model="qwen3.5:2b", response="Mocked output")


async def post(path: str, payload: dict[str, object]) -> Response:
    """Send an in-process POST request to the ASGI application."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.post(path, json=payload)


@patch(
    "llm_app.fastapi_llms.send_prompt",
    new_callable=AsyncMock,
    return_value=MOCK_RESPONSE,
)
def test_generate_returns_typed_response(mock_send: AsyncMock) -> None:
    response = asyncio.run(post("/generate", {"prompt": "Hello"}))

    assert response.status_code == 200
    assert response.json() == {
        "model": "qwen3.5:2b",
        "response": "Mocked output",
    }
    call = mock_send.await_args
    assert call is not None
    request = call.args[0]
    assert request.model == "qwen3.5:2b"


def test_generate_rejects_missing_prompt() -> None:
    response = asyncio.run(post("/generate", {}))

    assert response.status_code == 422


def test_generate_rejects_invalid_parameters() -> None:
    response = asyncio.run(
        post(
            "/generate",
            {"prompt": "Hello", "temperature": -1, "max_tokens": 0},
        )
    )

    assert response.status_code == 422
