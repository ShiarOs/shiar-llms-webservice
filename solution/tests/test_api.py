"""Endpoint tests for the complete reference service."""

import asyncio
from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient, Response

from solution.llm_app.fastapi_llms import app
from solution.llm_app.models import GenerateResponse, RagGenerateResponse


async def post(path: str, payload: dict[str, object]) -> Response:
    """Send an in-process POST request to the reference ASGI application."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.post(path, json=payload)


@patch(
    "solution.llm_app.fastapi_llms.send_prompt",
    new_callable=AsyncMock,
    return_value=GenerateResponse(model="qwen3.5:2b", response="Generated"),
)
def test_generate_endpoint(mock_send: AsyncMock) -> None:
    response = asyncio.run(post("/generate", {"prompt": "Hello"}))

    assert response.status_code == 200
    assert response.json()["model"] == "qwen3.5:2b"
    mock_send.assert_awaited_once()


@patch(
    "solution.llm_app.fastapi_llms.answer_climate_question",
    new_callable=AsyncMock,
    return_value=RagGenerateResponse(
        answer="Agriculture is exposed to changing rainfall [page-9].",
        sources=["page-9"],
    ),
)
def test_climate_endpoint_returns_sources(mock_answer: AsyncMock) -> None:
    response = asyncio.run(
        post(
            "/climate-change",
            {"question": "How can climate change affect agriculture?"},
        )
    )

    assert response.status_code == 200
    assert response.json()["sources"] == ["page-9"]
    mock_answer.assert_awaited_once()


def test_generate_rejects_whitespace_prompt() -> None:
    response = asyncio.run(post("/generate", {"prompt": "   "}))

    assert response.status_code == 422
