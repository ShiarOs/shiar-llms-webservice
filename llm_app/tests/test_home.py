"""Tests for the application's root endpoint."""

import asyncio

from httpx import ASGITransport, AsyncClient, Response

from llm_app.fastapi_llms import app


async def get(path: str) -> Response:
    """Send an in-process GET request to the ASGI application."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.get(path)


def test_home_returns_service_status() -> None:
    response = asyncio.run(get("/"))

    assert response.status_code == 200
    assert response.json() == {"message": "Local LLM service is ready"}
