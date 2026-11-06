"""Complete reference FastAPI application."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from solution.llm_app.inference import redis_client, send_prompt
from solution.llm_app.models import (
    GenerateRequest,
    GenerateResponse,
    RagGenerateRequest,
    RagGenerateResponse,
)
from solution.llm_app.rag import answer_climate_question, close_vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        try:
            await asyncio.to_thread(close_vector_store)
        finally:
            await redis_client.aclose()


app = FastAPI(title="Local LLM Web Service - Reference", lifespan=lifespan)


@app.get("/")
def home() -> dict[str, str]:
    """Return a small health-style response."""
    return {"message": "Local LLM service is ready"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(payload: GenerateRequest) -> GenerateResponse:
    """Generate text through the cached Ollama path."""
    return await send_prompt(payload)


@app.post("/climate-change", response_model=RagGenerateResponse)
async def climate_change(payload: RagGenerateRequest) -> RagGenerateResponse:
    """Answer a question from retrieved climate-report context."""
    return await answer_climate_question(payload.question)
