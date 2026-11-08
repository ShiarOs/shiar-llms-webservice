"""FastAPI routes for direct generation and climate-report questions."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from llm_app.rag_models import RagGenerateRequest, RagGenerateResponse
from llm_app.rag_utils import answer_climate_question, close_vector_store
from llm_app.models import GenerateRequest, GenerateResponse
from llm_app.utils import redis_client, send_prompt


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        try:
            await asyncio.to_thread(close_vector_store)
        finally:
            await redis_client.aclose()


app = FastAPI(title="Local LLM Web Service", lifespan=lifespan)


@app.get("/")
def home() -> dict[str, str]:
    """Return a small health-style response."""
    return {"message": "Local LLM service is ready"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(payload: GenerateRequest) -> GenerateResponse:
    """Generate text with the configured local Ollama model."""
    return await send_prompt(payload)


@app.post("/climate-change", response_model=RagGenerateResponse)
async def climate_change(payload: RagGenerateRequest) -> RagGenerateResponse:
    """Answer a question using retrieved passages from the climate report."""
    return await answer_climate_question(payload.question)

# """FastAPI routes supplied as the starting point."""

# from contextlib import asynccontextmanager

# from fastapi import FastAPI

# from llm_app.models import GenerateRequest, GenerateResponse
# from llm_app.utils import redis_client, send_prompt


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     try:
#         yield
#     finally:
#         await redis_client.aclose()


# app = FastAPI(title="Local LLM Web Service", lifespan=lifespan)


# @app.get("/")
# def home() -> dict[str, str]:
#     """Return a small health-style response."""
#     return {"message": "Local LLM service is ready"}


# @app.post("/generate", response_model=GenerateResponse)
# async def generate_text(payload: GenerateRequest) -> GenerateResponse:
#     """Generate text with the configured local Ollama model."""
#     return await send_prompt(payload)
