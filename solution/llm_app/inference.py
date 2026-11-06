"""Asynchronous Ollama inference with resilient Redis caching."""

import asyncio
import hashlib
import json
import logging

import httpx
import redis.asyncio as redis
from fastapi import HTTPException
from langchain_ollama import OllamaLLM

from solution.llm_app.config import (
    CACHE_TTL,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
    REDIS_URL,
)
from solution.llm_app.models import GenerateRequest, GenerateResponse

logger = logging.getLogger(__name__)
redis_client = redis.from_url(
    REDIS_URL, decode_responses=True, socket_connect_timeout=1, socket_timeout=1
)


def build_cache_key(payload: GenerateRequest) -> str:
    """Hash every request value that can affect generation."""
    raw = json.dumps(
        {**payload.model_dump(mode="json"), "reasoning": False},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"llm-webservice:{digest}"


async def send_prompt(payload: GenerateRequest) -> GenerateResponse:
    """Return cached text when available, otherwise call Ollama."""
    cache_key = build_cache_key(payload)
    try:
        cached = await asyncio.wait_for(redis_client.get(cache_key), timeout=1)
        if cached is not None and cached.strip():
            return GenerateResponse(model=payload.model, response=cached)
    except (redis.RedisError, TimeoutError) as exc:
        logger.warning("Redis GET failed for key %s: %s", cache_key, exc)

    try:
        llm = OllamaLLM(
            base_url=OLLAMA_BASE_URL,
            model=payload.model,
            temperature=payload.temperature,
            num_predict=payload.max_tokens,
            # Keep the small token budget for the answer, not hidden thinking.
            reasoning=False,
            async_client_kwargs={"timeout": OLLAMA_TIMEOUT},
        )
        text = await asyncio.wait_for(llm.ainvoke(payload.prompt), OLLAMA_TIMEOUT)
    except (TimeoutError, httpx.TimeoutException) as exc:
        logger.warning("Ollama generation timed out")
        raise HTTPException(status_code=504, detail="Ollama request timed out") from exc
    except Exception as exc:
        logger.exception("Ollama generation failed")
        raise HTTPException(status_code=502, detail="Ollama generation failed") from exc

    if not text.strip():
        raise HTTPException(status_code=502, detail="Ollama returned an empty response")

    try:
        await asyncio.wait_for(
            redis_client.set(cache_key, text, ex=CACHE_TTL), timeout=1
        )
    except (redis.RedisError, TimeoutError) as exc:
        logger.warning("Redis SET failed for key %s: %s", cache_key, exc)

    return GenerateResponse(model=payload.model, response=text)
