"""Request and response contracts for direct text generation."""

from pydantic import BaseModel, ConfigDict, Field

from llm_app.config import OLLAMA_LLM_MODEL


class GenerateRequest(BaseModel):
    """Validated generation parameters supplied by an API client."""

    model_config = ConfigDict(str_strip_whitespace=True)

    prompt: str = Field(
        min_length=1,
        max_length=8000,
        description="The input prompt for text generation",
        examples=["Explain asynchronous APIs in one sentence."],
    )
    model: str = Field(
        default=OLLAMA_LLM_MODEL,
        min_length=1,
        max_length=128,
        description="The Ollama model used for generation",
    )
    max_tokens: int = Field(
        default=200,
        ge=1,
        le=4096,
        description="Maximum number of output tokens",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )


class GenerateResponse(BaseModel):
    """Text returned by the selected local model."""

    model: str
    response: str
