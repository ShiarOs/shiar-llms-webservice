"""Request and response contracts for the reference service."""

from pydantic import BaseModel, ConfigDict, Field

from solution.llm_app.config import OLLAMA_LLM_MODEL


class GenerateRequest(BaseModel):
    """Validated direct-generation request."""

    model_config = ConfigDict(str_strip_whitespace=True)

    prompt: str = Field(min_length=1, max_length=8000)
    model: str = Field(default=OLLAMA_LLM_MODEL, min_length=1, max_length=128)
    max_tokens: int = Field(default=200, ge=1, le=4096)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    """Direct-generation response."""

    model: str
    response: str


class RagGenerateRequest(BaseModel):
    """Question to answer from the climate report."""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)


class RagGenerateResponse(BaseModel):
    """Grounded answer and source labels from retrieved chunks."""

    answer: str
    sources: list[str] = Field(default_factory=list)
