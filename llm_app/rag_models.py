"""Request and response contracts for grounded climate questions."""

from pydantic import BaseModel, ConfigDict, Field


class RagGenerateRequest(BaseModel):
    """A question to answer from the indexed climate report."""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question to answer from retrieved document context",
        examples=["How can climate change affect agriculture?"],
    )


class RagGenerateResponse(BaseModel):
    """A grounded answer and the retrieved report pages used as context."""

    answer: str
    sources: list[str] = Field(default_factory=list)
