from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Citation(BaseModel):
    source: str
    section: str
    excerpt: str
    score: float


class AskResponse(BaseModel):
    question: str
    answered: bool
    answer: str
    citations: list[Citation]
    reason: str | None = None


class HealthResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_indexed: int
