from fastapi import FastAPI

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_WORDS,
    DOCUMENTS_DIR,
    MAX_RESULTS,
    MIN_RETRIEVAL_SCORE,
)
from app.knowledge_base import KnowledgeBase, load_markdown_documents
from app.schemas import AskRequest, AskResponse, HealthResponse
from app.service import GroundedAssistant

app = FastAPI(
    title="Grounded Enterprise Knowledge Assistant",
    version="0.1.0",
    description="Local-first document retrieval with citations and abstention.",
)

chunks = load_markdown_documents(
    DOCUMENTS_DIR, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP
)
assistant = GroundedAssistant(
    KnowledgeBase(chunks),
    min_score=MIN_RETRIEVAL_SCORE,
    max_results=MAX_RESULTS,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        documents_loaded=len({chunk.source for chunk in chunks}),
        chunks_indexed=len(chunks),
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return assistant.ask(request.question)
